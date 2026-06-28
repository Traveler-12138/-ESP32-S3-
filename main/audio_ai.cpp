extern "C" {
#include "audio_ai.h"
}

#include <math.h>
#include <string.h>
#include <new>

#include "driver/i2s_std.h"
#include "esp_dsp.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===================== 硬件连接配置（根据实际接线修改） =====================
#define AUDIO_I2S_PORT          I2S_NUM_0
#define AUDIO_I2S_BCLK_GPIO     GPIO_NUM_4
#define AUDIO_I2S_WS_GPIO       GPIO_NUM_5
#define AUDIO_I2S_DIN_GPIO      GPIO_NUM_6
// 如果麦克风是PDM接口（如MP34DT01），请改用 driver/i2s_pdm.h 并调整初始化。

// ===================== 音频/推理参数 =====================
#define TAG                     "AUDIO_AI"
#define SAMPLE_RATE             16000
#define AUDIO_FRAME_MS          1000    // 每次推理使用1秒音频
#define AUDIO_BUFFER_SAMPLES    (SAMPLE_RATE * AUDIO_FRAME_MS / 1000)

#define FFT_SIZE                512
#define FFT_STEP                256     // 50% 重叠
#define NUM_FRAMES              ((AUDIO_BUFFER_SAMPLES - FFT_SIZE) / FFT_STEP + 1)
#define NUM_MEL_BINS            40      // 必须与 prepare_model.py 中 AUDIO_FEATURE_SIZE 一致

#define TENSOR_ARENA_SIZE       (24 * 1024)

// 模型类别索引（与训练脚本一致）
#define CLASS_NOISE             0
#define CLASS_KNOCK             1
#define CLASS_DOORBELL          2

// ===================== 静态变量 =====================
static i2s_chan_handle_t s_rx_chan = NULL;
static TaskHandle_t s_audio_task_handle = NULL;
static SemaphoreHandle_t s_result_mutex = NULL;

static audio_result_t s_latest_result = {};

// TFLite Micro 对象（使用placement new静态分配，避免堆内存碎片）
static tflite::MicroMutableOpResolver<10> *s_resolver = NULL;
static tflite::MicroInterpreter *s_interpreter = NULL;
static const tflite::Model *s_model = NULL;

static uint8_t s_tensor_arena[TENSOR_ARENA_SIZE] = {0};
static uint8_t s_resolver_buf[sizeof(tflite::MicroMutableOpResolver<10>)] = {0};
static uint8_t s_interpreter_buf[sizeof(tflite::MicroInterpreter)] = {0};

// 音频/特征缓冲区
static int16_t s_audio_buffer[AUDIO_BUFFER_SAMPLES] = {0};
static float   s_fft_buffer[FFT_SIZE * 2] = {0};       // 复数缓冲：实部/虚部交错
static float   s_mel_filterbank[NUM_MEL_BINS][FFT_SIZE / 2 + 1] = {0};
static float   s_feature_buffer[NUM_MEL_BINS] = {0};

// ===================== 工具函数 =====================
static inline float hz_to_mel(float f)
{
    return 2595.0f * log10f(1.0f + f / 700.0f);
}

static inline float mel_to_hz(float m)
{
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

static void init_mel_filterbank(void)
{
    float mel_low = hz_to_mel(0.0f);
    float mel_high = hz_to_mel((float)SAMPLE_RATE / 2.0f);
    float mel_points[NUM_MEL_BINS + 2];
    float freq_points[NUM_MEL_BINS + 2];

    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        mel_points[i] = mel_low + i * (mel_high - mel_low) / (NUM_MEL_BINS + 1);
        freq_points[i] = mel_to_hz(mel_points[i]);
    }

    int num_freq_bins = FFT_SIZE / 2 + 1;
    float bin_freq_step = (float)SAMPLE_RATE / (float)FFT_SIZE;

    for (int m = 1; m <= NUM_MEL_BINS; m++) {
        for (int k = 0; k < num_freq_bins; k++) {
            float freq = k * bin_freq_step;
            float weight = 0.0f;
            if (freq >= freq_points[m - 1] && freq <= freq_points[m]) {
                weight = (freq - freq_points[m - 1]) / (freq_points[m] - freq_points[m - 1]);
            } else if (freq >= freq_points[m] && freq <= freq_points[m + 1]) {
                weight = (freq_points[m + 1] - freq) / (freq_points[m + 1] - freq_points[m]);
            }
            s_mel_filterbank[m - 1][k] = weight;
        }
    }
}

static void extract_features(void)
{
    memset(s_feature_buffer, 0, sizeof(s_feature_buffer));
    int num_freq_bins = FFT_SIZE / 2 + 1;

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int offset = frame * FFT_STEP;

        // 1. 加载到FFT缓冲区并应用汉明窗
        for (int i = 0; i < FFT_SIZE; i++) {
            float w = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (FFT_SIZE - 1));
            float sample = (float)s_audio_buffer[offset + i] / 32768.0f;
            s_fft_buffer[2 * i] = sample * w;
            s_fft_buffer[2 * i + 1] = 0.0f;
        }

        // 2. 使用ESP-DSP做复数FFT（输出实部/虚部交错）
        dsps_fft2r_fc32(s_fft_buffer, FFT_SIZE);
        dsps_bit_rev_fc32(s_fft_buffer, FFT_SIZE);
        // dsps_fft2r_fc32 未归一化，需除以FFT_SIZE
        for (int i = 0; i < FFT_SIZE * 2; i++) {
            s_fft_buffer[i] /= (float)FFT_SIZE;
        }

        // 3. 计算功率谱并应用Mel滤波器
        for (int k = 0; k < num_freq_bins; k++) {
            float real = s_fft_buffer[2 * k];
            float imag = s_fft_buffer[2 * k + 1];
            float power = real * real + imag * imag;
            for (int m = 0; m < NUM_MEL_BINS; m++) {
                s_feature_buffer[m] += power * s_mel_filterbank[m][k];
            }
        }
    }

    // 4. 对数压缩并平均
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        s_feature_buffer[m] /= (float)NUM_FRAMES;
        s_feature_buffer[m] = logf(s_feature_buffer[m] + 1e-6f);
    }
}

static esp_err_t run_inference(audio_result_t *out)
{
    if (!s_interpreter) {
        return ESP_ERR_INVALID_STATE;
    }

    TfLiteTensor *input = s_interpreter->input(0);
    if (input == nullptr) {
        ESP_LOGE(TAG, "获取输入张量失败");
        return ESP_ERR_INVALID_STATE;
    }

    // 填充输入：支持float32与int8量化模型
    if (input->type == kTfLiteFloat32) {
        for (int i = 0; i < NUM_MEL_BINS; i++) {
            input->data.f[i] = s_feature_buffer[i];
        }
    } else if (input->type == kTfLiteInt8) {
        float scale = input->params.scale;
        int zero_point = input->params.zero_point;
        for (int i = 0; i < NUM_MEL_BINS; i++) {
            int32_t val = (int32_t)(s_feature_buffer[i] / scale) + zero_point;
            if (val < -128) val = -128;
            if (val > 127) val = 127;
            input->data.int8[i] = (int8_t)val;
        }
    } else {
        ESP_LOGE(TAG, "不支持的输入张量类型: %d", input->type);
        return ESP_ERR_NOT_SUPPORTED;
    }

    TfLiteStatus invoke_status = s_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "推理失败: %d", invoke_status);
        return ESP_FAIL;
    }

    TfLiteTensor *output = s_interpreter->output(0);
    if (output == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }

    float probs[3] = {0.0f};
    if (output->type == kTfLiteFloat32) {
        for (int i = 0; i < 3; i++) {
            probs[i] = output->data.f[i];
        }
    } else if (output->type == kTfLiteInt8) {
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        for (int i = 0; i < 3; i++) {
            probs[i] = (output->data.int8[i] - zero_point) * scale;
        }
    } else {
        return ESP_ERR_NOT_SUPPORTED;
    }

    // 找到最大概率类别
    int max_idx = 0;
    for (int i = 1; i < 3; i++) {
        if (probs[i] > probs[max_idx]) max_idx = i;
    }

    out->valid = true;
    out->confidence = probs[max_idx];
    memcpy(out->raw_probs, probs, sizeof(probs));

    if (max_idx == CLASS_NOISE || out->confidence < 0.5f) {
        out->event = AUDIO_EVENT_NONE;
    } else if (max_idx == CLASS_KNOCK) {
        out->event = AUDIO_EVENT_KNOCK;
    } else if (max_idx == CLASS_DOORBELL) {
        out->event = AUDIO_EVENT_DOORBELL;
    } else {
        out->event = AUDIO_EVENT_UNKNOWN;
    }

    return ESP_OK;
}

static void audio_task(void *pvParameters)
{
    size_t bytes_read = 0;
    int32_t read_idx = 0;
    int16_t temp_buf[256];

    ESP_LOGI(TAG, "音频推理任务已启动");

    while (1) {
        // 读取一小段音频到临时缓冲区
        if (i2s_channel_read(s_rx_chan, temp_buf, sizeof(temp_buf), &bytes_read, portMAX_DELAY) == ESP_OK) {
            int32_t samples_read = bytes_read / sizeof(int16_t);

            // 写入环形缓冲区
            for (int i = 0; i < samples_read; i++) {
                s_audio_buffer[read_idx] = temp_buf[i];
                read_idx++;
                if (read_idx >= AUDIO_BUFFER_SAMPLES) {
                    read_idx = 0;

                    // 缓冲区满 -> 执行一次推理
                    audio_result_t result = {};
                    extract_features();
                    if (run_inference(&result) == ESP_OK) {
                        xSemaphoreTake(s_result_mutex, portMAX_DELAY);
                        memcpy(&s_latest_result, &result, sizeof(audio_result_t));
                        xSemaphoreGive(s_result_mutex);

                        ESP_LOGI(TAG, "推理结果: %s, 置信度=%.2f (噪声=%.2f 敲门=%.2f 门铃=%.2f)",
                                 result.event == AUDIO_EVENT_NONE ? "无" :
                                 result.event == AUDIO_EVENT_KNOCK ? "敲门" :
                                 result.event == AUDIO_EVENT_DOORBELL ? "门铃" : "未知",
                                 result.confidence,
                                 result.raw_probs[0], result.raw_probs[1], result.raw_probs[2]);
                    }
                }
            }
        }
    }
}

// ===================== 接口实现 =====================
esp_err_t audio_ai_init(void)
{
    if (s_result_mutex == NULL) {
        s_result_mutex = xSemaphoreCreateMutex();
    }

    // 初始化ESP-DSP
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, FFT_SIZE);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "ESP-DSP FFT 初始化失败: %s", esp_err_to_name(ret));
        return ret;
    }

    // 初始化Mel滤波器组
    init_mel_filterbank();

    // 初始化I2S标准模式（接收）
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(AUDIO_I2S_PORT, I2S_ROLE_MASTER);
    ret = i2s_new_channel(&chan_cfg, NULL, &s_rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S 通道创建失败: %s", esp_err_to_name(ret));
        return ret;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = AUDIO_I2S_BCLK_GPIO,
            .ws = AUDIO_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = AUDIO_I2S_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    ret = i2s_channel_init_std_mode(s_rx_chan, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S 标准模式初始化失败: %s", esp_err_to_name(ret));
        i2s_del_channel(s_rx_chan);
        s_rx_chan = NULL;
        return ret;
    }

    // 加载TFLite模型
    s_model = tflite::GetModel(g_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "模型版本不匹配: %d vs %d", s_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }

    s_resolver = new (s_resolver_buf) tflite::MicroMutableOpResolver<10>();
    s_resolver->AddFullyConnected();
    s_resolver->AddRelu();
    s_resolver->AddSoftmax();
    s_resolver->AddQuantize();
    s_resolver->AddDequantize();
    s_resolver->AddReshape();

    s_interpreter = new (s_interpreter_buf) tflite::MicroInterpreter(
        s_model, *s_resolver, s_tensor_arena, TENSOR_ARENA_SIZE);

    TfLiteStatus alloc_status = s_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "张量分配失败: %d", alloc_status);
        return ESP_FAIL;
    }

    TfLiteTensor *input = s_interpreter->input(0);
    ESP_LOGI(TAG, "模型输入: 维度=%d, 类型=%d, 字节=%d",
             input->dims->size, input->type, input->bytes);
    if (input->dims->data[input->dims->size - 1] != NUM_MEL_BINS) {
        ESP_LOGW(TAG, "模型输入特征数 (%d) != NUM_MEL_BINS (%d)，请检查 prepare_model.py",
                 input->dims->data[input->dims->size - 1], NUM_MEL_BINS);
    }

    ESP_LOGI(TAG, "音频AI初始化完成。FFT点数=%d, Mel滤波组=%d, 推理窗口=%dms",
             FFT_SIZE, NUM_MEL_BINS, AUDIO_FRAME_MS);
    return ESP_OK;
}

esp_err_t audio_ai_start(void)
{
    if (!s_rx_chan || !s_interpreter) {
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t ret = i2s_channel_enable(s_rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S 通道使能失败: %s", esp_err_to_name(ret));
        return ret;
    }

    if (s_audio_task_handle == NULL) {
        BaseType_t xRet = xTaskCreatePinnedToCore(
            audio_task, "audio_task", 8192, NULL, 5, &s_audio_task_handle, 1);
        if (xRet != pdPASS) {
            return ESP_ERR_NO_MEM;
        }
    }

    return ESP_OK;
}

esp_err_t audio_ai_stop(void)
{
    if (s_audio_task_handle) {
        vTaskDelete(s_audio_task_handle);
        s_audio_task_handle = NULL;
    }
    if (s_rx_chan) {
        i2s_channel_disable(s_rx_chan);
    }
    return ESP_OK;
}

esp_err_t audio_ai_get_latest_result(audio_result_t *result)
{
    if (!result) return ESP_ERR_INVALID_ARG;

    xSemaphoreTake(s_result_mutex, portMAX_DELAY);
    memcpy(result, &s_latest_result, sizeof(audio_result_t));
    xSemaphoreGive(s_result_mutex);
    return ESP_OK;
}

esp_err_t audio_ai_run_once(audio_result_t *result)
{
    if (!s_interpreter) return ESP_ERR_INVALID_STATE;

    size_t bytes_read = 0;
    int16_t *temp_buf = (int16_t *)heap_caps_malloc(AUDIO_BUFFER_SAMPLES * sizeof(int16_t), MALLOC_CAP_8BIT);
    if (!temp_buf) return ESP_ERR_NO_MEM;

    i2s_channel_read(s_rx_chan, temp_buf, AUDIO_BUFFER_SAMPLES * sizeof(int16_t), &bytes_read, portMAX_DELAY);
    memcpy(s_audio_buffer, temp_buf, AUDIO_BUFFER_SAMPLES * sizeof(int16_t));
    heap_caps_free(temp_buf);

    extract_features();
    esp_err_t ret = run_inference(result);

    xSemaphoreTake(s_result_mutex, portMAX_DELAY);
    memcpy(&s_latest_result, result, sizeof(audio_result_t));
    xSemaphoreGive(s_result_mutex);

    return ret;
}
