extern "C" {
#include "audio_ai.h"
#include "mic.h"
}

#include <math.h>
#include <string.h>
#include <new>

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

// 麦克风硬件配置已移至 mic.h / mic.c，此处不再重复定义
// INMP441 接线: BCLK=GPIO17, WS=GPIO18, SD=GPIO16

// ===================== 音频/推理参数 =====================
#define TAG                     "AUDIO_AI"
#define SAMPLE_RATE             16000
#define AUDIO_FRAME_MS          1000    // 每次推理使用1秒音频
#define AUDIO_BUFFER_SAMPLES    (SAMPLE_RATE * AUDIO_FRAME_MS / 1000)

#define FFT_SIZE                512
#define FFT_STEP                256     // 50% 重叠
#define NUM_FRAMES              ((AUDIO_BUFFER_SAMPLES - FFT_SIZE) / FFT_STEP + 1)
#define NUM_MEL_BINS            40      // 必须与 prepare_model.py 中 AUDIO_FEATURE_SIZE 一致
#define FEATURE_2D_SIZE         (NUM_FRAMES * NUM_MEL_BINS)  // 60*40=2400

// RMS 归一化目标 (必须与 train_model.py RMS_TARGET 完全一致)
#define RMS_TARGET              0.03f
#define RMS_EPS                 1e-6f

// === 静音门限与增益上限 ===
// 上次设 0.005 太高：敲门是瞬态信号，1 秒窗口里大量静音段会把整体 RMS 拉到
// 0.003~0.005，连真实敲门都被静音门限挡掉。实测 INMP441 纯底噪 RMS < 0.002
// (峰值 < 50)，有声音时 RMS ≥ 0.003 (峰值 > 100)。门限降到 0.003 精准挡底噪。
#define SILENCE_RMS_THRESHOLD   0.003f
// 增益上限提到 8.0：RMS=0.003 时 gain=10 才能归一化到 0.03，但 10 倍放大底噪
// 会引入失真。8 倍是折中：effective RMS≈0.024，足够模型识别又不过度放大。
#define MAX_NORM_GAIN           8.0f

// 触发阈值 (单次置信度下限 + 连续确认次数)
// 原值 0.60 太高：日志显示大量真实敲门置信度在 0.55~0.59，被当成噪声丢弃。
// 降到 0.50：能抓住边缘敲门，配合连续确认防误报。
#define CONFIDENCE_THRESHOLD    0.50f
#define CONFIRM_COUNT           2       // 连续 N 次同类才触发事件
// 噪声容忍：连续确认中允许插入的"真正噪声/静音"帧数，超过则清零。
// 原值 2 太低：两次敲门间隔常有 3~5 帧噪声/静音。提到 5 给瞬态信号足够恢复时间。
#define NOISE_TOLERANCE         5

// CNN 模型 tensor arena
// 模型实测需 48192 字节工作内存, TFLite Micro 内部对齐再损耗 ~2KB,
// 48KB 刚好卡在临界值导致 AllocateTensors() 失败 (available 仅 47064)。
// 去掉 normalized_audio 后 DRAM 充裕, 给 64KB 确保通过并留足余量。
#define TENSOR_ARENA_SIZE       (64 * 1024)

// 模型类别索引（与训练脚本一致）
#define CLASS_NOISE             0
#define CLASS_KNOCK             1
#define CLASS_DOORBELL          2

// ===================== 静态变量 =====================
static TaskHandle_t s_audio_task_handle = NULL;
static SemaphoreHandle_t s_result_mutex = NULL;

static audio_result_t s_latest_result = {};

// TFLite Micro 对象（使用placement new静态分配，避免堆内存碎片）
static tflite::MicroMutableOpResolver<20> *s_resolver = NULL;
static tflite::MicroInterpreter *s_interpreter = NULL;
static const tflite::Model *s_model = NULL;

static uint8_t s_tensor_arena[TENSOR_ARENA_SIZE] = {0};
static uint8_t s_resolver_buf[sizeof(tflite::MicroMutableOpResolver<20>)] = {0};
static uint8_t s_interpreter_buf[sizeof(tflite::MicroInterpreter)] = {0};

// 音频/特征缓冲区
static int16_t s_audio_buffer[AUDIO_BUFFER_SAMPLES] = {0};
static float   s_fft_buffer[FFT_SIZE * 2] = {0};       // 复数缓冲：实部/虚部交错
static float   s_mel_filterbank[NUM_MEL_BINS][FFT_SIZE / 2 + 1] = {0};
// 2D 时频图特征 (NUM_FRAMES, NUM_MEL_BINS) = (60, 40), 平铺为 1D
static float   s_feature_buffer[FEATURE_2D_SIZE] = {0};

// 连续确认状态 (用于防误报)
static int      s_last_class = -1;     // 上次预测类别
static int      s_confirm_count = 0;   // 当前类别连续出现次数
static int      s_noise_streak = 0;    // 连续真噪声/静音帧数 (max=NOISE)
static int      s_uncertain_streak = 0; // 连续不确定帧数 (max=KNOCK/DOORBELL 但置信度不足)
static int      s_inference_count = 0; // 总推理次数 (用于调试日志频率控制)
static bool     s_is_silence = false;  // 当前帧是否为静音 (RMS 低于门限)

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

    // === RMS 归一化 (与 train_model.py 完全一致) ===
    float sum_sq = 0.0f;
    for (int i = 0; i < AUDIO_BUFFER_SAMPLES; i++) {
        float s = (float)s_audio_buffer[i] / 32768.0f;
        sum_sq += s * s;
    }
    float rms = sqrtf(sum_sq / AUDIO_BUFFER_SAMPLES) + RMS_EPS;

    // === 静音门限: RMS 低于阈值时标记静音, run_inference 会直接返回 NONE ===
    // 这是解决"静音被放大 15 倍后误判为敲门"的核心防线。
    // 静音时仍计算特征 (日志需要), 但 gain 取 1.0 避免放大底噪产生误导性特征。
    s_is_silence = (rms < SILENCE_RMS_THRESHOLD);

    float gain = RMS_TARGET / rms;
    if (gain > MAX_NORM_GAIN) gain = MAX_NORM_GAIN;  // 增益上限, 防止弱信号过度放大

    if (s_inference_count % 10 == 0) {
        ESP_LOGI(TAG, "[诊断] 原始RMS=%.5f, 归一化增益=%.2f, 输入幅值范围=[%d,%d]%s",
                 rms, gain,
                 s_audio_buffer[0], s_audio_buffer[AUDIO_BUFFER_SAMPLES/2],
                 s_is_silence ? " [静音-跳过判决]" : "");
    }

    // 归一化直接在 FFT 填充时完成 (int16->[-1,1]->*gain->削顶->*窗),
    // 不再单独分配 normalized_audio 数组 (原 float[16000] 白占 62.5KB DRAM,
    // 是导致堆内存吃紧、arena 临界不足的元凶之一)。数学上完全等价。
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int offset = frame * FFT_STEP;

        for (int i = 0; i < FFT_SIZE; i++) {
            float w = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (FFT_SIZE - 1));
            float s = ((float)s_audio_buffer[offset + i] / 32768.0f) * gain;
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            s_fft_buffer[2 * i] = s * w;
            s_fft_buffer[2 * i + 1] = 0.0f;
        }

        dsps_fft2r_fc32(s_fft_buffer, FFT_SIZE);
        dsps_bit_rev_fc32(s_fft_buffer, FFT_SIZE);
        for (int i = 0; i < FFT_SIZE * 2; i++) {
            s_fft_buffer[i] /= (float)FFT_SIZE;
        }

        // 当前帧的 Mel 特征写入 s_feature_buffer[frame * NUM_MEL_BINS .. ]
        float *frame_feat = &s_feature_buffer[frame * NUM_MEL_BINS];
        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float mel_val = 0.0f;
            for (int k = 0; k < num_freq_bins; k++) {
                float real = s_fft_buffer[2 * k];
                float imag = s_fft_buffer[2 * k + 1];
                float power = real * real + imag * imag;
                mel_val += power * s_mel_filterbank[m][k];
            }
            frame_feat[m] = logf(mel_val + 1e-6f);
        }
    }

    if (s_inference_count % 10 == 0) {
        float fmin = s_feature_buffer[0], fmax = s_feature_buffer[0];
        float fsum = 0.0f;
        for (int i = 0; i < FEATURE_2D_SIZE; i++) {
            if (s_feature_buffer[i] < fmin) fmin = s_feature_buffer[i];
            if (s_feature_buffer[i] > fmax) fmax = s_feature_buffer[i];
            fsum += s_feature_buffer[i];
        }
        ESP_LOGI(TAG, "[诊断] 特征值范围=[%.2f, %.2f], 均值=%.2f (shape=%dx%d)",
                 fmin, fmax, fsum / FEATURE_2D_SIZE, NUM_FRAMES, NUM_MEL_BINS);
    }
}

static esp_err_t run_inference(audio_result_t *out)
{
    if (!s_interpreter) {
        return ESP_ERR_INVALID_STATE;
    }

    // === 静音短路: RMS 低于门限时直接判 NONE, 不送模型推理 ===
    // 静音帧算作"真噪声"(A类)，走 noise_streak 逻辑。
    // 只有连续 NOISE_TOLERANCE 帧静音才清零，避免敲门被单帧静音打断。
    if (s_is_silence) {
        out->valid = true;
        out->event = AUDIO_EVENT_NONE;
        out->confidence = 0.0f;
        out->raw_probs[0] = 1.0f;  // 视为噪声
        out->raw_probs[1] = 0.0f;
        out->raw_probs[2] = 0.0f;

        s_noise_streak++;
        s_uncertain_streak = 0;  // 静音不算不确定
        if (s_noise_streak > NOISE_TOLERANCE) {
            s_last_class = CLASS_NOISE;
            s_confirm_count = 0;
        }
        return ESP_OK;
    }

    TfLiteTensor *input = s_interpreter->input(0);
    if (input == nullptr) {
        ESP_LOGE(TAG, "获取输入张量失败");
        return ESP_ERR_INVALID_STATE;
    }

    // 填充输入：支持 float32 与 int8 量化模型
    // CNN 输入形状 (1, NUM_FRAMES, NUM_MEL_BINS) = (1, 60, 40)
    int input_size = input->bytes / (input->type == kTfLiteFloat32 ? 4 : 1);
    if (input_size != FEATURE_2D_SIZE) {
        ESP_LOGE(TAG, "模型输入大小 (%d) != 特征大小 (%d), 请检查模型结构",
                 input_size, FEATURE_2D_SIZE);
        return ESP_ERR_INVALID_STATE;
    }

    if (input->type == kTfLiteFloat32) {
        for (int i = 0; i < FEATURE_2D_SIZE; i++) {
            input->data.f[i] = s_feature_buffer[i];
        }
    } else if (input->type == kTfLiteInt8) {
        float scale = input->params.scale;
        int zero_point = input->params.zero_point;
        for (int i = 0; i < FEATURE_2D_SIZE; i++) {
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

    // === 连续确认机制 (三态分离，修复核心逻辑BUG) ===
    // 原逻辑: max_idx==NOISE 或 confidence<阈值 → 统一当噪声帧处理。
    // BUG: knock=0.58(max类是敲门但置信度不到0.60)被当成噪声帧，
    //      noise_streak 涨到3就清零 confirm_count，导致敲门永远凑不够2次。
    //
    // 修复: 分三种状态分别处理：
    //   A. 真噪声 (max=NOISE)         → noise_streak++，超容忍才清零
    //   B. 边缘敲门/门铃 (max=KNOCK/DOORBELL 但置信度在0.50~阈值之间)
    //      → 不递增confirm，不递增noise_streak，走单独的"不确定"容忍
    //   C. 确信敲门/门铃 (max=KNOCK/DOORBELL 且 ≥阈值)
    //      → noise_streak=0, confirm_count++
    if (max_idx == CLASS_NOISE) {
        // A类: 真噪声帧 — 累加噪声连胜
        s_noise_streak++;
        if (s_noise_streak > NOISE_TOLERANCE) {
            s_last_class = CLASS_NOISE;
            s_confirm_count = 0;
        }
    } else if (out->confidence < CONFIDENCE_THRESHOLD) {
        // B类: max是敲门/门铃但置信度不到阈值 — 不确定帧
        // 不当噪声处理！不递增 noise_streak，但也不递增 confirm_count。
        // 用独立的 uncertain_streak 跟踪，容忍度更高（因为这不是噪声，只是模型不确定）。
        s_uncertain_streak++;
        if (s_uncertain_streak > (NOISE_TOLERANCE * 2)) {
            // 长时间不确定才清零
            s_last_class = -1;
            s_confirm_count = 0;
            s_uncertain_streak = 0;
        }
    } else {
        // C类: 确信的敲门/门铃帧
        s_noise_streak = 0;
        s_uncertain_streak = 0;
        if (max_idx == s_last_class) {
            s_confirm_count++;
        } else {
            s_last_class = max_idx;
            s_confirm_count = 1;
        }
    }

    // 触发条件: 非 noise 类 + 置信度达标 + 连续确认
    if (max_idx == CLASS_NOISE) {
        out->event = AUDIO_EVENT_NONE;
    } else if (out->confidence < CONFIDENCE_THRESHOLD) {
        out->event = AUDIO_EVENT_NONE;
    } else if (s_confirm_count < CONFIRM_COUNT) {
        // 置信度达标但还没连续确认够次数, 暂不触发
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
    int32_t read_idx = 0;
    int16_t temp_buf[256];

    ESP_LOGI(TAG, "音频推理任务已启动");

    // 调试：前5次读取打印采样值，验证麦克风是否正常
    int debug_count = 0;

    while (1) {
        // 通过 mic 模块读取音频（16位样本）
        int samples_read = mic_read_samples(temp_buf, 256, pdMS_TO_TICKS(100));

        if (samples_read > 0) {
            // 调试：打印前5次读取的采样值
            if (debug_count < 5) {
                int16_t min_val = 0, max_val = 0;
                for (int i = 0; i < samples_read; i++) {
                    if (i == 0 || temp_buf[i] < min_val) min_val = temp_buf[i];
                    if (i == 0 || temp_buf[i] > max_val) max_val = temp_buf[i];
                }
                ESP_LOGI(TAG, "调试[%d]: 读了%d个样本, 最小值=%d, 最大值=%d (对麦克风说话时最大值应该变化)",
                         debug_count, samples_read, min_val, max_val);
                debug_count++;
            }

            // 写入环形缓冲区
            for (int i = 0; i < samples_read; i++) {
                s_audio_buffer[read_idx] = temp_buf[i];
                read_idx++;
                if (read_idx >= AUDIO_BUFFER_SAMPLES) {
                    read_idx = 0;

                    // 缓冲区满 -> 执行一次推理
                    audio_result_t result = {};
                    s_inference_count++;
                    extract_features();
                    if (run_inference(&result) == ESP_OK) {
                        result.inference_seq = s_inference_count;  // 打上本次推理的序列号
                        xSemaphoreTake(s_result_mutex, portMAX_DELAY);
                        memcpy(&s_latest_result, &result, sizeof(audio_result_t));
                        xSemaphoreGive(s_result_mutex);

                        // 日志格式区分"静音短路"和"真实推理"，避免用户混淆
                        if (s_is_silence) {
                            ESP_LOGI(TAG, "[静音] 跳过推理 (噪声连胜%d/容忍%d)",
                                     s_noise_streak, NOISE_TOLERANCE);
                        } else {
                            ESP_LOGI(TAG, "推理: %s %.2f (噪=%.2f 敲=%.2f 铃=%.2f) [确认%d 噪声%d/不确定%d]",
                                     result.event == AUDIO_EVENT_NONE ? "无" :
                                     result.event == AUDIO_EVENT_KNOCK ? "敲门" :
                                     result.event == AUDIO_EVENT_DOORBELL ? "门铃" : "未知",
                                     result.confidence,
                                     result.raw_probs[0], result.raw_probs[1], result.raw_probs[2],
                                     s_confirm_count, s_noise_streak, s_uncertain_streak);
                        }
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

    // 初始化麦克风（使用 mic.c 模块，基于组员验证过的配置）
    ret = mic_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "麦克风初始化失败: %s", esp_err_to_name(ret));
        return ret;
    }

    // 加载TFLite模型
    s_model = tflite::GetModel(g_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "模型版本不匹配: %d vs %d", s_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }

    s_resolver = new (s_resolver_buf) tflite::MicroMutableOpResolver<20>();
    // === CNN + BatchNorm INT8 量化所需的全部算子 ===
    // 卷积/池化
    s_resolver->AddConv2D();
    s_resolver->AddMaxPool2D();
    s_resolver->AddAveragePool2D();   // GAP 有时映射为此
    // 全连接/分类
    s_resolver->AddFullyConnected();
    s_resolver->AddSoftmax();
    // 激活
    s_resolver->AddRelu();
    s_resolver->AddLogistic();        // sigmoid 预留
    // BatchNormalization 量化后 = MUL + ADD (scale*x + bias)
    s_resolver->AddMul();
    s_resolver->AddAdd();
    s_resolver->AddSub();
    // 形状/重塑
    s_resolver->AddReshape();
    s_resolver->AddShape();
    s_resolver->AddSqueeze();
    s_resolver->AddStridedSlice();
    s_resolver->AddMean();            // GlobalAveragePooling -> Mean
    // 量化/反量化
    s_resolver->AddQuantize();
    s_resolver->AddDequantize();
    // 预留 (BN 优化的边缘情况)
    s_resolver->AddPad();
    s_resolver->AddPack();
    s_resolver->AddConcatenation();

    s_interpreter = new (s_interpreter_buf) tflite::MicroInterpreter(
        s_model, *s_resolver, s_tensor_arena, TENSOR_ARENA_SIZE);

    TfLiteStatus alloc_status = s_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "张量分配失败: %d", alloc_status);
        return ESP_FAIL;
    }

    TfLiteTensor *input = s_interpreter->input(0);
    ESP_LOGI(TAG, "模型输入: dims=[");
    for (int d = 0; d < input->dims->size; d++) {
        printf("%d%s", input->dims->data[d], d < input->dims->size - 1 ? "," : "");
    }
    printf("], 类型=%d, 字节=%d\n", input->type, input->bytes);

    // 验证输入形状: 应为 (1, 60, 40) 或 (1, 60, 40, 1)
    int expected_last_dim = NUM_MEL_BINS;
    if (input->dims->data[input->dims->size - 1] != expected_last_dim) {
        ESP_LOGW(TAG, "模型最后一维 (%d) != NUM_MEL_BINS (%d), 请检查模型",
                 input->dims->data[input->dims->size - 1], NUM_MEL_BINS);
    }

    ESP_LOGI(TAG, "音频AI初始化完成。FFT点数=%d, Mel滤波组=%d, 推理窗口=%dms",
             FFT_SIZE, NUM_MEL_BINS, AUDIO_FRAME_MS);
    return ESP_OK;
}

esp_err_t audio_ai_start(void)
{
    if (!s_interpreter) {
        return ESP_ERR_INVALID_STATE;
    }

    // 麦克风在 mic_init() 中已启动，无需额外 enable

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
    // 麦克风卸载由系统管理，此处不处理
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

    // 通过 mic 模块读取1秒音频
    int total_read = 0;
    while (total_read < AUDIO_BUFFER_SAMPLES) {
        int n = mic_read_samples(&s_audio_buffer[total_read], AUDIO_BUFFER_SAMPLES - total_read, pdMS_TO_TICKS(100));
        if (n <= 0) break;
        total_read += n;
    }

    extract_features();
    esp_err_t ret = run_inference(result);
    result->inference_seq = s_inference_count;

    xSemaphoreTake(s_result_mutex, portMAX_DELAY);
    memcpy(&s_latest_result, result, sizeof(audio_result_t));
    xSemaphoreGive(s_result_mutex);

    return ret;
}
