// mic.c - INMP441 麦克风驱动
// 使用组员验证过的旧版 I2S API（driver/i2s.h）
// 经实测，INMP441 在 ESP-IDF v5.x 新版 I2S API 下数据会错位，
// 旧版 API 配置与组员代码完全一致，已验证可正常采集。
#include <stdio.h>
#include <string.h>
#include "mic.h"
#include "esp_log.h"
#include "driver/i2s.h"
#include "freertos/FreeRTOS.h"
#include "hal/gpio_types.h"

static const char *TAG = "MIC";

// 引脚配置（与组员接线一致）
#define MIC_I2S_PORT       I2S_NUM_0
#define MIC_BCLK_PIN       GPIO_NUM_17   // 位时钟 -> INMP441 SCK
#define MIC_WS_PIN         GPIO_NUM_18   // 左右声道时钟 -> INMP441 WS/LRCK
#define MIC_DOUT_PIN       GPIO_NUM_16   // 数据输入 -> INMP441 SD

// 采样参数
#define MIC_SAMPLE_RATE    16000
#define MIC_DMA_BUF_COUNT  4
#define MIC_DMA_BUF_LEN    1024

static bool s_initialized = false;

esp_err_t mic_init(void)
{
    ESP_LOGI(TAG, "正在初始化 INMP441 麦克风（旧版 I2S API，与组员配置一致）...");

    // I2S 配置（与组员验证过的代码完全相同）
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = MIC_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,  // INMP441 L/R 接 GND -> 左声道
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = MIC_DMA_BUF_COUNT,
        .dma_buf_len = MIC_DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0,
    };

    esp_err_t ret = i2s_driver_install(MIC_I2S_PORT, &i2s_config, 0, NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S 驱动安装失败: %s", esp_err_to_name(ret));
        return ret;
    }

    // 引脚配置
    i2s_pin_config_t pin_config = {
        .bck_io_num   = MIC_BCLK_PIN,
        .ws_io_num    = MIC_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,  // 不使用数据输出
        .data_in_num  = MIC_DOUT_PIN,
    };

    ret = i2s_set_pin(MIC_I2S_PORT, &pin_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S 引脚设置失败: %s", esp_err_to_name(ret));
        return ret;
    }

    s_initialized = true;
    ESP_LOGI(TAG, "麦克风初始化完成。BCLK=GPIO%d, WS=GPIO%d, DIN=GPIO%d, 采样率=%dHz, 左声道",
             MIC_BCLK_PIN, MIC_WS_PIN, MIC_DOUT_PIN, MIC_SAMPLE_RATE);
    return ESP_OK;
}

int mic_read_samples(int16_t *buffer, int num_samples, TickType_t timeout)
{
    if (!s_initialized) {
        ESP_LOGE(TAG, "麦克风未初始化，无法读取");
        return 0;
    }

    size_t bytes_read = 0;
    size_t bytes_to_read = num_samples * sizeof(int16_t);

    esp_err_t ret = i2s_read(MIC_I2S_PORT, buffer, bytes_to_read, &bytes_read, timeout);
    if (ret == ESP_OK) {
        return (int)(bytes_read / sizeof(int16_t));
    }
    return 0;
}
