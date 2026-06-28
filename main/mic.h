// mic.h - INMP441 麦克风驱动（基于组员验证过的旧版 I2S API）
#ifndef MIC_H
#define MIC_H

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 初始化 INMP441 麦克风（I2S）
 * @return ESP_OK 或错误码
 */
esp_err_t mic_init(void);

/**
 * 批量读取音频样本（16位）
 * @param buffer 输出缓冲区
 * @param num_samples 要读取的样本数
 * @param timeout 超时时间
 * @return 实际读取的样本数
 */
int mic_read_samples(int16_t *buffer, int num_samples, TickType_t timeout);

#ifdef __cplusplus
}
#endif

#endif
