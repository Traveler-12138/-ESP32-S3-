#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file audio_ai.h
 * @brief 边缘AI音频分类接口
 *
 * 负责从MEMS麦克风采集音频，提取特征并运行TFLite Micro模型，
 * 本地识别敲门声(knock)与门铃声(doorbell)。
 */

typedef enum {
    AUDIO_EVENT_NONE = 0,      // 无有效声音事件
    AUDIO_EVENT_KNOCK,         // 敲门声
    AUDIO_EVENT_DOORBELL,      // 门铃声/传统门铃
    AUDIO_EVENT_UNKNOWN      // 检测到声音但类别未知
} audio_event_t;

typedef struct {
    audio_event_t event;       // 识别事件
    float confidence;          // 最高类置信度 [0, 1]
    bool valid;                // 是否完成一次有效推理
    float raw_probs[3];        // 三类原始概率 [noise, knock, doorbell]
} audio_result_t;

/**
 * @brief 初始化音频采集与推理引擎
 *
 * 配置I2S麦克风、加载TFLite模型、分配tensor arena。
 */
esp_err_t audio_ai_init(void);

/**
 * @brief 启动后台音频采集与推理任务
 */
esp_err_t audio_ai_start(void);

/**
 * @brief 停止音频任务
 */
esp_err_t audio_ai_stop(void);

/**
 * @brief 获取最近一次推理结果（线程安全）
 */
esp_err_t audio_ai_get_latest_result(audio_result_t *result);

/**
 * @brief 手动触发一次推理（调试用）
 */
esp_err_t audio_ai_run_once(audio_result_t *result);

#ifdef __cplusplus
}
#endif
