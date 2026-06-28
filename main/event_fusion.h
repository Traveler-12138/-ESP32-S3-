#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file event_fusion.h
 * @brief 三通道感知融合判决模块
 *
 * 输入：物理按键、PIR人体红外、音频AI识别结果。
 * 逻辑：任意两通道在指定时间窗口内同时有效，才触发正式来访提醒。
 *       单通道触发仅作为日志/低优先级事件。
 */

typedef enum {
    FUSION_EVENT_NONE = 0,
    FUSION_EVENT_SINGLE_CHANNEL,   // 单通道触发（仅日志）
    FUSION_EVENT_VISITOR,          // 至少两通道触发 -> 正式来访提醒
    FUSION_EVENT_EMERGENCY         // 室内紧急呼叫按钮（直接高优先级）
} fusion_event_t;

typedef struct {
    fusion_event_t event;          // 融合判决结果
    bool button_active;            // 按键有效
    bool pir_active;               // PIR有效
    bool audio_active;             // 音频有效
    bool emergency_active;         // 紧急呼叫有效
    uint32_t timestamp_ms;         // 触发时间戳
} fusion_result_t;

/**
 * @brief 初始化GPIO、中断与融合状态
 */
esp_err_t event_fusion_init(void);

/**
 * @brief 更新音频AI通道状态（由外部周期性调用）
 * @param audio_triggered 音频是否识别到敲门/门铃
 */
void event_fusion_update_audio(bool audio_triggered);

/**
 * @brief 执行一次融合判决
 * @param result 输出结果
 */
esp_err_t event_fusion_decide(fusion_result_t *result);

/**
 * @brief 获取当前紧急呼叫状态（调试用）
 */
bool event_fusion_get_emergency_state(void);

/**
 * @brief 获取当前按键状态（调试用）
 */
bool event_fusion_get_button_state(void);

/**
 * @brief 获取当前PIR状态（调试用）
 */
bool event_fusion_get_pir_state(void);

#ifdef __cplusplus
}
#endif
