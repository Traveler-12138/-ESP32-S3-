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
 * 输入通道：
 *   - 物理按键（门外门铃按钮，由 door_button 模块驱动，含消抖/边沿检测）
 *   - PIR 人体红外（由 pir 模块驱动，轮询）
 *   - 音频AI识别（由 audio_ai 模块提供，通过 update_audio 外部传入）
 *
 * 判决逻辑：
 *   - 任意单通道在时间窗口内触发 -> SINGLE_CHANNEL（仅日志记录，不提醒）
 *   - 任意两通道在时间窗口内同时有效 -> VISITOR（正式来访提醒，5秒冷却期去重）
 *   - 紧急呼叫（来自另一块板子的远端通信）-> EMERGENCY（最高优先级）
 *
 * 紧急按钮说明：
 *   紧急按钮不接在本板 GPIO 上，而是安装在另一块板子上，
 *   通过通信（ESP-NOW/MQTT/串口等）将按下事件传给本板，
 *   通信层收到后调用 event_fusion_trigger_emergency() 即可。
 */

typedef enum {
    FUSION_EVENT_NONE = 0,
    FUSION_EVENT_SINGLE_CHANNEL,   // 单通道触发（仅日志）
    FUSION_EVENT_VISITOR,          // 至少两通道触发 -> 正式来访提醒
    FUSION_EVENT_EMERGENCY         // 室内紧急呼叫按钮（远端通信传入，最高优先级）
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
 * @brief 初始化融合模块（内部会初始化按键与PIR驱动）
 */
esp_err_t event_fusion_init(void);

/**
 * @brief 更新音频AI通道状态（由外部周期性调用）
 * @param audio_triggered 音频是否识别到敲门/门铃
 */
void event_fusion_update_audio(bool audio_triggered);

/**
 * @brief 触发紧急呼叫（供远端通信层调用）
 *
 * 紧急按钮安装在另一块板子上，当该板子通过通信告知"紧急按钮按下"时，
 * 通信处理代码调用本接口，融合模块会在后续判决中输出 EMERGENCY 事件。
 */
void event_fusion_trigger_emergency(void);

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
 * @brief 获取当前按键状态（调试用，保持窗口内有效）
 */
bool event_fusion_get_button_state(void);

/**
 * @brief 获取当前PIR状态（调试用，保持窗口内有效）
 */
bool event_fusion_get_pir_state(void);

#ifdef __cplusplus
}
#endif
