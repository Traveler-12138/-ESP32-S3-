#include "event_fusion.h"

#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"

#include "door_button.h"
#include "pir.h"

// ===================== 软件参数 =====================
// 三个通道各自的有效保持窗口：任一通道触发后，在此窗口内其状态视为 active，
// 让时间上相近的事件能在同一次 decide() 中相遇，真正实现"时间窗口内两通道"对齐。
#define TAG                     "FUSION"
#define BUTTON_HOLD_MS          3000    // 按键按下后保持有效（访客按一下即记录，等待PIR/音频对齐）
#define PIR_HOLD_MS             2000    // PIR触发后保持有效
#define AUDIO_HOLD_MS           1500    // 音频识别结果保持有效
#define EMERGENCY_HOLD_MS       5000    // 紧急呼叫保持有效（确保上层一定处理到）
#define VISITOR_COOLDOWN_MS     5000    // VISITOR触发后的冷却期：期内两通道仍active也不重复报警

// ===================== 状态结构 =====================
typedef struct {
    bool button_active;
    bool pir_active;
    bool audio_active;
    bool emergency_active;

    uint32_t button_trigger_ms;
    uint32_t pir_trigger_ms;
    uint32_t audio_trigger_ms;
    uint32_t emergency_trigger_ms;
} fusion_state_t;

static fusion_state_t s_state = {0};
static uint32_t s_last_log_ms = 0;
static uint32_t s_last_visitor_ms = 0;   // 上次 VISITOR 触发时间戳（用于冷却期去重）

// ===================== 通道更新 =====================
static void update_button_channel(void)
{
    uint32_t now = esp_timer_get_time() / 1000ULL;
    // 使用组员 door_button 驱动的消抖后状态（低电平有效，true=按下）
    // 按下瞬间打时间戳并进入保持窗口；窗口到期且已松开才失效
    if (door_button_is_pressed()) {
        if (!s_state.button_active) {
            s_state.button_active = true;
            s_state.button_trigger_ms = now;
        }
        // 按住期间持续刷新时间戳，保证松开后才起算保持窗口
        s_state.button_trigger_ms = now;
    } else if (s_state.button_active && (now - s_state.button_trigger_ms > BUTTON_HOLD_MS)) {
        s_state.button_active = false;
    }
}

static void update_pir_channel(void)
{
    uint32_t now = esp_timer_get_time() / 1000ULL;
    if (pir_is_detected()) {
        s_state.pir_trigger_ms = now;
        s_state.pir_active = true;
    } else if (now - s_state.pir_trigger_ms > PIR_HOLD_MS) {
        s_state.pir_active = false;
    }
}

static bool is_audio_active(void)
{
    uint32_t now = esp_timer_get_time() / 1000ULL;
    if (s_state.audio_active && (now - s_state.audio_trigger_ms <= AUDIO_HOLD_MS)) {
        return true;
    }
    s_state.audio_active = false;
    return false;
}

static bool is_emergency_active(void)
{
    uint32_t now = esp_timer_get_time() / 1000ULL;
    if (s_state.emergency_active && (now - s_state.emergency_trigger_ms <= EMERGENCY_HOLD_MS)) {
        return true;
    }
    s_state.emergency_active = false;
    return false;
}

// ===================== 接口实现 =====================
esp_err_t event_fusion_init(void)
{
    // 按键模块（GPIO4 按键，含消抖/边沿/中断）
    ESP_ERROR_CHECK(door_button_init());
    // PIR 模块（GPIO7 轮询）
    ESP_ERROR_CHECK(pir_init());

    memset(&s_state, 0, sizeof(s_state));
    ESP_LOGI(TAG, "融合模块初始化完成（按键+PIR由组员驱动接入，紧急按钮预留远端通信接口）");
    return ESP_OK;
}

void event_fusion_update_audio(bool audio_triggered)
{
    // 仅在 audio_active 从 false 跳变为 true 时设置触发时间戳。
    // main 循环每 200ms 查询、推理约 1.4s 一次，若每次 true 都刷新时间戳，
    // 保持窗口会被反复刷新永不过期。改为首次触发起算，到期自然过期。
    if (audio_triggered && !s_state.audio_active) {
        s_state.audio_active = true;
        s_state.audio_trigger_ms = esp_timer_get_time() / 1000ULL;
    }
}

void event_fusion_trigger_emergency(void)
{
    // 供远端通信层（ESP-NOW/MQTT/串口）收到紧急按钮消息后调用。
    // 本板不接紧急按钮 GPIO，紧急按钮在另一块板子上。
    s_state.emergency_active = true;
    s_state.emergency_trigger_ms = esp_timer_get_time() / 1000ULL;
    ESP_LOGW(TAG, "紧急呼叫已由远端通信触发");
}

esp_err_t event_fusion_decide(fusion_result_t *result)
{
    if (!result) return ESP_ERR_INVALID_ARG;

    memset(result, 0, sizeof(fusion_result_t));

    update_button_channel();
    update_pir_channel();

    bool audio = is_audio_active();
    bool pir = s_state.pir_active;
    bool button = s_state.button_active;
    bool emergency = is_emergency_active();

    result->button_active = button;
    result->pir_active = pir;
    result->audio_active = audio;
    result->emergency_active = emergency;
    result->timestamp_ms = esp_timer_get_time() / 1000ULL;

    // 紧急呼叫按钮优先级最高，直接触发
    if (emergency) {
        result->event = FUSION_EVENT_EMERGENCY;
        ESP_LOGW(TAG, "紧急呼叫已触发！");
        return ESP_OK;
    }

    // 计算在时间窗口内的有效通道数
    int active_count = (button ? 1 : 0) + (pir ? 1 : 0) + (audio ? 1 : 0);

    if (active_count >= 2) {
        // 冷却期去重：VISITOR 触发后进入冷却期，期内两通道仍 active 也不重复报警。
        // 解决"一次按键在3s保持窗口内被200ms循环反复报14次有访客"的问题。
        uint32_t now = esp_timer_get_time() / 1000ULL;
        if (now - s_last_visitor_ms >= VISITOR_COOLDOWN_MS) {
            result->event = FUSION_EVENT_VISITOR;
            s_last_visitor_ms = now;
            ESP_LOGI(TAG, "融合判定-有访客: 按键=%d, PIR=%d, 音频=%d", button, pir, audio);
        } else {
            // 冷却期内：不重复触发，返回 NONE（已报过，上层无需再处理）
            result->event = FUSION_EVENT_NONE;
        }
    } else if (active_count == 1) {
        result->event = FUSION_EVENT_SINGLE_CHANNEL;
        uint32_t now = esp_timer_get_time() / 1000ULL;
        if (now - s_last_log_ms > 500) {
            ESP_LOGI(TAG, "单通道触发(仅记录): 按键=%d, PIR=%d, 音频=%d", button, pir, audio);
            s_last_log_ms = now;
        }
    } else {
        result->event = FUSION_EVENT_NONE;
    }

    return ESP_OK;
}

bool event_fusion_get_emergency_state(void)
{
    return is_emergency_active();
}

bool event_fusion_get_button_state(void)
{
    return s_state.button_active;
}

bool event_fusion_get_pir_state(void)
{
    return s_state.pir_active;
}
