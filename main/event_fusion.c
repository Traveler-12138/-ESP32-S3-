#include "event_fusion.h"

#include <string.h>
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ===================== 硬件连接配置（根据实际接线修改） =====================
// 注意：GPIO16/17/18 已被 I2S 麦克风占用（DIN/BCLK/WS），不能重复使用！
#define BUTTON_GPIO             GPIO_NUM_4    // 按键（原GPIO18与麦克风WS冲突，已改）
#define PIR_GPIO                GPIO_NUM_19
#define EMERGENCY_GPIO          GPIO_NUM_21

// ===================== 软件参数 =====================
#define TAG                     "FUSION"
#define DEBOUNCE_MS             50      // 按键消抖时间
#define PIR_HOLD_MS             2000    // PIR信号保持有效时间
#define AUDIO_HOLD_MS           1500    // 音频识别结果保持有效时间

// ===================== 状态结构 =====================
typedef struct {
    bool button_raw;
    bool pir_raw;
    bool emergency_raw;
    bool audio_active;

    uint32_t button_last_ms;
    uint32_t pir_trigger_ms;
    uint32_t audio_trigger_ms;
    uint32_t emergency_trigger_ms;

    bool button_debounced;
    bool pir_active;
} channel_state_t;

static channel_state_t s_state = {0};
static uint32_t s_last_log_ms = 0;

// ===================== GPIO中断处理 =====================
static void IRAM_ATTR gpio_isr_handler(void *arg)
{
    uint32_t gpio_num = (uint32_t)arg;
    uint32_t now = esp_timer_get_time() / 1000ULL;

    if (gpio_num == BUTTON_GPIO) {
        s_state.button_raw = gpio_get_level(BUTTON_GPIO) == 1;
        s_state.button_last_ms = now;
    } else if (gpio_num == EMERGENCY_GPIO) {
        s_state.emergency_raw = gpio_get_level(EMERGENCY_GPIO) == 1;
        if (s_state.emergency_raw) {
            s_state.emergency_trigger_ms = now;
        }
    }
}

static void update_pir_state(void)
{
    uint32_t now = esp_timer_get_time() / 1000ULL;
    s_state.pir_raw = gpio_get_level(PIR_GPIO) == 1;

    if (s_state.pir_raw) {
        s_state.pir_trigger_ms = now;
        s_state.pir_active = true;
    } else if (now - s_state.pir_trigger_ms > PIR_HOLD_MS) {
        s_state.pir_active = false;
    }
}

static void update_button_state(void)
{
    uint32_t now = esp_timer_get_time() / 1000ULL;
    if (s_state.button_raw && (now - s_state.button_last_ms >= DEBOUNCE_MS)) {
        s_state.button_debounced = true;
    } else if (!s_state.button_raw) {
        s_state.button_debounced = false;
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
    if (s_state.emergency_raw && (now - s_state.emergency_trigger_ms <= 5000)) {
        return true;
    }
    s_state.emergency_raw = false;
    return false;
}

// ===================== 接口实现 =====================
esp_err_t event_fusion_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << BUTTON_GPIO) | (1ULL << PIR_GPIO) | (1ULL << EMERGENCY_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,  // 默认下拉，高电平触发
        .intr_type = GPIO_INTR_ANYEDGE,
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));

    ESP_ERROR_CHECK(gpio_install_isr_service(0));
    ESP_ERROR_CHECK(gpio_isr_handler_add(BUTTON_GPIO, gpio_isr_handler, (void *)BUTTON_GPIO));
    ESP_ERROR_CHECK(gpio_isr_handler_add(EMERGENCY_GPIO, gpio_isr_handler, (void *)EMERGENCY_GPIO));

    memset(&s_state, 0, sizeof(s_state));
    ESP_LOGI(TAG, "融合模块初始化完成。按键=%d, PIR=%d, 紧急按钮=%d", BUTTON_GPIO, PIR_GPIO, EMERGENCY_GPIO);
    return ESP_OK;
}

void event_fusion_update_audio(bool audio_triggered)
{
    // 仅在 audio_active 从 false 跳变为 true 时设置触发时间戳。
    // 原实现: 只要 audio_triggered 为 true 就刷新 audio_trigger_ms,
    // 而 main 循环每 200ms 查询一次、推理约 1.4s 一次, 导致同一个 KNOCK
    // 判决在保持窗口内被反复刷新, 窗口永远不过期, 单通道日志不停打印。
    // 修复后: 保持窗口从"首次触发"起算, 到期自然过期, 下次新触发才重新计时。
    if (audio_triggered && !s_state.audio_active) {
        s_state.audio_active = true;
        s_state.audio_trigger_ms = esp_timer_get_time() / 1000ULL;
    }
    // audio_active 的清除 (过期) 由 is_audio_active() 统一处理, 此处不主动清
}

esp_err_t event_fusion_decide(fusion_result_t *result)
{
    if (!result) return ESP_ERR_INVALID_ARG;

    memset(result, 0, sizeof(fusion_result_t));

    update_pir_state();
    update_button_state();

    bool audio = is_audio_active();
    bool pir = s_state.pir_active;
    bool button = s_state.button_debounced;
    bool emergency = is_emergency_active();

    result->button_active = button;
    result->pir_active = pir;
    result->audio_active = audio;
    result->emergency_active = emergency;
    result->timestamp_ms = esp_timer_get_time() / 1000ULL;

    // 紧急呼叫按钮优先级最高，直接触发
    if (emergency) {
        result->event = FUSION_EVENT_EMERGENCY;
        ESP_LOGI(TAG, "紧急呼叫已触发！");
        return ESP_OK;
    }

    // 计算在时间窗口内的有效通道数
    int active_count = (button ? 1 : 0) + (pir ? 1 : 0) + (audio ? 1 : 0);

    if (active_count >= 2) {
        result->event = FUSION_EVENT_VISITOR;
        ESP_LOGI(TAG, "融合判定-有访客: 按键=%d, PIR=%d, 音频=%d", button, pir, audio);
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
    return s_state.button_debounced;
}

bool event_fusion_get_pir_state(void)
{
    return s_state.pir_active;
}
