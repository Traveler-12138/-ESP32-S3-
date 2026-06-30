#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "audio_ai.h"
#include "event_fusion.h"

static const char *TAG = "MAIN";

/**
 * @brief 来访提醒回调（预留：后续接入RGB灯带、OLED、振动器、MQTT等）
 */
static void on_visitor_alert(const fusion_result_t *fusion, const audio_result_t *audio)
{
    ESP_LOGI(TAG, "===== 正式来访提醒 =====");
    ESP_LOGI(TAG, "触发通道: button=%d, pir=%d, audio=%d",
             fusion->button_active, fusion->pir_active, fusion->audio_active);
    if (audio->valid) {
        const char *audio_name =
            audio->event == AUDIO_EVENT_KNOCK ? "敲门声" :
            audio->event == AUDIO_EVENT_DOORBELL ? "门铃声" : "无/未知";
        ESP_LOGI(TAG, "音频识别结果: %s, 置信度=%.2f", audio_name, audio->confidence);
    }
    // TODO: 驱动RGB灯带频闪、OLED显示、振动器、MQTT上报等
}

static void on_emergency_alert(void)
{
    ESP_LOGI(TAG, "===== 紧急呼叫告警 =====");
    // TODO: 触发强提醒并通知监护人
}

void app_main(void)
{
    ESP_LOGI(TAG, "智能门铃边缘AI启动中...");

    // 初始化NVS（部分组件如WiFi需要，虽然本模块离线运行但保留）
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 初始化边缘AI音频识别
    ESP_ERROR_CHECK(audio_ai_init());
    ESP_ERROR_CHECK(audio_ai_start());

    // 初始化三通道感知融合
    ESP_ERROR_CHECK(event_fusion_init());

    ESP_LOGI(TAG, "系统初始化完成，开始运行融合判决循环...");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(200));

        // 1. 获取最新音频识别结果
        audio_result_t audio = {0};
        audio_ai_get_latest_result(&audio);

        // 2. 更新音频通道状态
        // audio_ai.cpp 已经做了完整的三态判断(噪声/不确定/确信)和连续确认,
        // 这里只需要检查 event 是否为 KNOCK/DOORBELL 即可, 不再加二次阈值。
        // 原代码的 >0.55 与 audio_ai.cpp 的 0.60 不一致, 会导致 audio_ai 判定
        // KNOCK 但 main.c 不认的矛盾。现在 audio_ai.cpp 阈值降到 0.50, 这里也去掉。
        bool audio_triggered = audio.valid &&
                               (audio.event == AUDIO_EVENT_KNOCK || audio.event == AUDIO_EVENT_DOORBELL);
        event_fusion_update_audio(audio_triggered);

        // 3. 执行融合判决
        fusion_result_t fusion = {0};
        event_fusion_decide(&fusion);

        // 4. 根据判决结果触发不同动作
        switch (fusion.event) {
            case FUSION_EVENT_VISITOR:
                on_visitor_alert(&fusion, &audio);
                break;
            case FUSION_EVENT_EMERGENCY:
                on_emergency_alert();
                break;
            case FUSION_EVENT_SINGLE_CHANNEL:
                // 仅日志记录，不触发提醒
                break;
            case FUSION_EVENT_NONE:
            default:
                break;
        }
    }
}
