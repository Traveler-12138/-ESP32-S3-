#include "pir.h"
#include "esp_log.h"
#include <stdbool.h>

// （靠红色电阻一方，从左往右）VCC -5V，OUT -GPIO7，GND
static const char *TAG = "PIR";

esp_err_t pir_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << PIR_GPIO_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,   // PIR模块通常有下拉，不需要上拉
        .pull_down_en = GPIO_PULLDOWN_ENABLE, // 确保未触发时电平稳定为低
        .intr_type = GPIO_INTR_DISABLE,       // 轮询方式读取，不使用中断
    };

    esp_err_t ret = gpio_config(&io_conf);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "PIR initialized on GPIO%d", PIR_GPIO_PIN);
    } else {
        ESP_LOGE(TAG, "PIR initialization failed");
    }
    return ret;
}

bool pir_is_detected(void)
{
    // 读取GPIO电平，高电平表示检测到人
    return gpio_get_level(PIR_GPIO_PIN) == 1;
}
