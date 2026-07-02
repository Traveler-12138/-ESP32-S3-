#ifndef DOOR_BUTTON_H
#define DOOR_BUTTON_H

#include "esp_err.h"
#include "driver/gpio.h"
#include "stdbool.h"

#ifdef __cplusplus
extern "C" {
#endif

// 门外按键 GPIO 引脚配置（按键 GPIO4，低电平有效）
#define DOOR_BUTTON_GPIO_PIN     GPIO_NUM_4

// 按键消抖时间（毫秒）
#define DOOR_BUTTON_DEBOUNCE_MS  50

// 初始化门外按键
esp_err_t door_button_init(void);

// 获取按键当前状态（已消抖）
bool door_button_is_pressed(void);

// 获取按键原始电平状态（未消抖）
bool door_button_get_raw_level(void);

// 检测按键是否刚从释放变为按下（下降沿）
bool door_button_is_falling_edge(void);

// 检测按键是否刚从按下变为释放（上升沿）
bool door_button_is_rising_edge(void);

// 清除边沿状态
void door_button_clear_edge(void);

#ifdef __cplusplus
}
#endif

#endif // DOOR_BUTTON_H
