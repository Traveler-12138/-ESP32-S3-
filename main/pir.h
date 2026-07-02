#ifndef PIR_H
#define PIR_H

#include "esp_err.h"
#include <stdbool.h>
#include "driver/gpio.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * PIR 传感器 GPIO 引脚。
 * 默认 GPIO7（与组员 hjc 接线一致）。
 * 若你的板子 PIR 接在其它引脚，修改此处或在 sdkconfig.defaults 覆盖即可。
 * 注意：GPIO16/17/18 已被 I2S 麦克风占用，不可复用。
 */
#ifndef PIR_GPIO_PIN
#define PIR_GPIO_PIN  GPIO_NUM_7
#endif

// 初始化PIR传感器
esp_err_t pir_init(void);

// 获取PIR传感器状态
// 返回: true 表示检测到人, false 表示无人
bool pir_is_detected(void);

#ifdef __cplusplus
}
#endif

#endif // PIR_H
