# 面向听障人群的无障碍智能门铃系统

全国大学生物联网设计大赛 · 乐鑫赛道作品。

基于 ESP32-S3 + ESP-IDF 的边缘 AI 智能门铃，通过敲门声 / 门铃声 / 环境噪声
三分类模型本地识别来访信号，融合按键、PIR、音频三通道感知，为听障人群
提供视觉/触觉来访提醒。

## 项目结构

```
helloworld/
├── main/                  # 固件源码
│   ├── main.c             # 主程序 + 融合判决循环
│   ├── audio_ai.cpp/.h    # TFLite Micro 音频识别 (特征提取+推理+三态判决)
│   ├── mic.c/.h           # INMP441 I2S 麦克风驱动
│   ├── event_fusion.c/.h  # 三通道感知融合 (按键+PIR+音频)
│   ├── model_data.c/.h    # 训练好的模型数据 (由训练脚本生成)
│   └── CMakeLists.txt
├── sdkconfig.defaults     # ESP-IDF 默认配置
├── CMakeLists.txt
└── README.md
```

> **模型训练文件已迁出**到 `E:\ESP32\Training_file\`（含数据集、训练脚本、
> Python 环境），本仓库只保留固件推理用的 `main/model_data.*`。
> 重新训练方法见 `E:\ESP32\Training_file\README.md`。

## 硬件

- 主控：ESP32-S3（WiFi + BLE + NPU）
- 麦克风：INMP441 I2S MEMS（BCLK=GPIO17, WS=GPIO18, DIN=GPIO16）
- 按键：GPIO4（高电平触发，下拉）
- PIR：GPIO19
- 紧急按钮：GPIO21

## 编译烧录

需要 ESP-IDF v5.x 环境：

```bash
idf.py build flash monitor
```

## 核心设计

- **边缘 AI**：TFLite Micro + ESP-DSP，40 维 log-Mel 特征，1 秒推理窗口
- **三通道融合**：按键 + PIR + 音频，至少两通道有效才触发正式提醒
- **离线优先**：核心识别本地完成，MQTT 仅用于来访记录上报
