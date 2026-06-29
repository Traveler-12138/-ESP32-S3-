# 音频三分类模型训练流程

> 目标：训练「敲门声 / 门铃声 / 环境噪声」三分类模型，替换占位模型，部署到 ESP32-S3。

## 整体流程

```
录音采集 → 特征提取 → 模型训练 → INT8量化 → 生成C数组 → 编译烧录
  (1)        (2)        (3)        (4)         (5)         (6)
```

---

## 第一步：安装 Python 依赖

在电脑上（不是 ESP32 上）执行：

```bash
# 推荐用 Python 3.10-3.12
pip install sounddevice numpy tensorflow scipy
```

> 如果你没有 NVIDIA 显卡，CPU 版 TensorFlow 也能跑，只是训练慢一点（几分钟）。

---

## 第二步：录制音频样本

用电脑麦克风录制三类声音。**每类至少 50 段**，越多越好。

```bash
# 录制噪声样本（50段，每段1.5秒）
python model/record_samples.py --class noise --count 50

# 录制敲门样本（100段，可以敲不同的门、不同力度）
python model/record_samples.py --class knock --count 100

# 录制门铃样本（100段，可以录不同门铃、手机播放门铃音频）
python model/record_samples.py --class doorbell --count 100
```

录音时注意：
- **噪声**：录真实环境噪声（风扇声、谈话声、电视声），不要对着麦克风吹气
- **敲门**：真的去敲木板/门，距离麦克风 30cm-1m
- **门铃**：可以真按门铃，或用手机播放门铃音频对着电脑麦克风

文件会保存在 `model/data/audio/{noise,knock,doorbell}/` 目录下。

---

## 第三步：验证特征提取（可选，推荐做）

确认特征提取逻辑正确：

```bash
python model/extract_features.py
```

输出示例：
```
特征矩阵: (250, 40)  (样本数, 特征数)
标签分布: noise=50, knock=100, doorbell=100
特征值范围: [-15.234, 3.456], 均值=-8.123
```

如果特征值范围异常（全是 0 或全是 nan），检查 WAV 文件是否正确。

---

## 第四步：训练模型

### 4.1 快速验证（float32，不量化）

```bash
python model/train_model.py --epochs 50
```

### 4.2 正式训练（INT8 量化，推荐部署用）

```bash
python model/train_model.py --quantize --epochs 80
```

训练过程会输出：
```
[1/5] 构建 Mel 滤波器组...
[2/5] 加载音频数据 + 增强(x3)...
      总样本数: 1000 (原始250 + 增强750)
[3/5] 训练模型 (epochs=80)...
      Epoch 1/80 - loss: 1.23 - accuracy: 0.45 - val_accuracy: 0.52
      ...
      最佳验证准确率: 0.9234
[4/5] 转换为 TFLite (INT8量化)...
      模型大小: 5432 字节 (5.3 KB)
[5/5] 生成 C 数组...
```

**验证准确率目标**：≥ 0.85（85%）。如果低于 0.7，说明数据不够或质量差，需要多录一些。

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数，数据少时用 80-100 |
| `--augment` | 3 | 每个样本增强次数（加噪/偏移/音量） |
| `--quantize` | 关 | 加上则启用 INT8 全量化 |
| `--audio-dir` | model/data/audio | 音频数据目录 |
| `--main-dir` | ../main | C 数组输出目录 |

---

## 第五步：重新编译烧录

训练完成后，`main/model_data.c` 和 `main/model_data.h` 会自动更新。

```bash
# 在 VS Code 终端执行
idf.py build flash monitor
```

### 期望效果

| 场景 | 推理结果 | 置信度 |
|------|---------|--------|
| 安静环境 | 无 (noise) | > 0.85 |
| 敲门 | 敲门 (knock) | > 0.80 |
| 按门铃 | 门铃 (doorbell) | > 0.80 |
| 说话/其他噪声 | 无 (noise) | > 0.70 |

### float32 vs INT8 对比

| 指标 | float32 | INT8 量化 |
|------|---------|-----------|
| 模型体积 | ~21 KB | ~5 KB |
| 推理速度 | 较快 | 更快 |
| 准确率 | 基准 | 略降 1-3% |
| 内存占用 | 较大 | 较小 |

**推荐用 INT8**，体积缩小 4 倍，准确率几乎不变。

---

## 数据增强说明

`train_model.py` 内置了 3 种数据增强（每个原始样本生成 3 个增强样本）：

1. **随机音量** (0.5x ~ 1.5x)：模拟敲门力度不同
2. **高斯噪声**：模拟不同环境噪声
3. **时间偏移** (±0.1秒)：模拟声音到达麦克风的时延

这样 250 个原始样本会扩充到 1000 个训练样本，有效防止过拟合。

---

## 常见问题

### Q: 验证准确率很低 (< 0.6) 怎么办？
- 数据太少：每类至少录 50 段
- 数据不均衡：确保三类数量接近
- 录音质量差：检查是否有杂音、音量过小
- 增加 `--augment 5` 和 `--epochs 100`

### Q: 录音时提示找不到麦克风？
- 检查电脑麦克风权限
- Windows: 设置 → 隐私 → 麦克风 → 允许应用访问

### Q: TensorFlow 安装失败？
- 用 Python 3.10-3.12
- Windows: `pip install tensorflow-cpu`（不需要显卡版）
- 或用 conda: `conda install tensorflow`

### Q: INT8 量化后推理报错？
- 确认 `audio_ai.cpp` 的 OpResolver 包含 `AddQuantize()` 和 `AddDequantize()`
- 当前代码已包含，无需修改

### Q: 可以用手机录音代替吗？
- 可以，但需要转为 16kHz/16-bit/单声道 WAV
- 用 Audacity 或 ffmpeg 转换:
  ```bash
  ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 output.wav
  ```
- 放到对应的 `model/data/audio/{类别}/` 目录

---

## 文件结构

```
model/
├── record_samples.py      # 录音脚本
├── extract_features.py    # 特征提取验证脚本
├── train_model.py         # 训练+量化+生成C数组（主脚本）
├── prepare_model.py       # 旧脚本（保留兼容，不再推荐）
├── README_TRAINING.md     # 本文件
└── data/
    └── audio/
        ├── noise/         # 噪声 WAV 文件
        ├── knock/         # 敲门 WAV 文件
        └── doorbell/      # 门铃 WAV 文件
```
