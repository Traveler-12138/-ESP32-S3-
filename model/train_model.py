"""
训练脚本 — 训练敲门/门铃/噪声三分类模型，支持数据增强和INT8量化。

完整流程:
    1. 加载 WAV 音频文件
    2. 数据增强（加噪/时间偏移/音量变化）扩充样本
    3. 提取 40 维 log-Mel 特征（与 extract_features.py 一致）
    4. 训练 Keras 模型
    5. INT8 量化（可选，推荐）
    6. 生成 main/model_data.c 和 main/model_data.h

用法:
    # 训练 float32 模型（快速验证）
    python model/train_model.py

    # 训练 INT8 量化模型（部署用，体积更小、推理更快）
    python model/train_model.py --quantize

    # 指定数据目录和输出
    python model/train_model.py --data-dir model/data --quantize --epochs 80
"""
import argparse
import os
import sys
import glob
import wave

import numpy as np

# ===================== 参数（与 audio_ai.cpp 一致） =====================
SAMPLE_RATE = 16000
AUDIO_BUFFER_SAMPLES = 16000
FFT_SIZE = 512
FFT_STEP = 256
NUM_FRAMES = (AUDIO_BUFFER_SAMPLES - FFT_SIZE) // FFT_STEP + 1
NUM_MEL_BINS = 40

CLASS_NAMES = ["noise", "knock", "doorbell"]
CLASS_LABELS = {"noise": 0, "knock": 1, "doorbell": 2}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DEFAULT_AUDIO_DIR = os.path.join(DEFAULT_DATA_DIR, "audio")
DEFAULT_MAIN_DIR = os.path.join(SCRIPT_DIR, "..", "main")


# ===================== 特征提取（与 C 代码完全一致） =====================
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def init_mel_filterbank():
    mel_low = hz_to_mel(0.0)
    mel_high = hz_to_mel(float(SAMPLE_RATE) / 2.0)
    mel_points = np.array([mel_low + i * (mel_high - mel_low) / (NUM_MEL_BINS + 1)
                           for i in range(NUM_MEL_BINS + 2)])
    freq_points = np.array([mel_to_hz(m) for m in mel_points])

    num_freq_bins = FFT_SIZE // 2 + 1
    bin_freq_step = float(SAMPLE_RATE) / float(FFT_SIZE)
    filterbank = np.zeros((NUM_MEL_BINS, num_freq_bins))

    for m in range(1, NUM_MEL_BINS + 1):
        for k in range(num_freq_bins):
            freq = k * bin_freq_step
            weight = 0.0
            if freq_points[m - 1] <= freq <= freq_points[m]:
                denom = freq_points[m] - freq_points[m - 1]
                if denom > 0:
                    weight = (freq - freq_points[m - 1]) / denom
            elif freq_points[m] <= freq <= freq_points[m + 1]:
                denom = freq_points[m + 1] - freq_points[m]
                if denom > 0:
                    weight = (freq_points[m + 1] - freq) / denom
            filterbank[m - 1][k] = weight
    return filterbank


def extract_features(audio_int16, filterbank):
    """与 audio_ai.cpp 的 extract_features() 完全一致。"""
    if len(audio_int16) >= AUDIO_BUFFER_SAMPLES:
        audio = audio_int16[:AUDIO_BUFFER_SAMPLES].astype(np.float64)
    else:
        audio = np.zeros(AUDIO_BUFFER_SAMPLES, dtype=np.float64)
        audio[:len(audio_int16)] = audio_int16.astype(np.float64)

    num_freq_bins = FFT_SIZE // 2 + 1
    features = np.zeros(NUM_MEL_BINS, dtype=np.float64)

    hamming_window = np.array([0.54 - 0.46 * np.cos(2.0 * np.pi * i / (FFT_SIZE - 1))
                               for i in range(FFT_SIZE)])

    for frame in range(NUM_FRAMES):
        offset = frame * FFT_STEP
        segment = audio[offset:offset + FFT_SIZE] / 32768.0
        windowed = segment * hamming_window
        fft_result = np.fft.fft(windowed, FFT_SIZE) / float(FFT_SIZE)
        power = np.abs(fft_result[:num_freq_bins]) ** 2
        features += power @ filterbank.T

    features = features / float(NUM_FRAMES)
    features = np.log(features + 1e-6)
    return features.astype(np.float32)


# ===================== 数据增强 =====================
def augment_audio(audio_int16, rng):
    """
    对原始音频做数据增强，返回增强后的 int16 数组。
    增强方式: 加高斯噪声 + 随机音量 + 时间偏移
    """
    audio = audio_int16.astype(np.float64)

    # 1. 随机音量缩放 (0.5 ~ 1.5)
    volume_factor = rng.uniform(0.5, 1.5)
    audio = audio * volume_factor

    # 2. 加高斯噪声（模拟环境噪声）
    noise_level = rng.uniform(0.001, 0.02)  # 噪声强度
    noise = rng.normal(0, noise_level * 32768, len(audio))
    audio = audio + noise

    # 3. 时间偏移（最多偏移 0.1 秒 = 1600 样本）
    shift = rng.randint(-1600, 1600)
    if shift > 0:
        audio = np.concatenate([np.zeros(shift), audio[:-shift]])
    elif shift < 0:
        audio = np.concatenate([audio[-shift:], np.zeros(-shift)])

    # 4. 裁剪到 int16 范围
    audio = np.clip(audio, -32768, 32767)
    return audio.astype(np.int16)


# ===================== 数据加载 =====================
def load_wav(filepath):
    with wave.open(filepath, "r") as wf:
        framerate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if framerate != SAMPLE_RATE:
        return None
    if sampwidth != 2:
        return None

    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels == 2:
        audio = audio[::2]
    return audio


def load_dataset(audio_dir, filterbank, augment_count=3):
    """
    加载所有 WAV 文件，提取特征并做数据增强。
    每个原始样本生成 augment_count 个增强样本 + 1 个原始样本。

    返回: (X, y) 特征矩阵和标签
    """
    all_features = []
    all_labels = []
    rng = np.random.RandomState(42)

    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(audio_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠️ 目录不存在: {cls_dir}")
            continue

        wav_files = sorted(glob.glob(os.path.join(cls_dir, "*.wav")))
        label = CLASS_LABELS[cls_name]
        print(f"  [{cls_name}] (标签={label}): {len(wav_files)} 个原始文件")

        for wav_path in wav_files:
            audio = load_wav(wav_path)
            if audio is None or len(audio) < FFT_SIZE:
                continue

            # 原始样本
            feat = extract_features(audio, filterbank)
            all_features.append(feat)
            all_labels.append(label)

            # 增强样本
            for _ in range(augment_count):
                aug_audio = augment_audio(audio, rng)
                aug_feat = extract_features(aug_audio, filterbank)
                all_features.append(aug_feat)
                all_labels.append(label)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    return X, y


# ===================== 模型构建与训练 =====================
def build_model(input_size=NUM_MEL_BINS, num_classes=3):
    """构建全连接分类模型。"""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(input_size,), name="input"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", name="dense_2"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(X, y, epochs=50):
    """训练模型，返回训练好的模型和验证准确率。"""
    import tensorflow as tf
    from tensorflow import keras

    # 打乱数据
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # 划分训练集/验证集 (80/20)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    model = build_model()

    # 早停：验证损失不再下降时停止
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
    )

    val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"  最佳验证准确率: {val_acc:.4f}")

    return model, X_train  # 返回 X_train 用于量化参考


def convert_to_tflite(model, representative_data, quantize=False):
    """转换为 TFLite，可选 INT8 量化。"""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("  启用 INT8 全量化...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 用真实特征作为量化参考数据
        def representative_dataset():
            for i in range(min(200, len(representative_data))):
                sample = representative_data[i:i+1].astype(np.float32)
                yield [sample]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        # float32 模型，不做任何优化，避免混合量化
        converter.optimizations = []

    tflite_model = converter.convert()
    return tflite_model


def model_to_c_array(tflite_model, out_c, out_h, array_name="g_model_data"):
    """将 TFLite 字节流转换为 C 数组源文件。"""
    n = len(tflite_model)
    hex_lines = []
    for i in range(0, n, 12):
        line = ", ".join("0x{:02x}".format(b) for b in tflite_model[i:i+12])
        hex_lines.append("    " + line + ",")
    body = "\n".join(hex_lines)

    with open(out_c, "w", encoding="utf-8") as f:
        f.write("// Auto-generated model data. DO NOT EDIT.\n")
        f.write("// 由 model/train_model.py 自动生成\n")
        f.write('#include "model_data.h"\n\n')
        f.write(f"const unsigned char {array_name}[] = {{\n")
        f.write(body + "\n")
        f.write("};\n\n")
        f.write(f"const unsigned int {array_name}_len = {n};\n")

    with open(out_h, "w", encoding="utf-8") as f:
        f.write("// Auto-generated model data. DO NOT EDIT.\n")
        f.write("// 由 model/train_model.py 自动生成\n")
        f.write("#pragma once\n\n")
        f.write(f"#define MODEL_DATA_LEN {n}\n")
        f.write(f"extern const unsigned char {array_name}[];\n")
        f.write(f"extern const unsigned int {array_name}_len;\n")


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="训练音频三分类模型并生成C数组")
    parser.add_argument("--audio-dir", type=str, default=DEFAULT_AUDIO_DIR,
                        help=f"音频数据目录 (默认: {DEFAULT_AUDIO_DIR})")
    parser.add_argument("--main-dir", type=str, default=DEFAULT_MAIN_DIR,
                        help=f"C数组输出目录 (默认: {DEFAULT_MAIN_DIR})")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数 (默认: 50)")
    parser.add_argument("--augment", type=int, default=3,
                        help="每个样本增强次数 (默认: 3)")
    parser.add_argument("--quantize", action="store_true",
                        help="启用 INT8 量化（推荐部署时使用）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("  音频三分类模型训练")
    print("=" * 60)
    print(f"  音频目录: {args.audio_dir}")
    print(f"  输出目录: {args.main_dir}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  增强次数: {args.augment} (每个样本)")
    print(f"  INT8量化: {'是' if args.quantize else '否'}")
    print()

    # 检查音频目录
    if not os.path.isdir(args.audio_dir):
        print(f"❌ 音频目录不存在: {args.audio_dir}")
        print(f"   请先运行: python model/record_samples.py --class noise --count 50")
        sys.exit(1)

    # 1. 构建滤波器组
    print("[1/5] 构建 Mel 滤波器组...")
    filterbank = init_mel_filterbank()
    print(f"      滤波器组: {filterbank.shape}")

    # 2. 加载并增强数据
    print(f"\n[2/5] 加载音频数据 + 增强(x{args.augment})...")
    X, y = load_dataset(args.audio_dir, filterbank, augment_count=args.augment)

    if len(X) == 0:
        print("❌ 没有找到任何音频文件！")
        sys.exit(1)

    print(f"      总样本数: {len(X)} (原始+增强)")
    print(f"      特征矩阵: {X.shape}")
    print(f"      标签分布: noise={np.sum(y==0)}, knock={np.sum(y==1)}, doorbell={np.sum(y==2)}")
    print(f"      特征范围: [{X.min():.3f}, {X.max():.3f}], 均值={X.mean():.3f}")

    # 3. 训练模型
    print(f"\n[3/5] 训练模型 (epochs={args.epochs})...")
    model, X_train = train_model(X, y, epochs=args.epochs)

    # 4. 转换为 TFLite
    print(f"\n[4/5] 转换为 TFLite{' (INT8量化)' if args.quantize else ' (float32)'}...")
    tflite_model = convert_to_tflite(model, X_train, quantize=args.quantize)
    print(f"      模型大小: {len(tflite_model)} 字节 ({len(tflite_model)/1024:.1f} KB)")

    # 保存 .tflite 文件
    tflite_path = os.path.join(SCRIPT_DIR, "doorbell_knock_classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"      已保存: {tflite_path}")

    # 5. 生成 C 数组
    print(f"\n[5/5] 生成 C 数组...")
    main_dir = os.path.abspath(args.main_dir)
    os.makedirs(main_dir, exist_ok=True)
    model_to_c_array(
        tflite_model,
        os.path.join(main_dir, "model_data.c"),
        os.path.join(main_dir, "model_data.h")
    )
    print(f"      已保存: {main_dir}/model_data.c")
    print(f"      已保存: {main_dir}/model_data.h")

    # 完成总结
    print("\n" + "=" * 60)
    print("  训练完成！")
    print("=" * 60)
    print(f"  模型类型: {'INT8量化' if args.quantize else 'float32'}")
    print(f"  模型大小: {len(tflite_model)} 字节")
    print(f"  下一步: 重新编译烧录 -> idf.py build flash monitor")
    print("=" * 60)


if __name__ == "__main__":
    main()
