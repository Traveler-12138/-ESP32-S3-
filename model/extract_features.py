"""
特征提取脚本 — 从 WAV 文件提取 40 维 log-Mel 特征。

⚠️ 本脚本的特征提取逻辑必须与 main/audio_ai.cpp 中的 C 实现完全一致，
   否则训练的模型在 ESP32 上运行时特征不匹配，识别率会很差。

C 代码关键参数（audio_ai.cpp）:
    SAMPLE_RATE       = 16000
    AUDIO_BUFFER_SAMPLES = 16000  (1秒)
    FFT_SIZE          = 512
    FFT_STEP          = 256       (50%重叠)
    NUM_FRAMES        = (16000-512)/256 + 1 = 61
    NUM_MEL_BINS      = 40
    窗函数             = Hamming: 0.54 - 0.46*cos(2*pi*i/511)
    FFT归一化          = 除以 FFT_SIZE
    功率谱             = real^2 + imag^2
    Mel滤波器          = 三角形, hz_to_mel = 2595*log10(1+f/700)
    最终特征           = log(sum_power / NUM_FRAMES + 1e-6)

用法:
    python model/extract_features.py
    # 自动扫描 data/audio/{noise,knock,doorbell}/ 下的 WAV 文件，
    # 输出 data/features.csv 和 data/labels.csv
"""
import os
import sys
import glob
import wave

import numpy as np

# ===================== 参数（与 audio_ai.cpp 完全一致） =====================
SAMPLE_RATE = 16000
AUDIO_BUFFER_SAMPLES = 16000     # 1秒
FFT_SIZE = 512
FFT_STEP = 256
NUM_FRAMES = (AUDIO_BUFFER_SAMPLES - FFT_SIZE) // FFT_STEP + 1  # 61
NUM_MEL_BINS = 40

CLASS_NAMES = ["noise", "knock", "doorbell"]
CLASS_LABELS = {"noise": 0, "knock": 1, "doorbell": 2}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")


def hz_to_mel(f):
    """与 C 代码一致: 2595 * log10(1 + f/700)"""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    """与 C 代码一致: 700 * (10^(m/2595) - 1)"""
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def init_mel_filterbank():
    """构建 Mel 三角滤波器组，与 audio_ai.cpp 的 init_mel_filterbank() 完全一致。"""
    mel_low = hz_to_mel(0.0)
    mel_high = hz_to_mel(float(SAMPLE_RATE) / 2.0)

    # 42 个 Mel 等分点
    mel_points = np.array([mel_low + i * (mel_high - mel_low) / (NUM_MEL_BINS + 1)
                           for i in range(NUM_MEL_BINS + 2)])
    freq_points = np.array([mel_to_hz(m) for m in mel_points])

    num_freq_bins = FFT_SIZE // 2 + 1  # 257
    bin_freq_step = float(SAMPLE_RATE) / float(FFT_SIZE)  # 31.25

    filterbank = np.zeros((NUM_MEL_BINS, num_freq_bins))

    for m in range(1, NUM_MEL_BINS + 1):
        for k in range(num_freq_bins):
            freq = k * bin_freq_step
            weight = 0.0
            if freq_points[m - 1] <= freq <= freq_points[m]:
                # 上升沿
                denom = freq_points[m] - freq_points[m - 1]
                if denom > 0:
                    weight = (freq - freq_points[m - 1]) / denom
            elif freq_points[m] <= freq <= freq_points[m + 1]:
                # 下降沿
                denom = freq_points[m + 1] - freq_points[m]
                if denom > 0:
                    weight = (freq_points[m + 1] - freq) / denom
            filterbank[m - 1][k] = weight

    return filterbank


def extract_features_from_audio(audio_int16, filterbank):
    """
    从 int16 音频数组提取 40 维 log-Mel 特征。
    与 audio_ai.cpp 的 extract_features() 完全一致。

    参数:
        audio_int16: numpy int16 数组，长度 >= 16000
        filterbank: Mel 滤波器组 (40, 257)
    返回:
        features: numpy float32 数组，长度 40
    """
    # 1. 截取或补零到 16000 样本（与 C 的环形缓冲区一致）
    if len(audio_int16) >= AUDIO_BUFFER_SAMPLES:
        audio = audio_int16[:AUDIO_BUFFER_SAMPLES].astype(np.float64)
    else:
        audio = np.zeros(AUDIO_BUFFER_SAMPLES, dtype=np.float64)
        audio[:len(audio_int16)] = audio_int16.astype(np.float64)

    num_freq_bins = FFT_SIZE // 2 + 1  # 257
    features = np.zeros(NUM_MEL_BINS, dtype=np.float64)

    # 2. Hamming 窗（与 C 一致: 0.54 - 0.46*cos(2*pi*i/(N-1))）
    hamming_window = np.array([0.54 - 0.46 * np.cos(2.0 * np.pi * i / (FFT_SIZE - 1))
                               for i in range(FFT_SIZE)])

    # 3. 逐帧处理
    for frame in range(NUM_FRAMES):
        offset = frame * FFT_STEP

        # 取 512 样本，归一化到 [-1, 1]，加窗
        segment = audio[offset:offset + FFT_SIZE] / 32768.0
        windowed = segment * hamming_window

        # 复数 FFT（虚部为0），与 C 的 dsps_fft2r_fc32 一致
        # np.fft.fft 等价于 C 的未归一化 FFT
        fft_result = np.fft.fft(windowed, FFT_SIZE)

        # 归一化：除以 FFT_SIZE（与 C 一致）
        fft_result = fft_result / float(FFT_SIZE)

        # 功率谱: real^2 + imag^2
        power = np.abs(fft_result[:num_freq_bins]) ** 2

        # 应用 Mel 滤波器组并累加
        features += power @ filterbank.T  # (257,) @ (257, 40) -> (40,)

    # 4. 平均 + 对数压缩
    features = features / float(NUM_FRAMES)
    features = np.log(features + 1e-6)

    return features.astype(np.float32)


def load_wav(filepath):
    """加载 16kHz 16-bit 单声道 WAV 文件。"""
    with wave.open(filepath, "r") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if framerate != SAMPLE_RATE:
        print(f"  ⚠️ 警告: {filepath} 采样率={framerate}，期望{SAMPLE_RATE}，跳过")
        return None

    if sampwidth != 2:
        print(f"  ⚠️ 警告: {filepath} 位深={sampwidth}字节，期望2字节(16-bit)，跳过")
        return None

    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels == 2:
        audio = audio[::2]  # 取左声道
    return audio


def main():
    print("=== 特征提取 ===")
    print(f"参数: 采样率={SAMPLE_RATE}, FFT={FFT_SIZE}, 步长={FFT_STEP}, "
          f"帧数={NUM_FRAMES}, Mel_bins={NUM_MEL_BINS}")

    filterbank = init_mel_filterbank()
    print(f"Mel 滤波器组: {filterbank.shape}")

    all_features = []
    all_labels = []

    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(AUDIO_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"⚠️ 目录不存在，跳过: {cls_dir}")
            continue

        wav_files = sorted(glob.glob(os.path.join(cls_dir, "*.wav")))
        if not wav_files:
            print(f"⚠️ 无 WAV 文件: {cls_dir}")
            continue

        label = CLASS_LABELS[cls_name]
        print(f"\n处理 [{cls_name}] (标签={label}): {len(wav_files)} 个文件")

        for i, wav_path in enumerate(wav_files):
            audio = load_wav(wav_path)
            if audio is None:
                continue

            features = extract_features_from_audio(audio, filterbank)
            all_features.append(features)
            all_labels.append(label)

            if (i + 1) % 20 == 0 or i == len(wav_files) - 1:
                print(f"  已处理 {i+1}/{len(wav_files)}")

    if not all_features:
        print("\n❌ 没有找到任何音频文件！")
        print(f"请先用 record_samples.py 录制样本: python model/record_samples.py --class noise --count 50")
        sys.exit(1)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\n=== 提取完成 ===")
    print(f"特征矩阵: {X.shape}  (样本数, 特征数)")
    print(f"标签分布: noise={np.sum(y==0)}, knock={np.sum(y==1)}, doorbell={np.sum(y==2)}")
    print(f"特征值范围: [{X.min():.3f}, {X.max():.3f}], 均值={X.mean():.3f}")

    # 保存
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savetxt(os.path.join(DATA_DIR, "features.csv"), X, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(DATA_DIR, "labels.csv"), y, delimiter=",", fmt="%d")
    print(f"\n已保存: {DATA_DIR}/features.csv")
    print(f"已保存: {DATA_DIR}/labels.csv")


if __name__ == "__main__":
    main()
