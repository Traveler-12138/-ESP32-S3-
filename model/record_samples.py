"""
录音采集脚本 — 用电脑麦克风录制敲门/门铃/噪声样本。

用法:
    python model/record_samples.py --class noise    --count 50
    python model/record_samples.py --class knock    --count 100
    python model/record_samples.py --class doorbell --count 100

每段录制 1.5 秒（16kHz, 16-bit, 单声道），自动保存为 WAV。
"""
import argparse
import os
import sys
import time
import wave

try:
    import sounddevice as sd
except ImportError:
    print("缺少 sounddevice，请先安装: pip install sounddevice numpy")
    sys.exit(1)

import numpy as np

# 与 ESP32 采集参数一致
SAMPLE_RATE = 16000
DURATION_SEC = 1.5          # 每段录音长度（秒），略大于1秒推理窗口
RECORD_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "audio")


def record_one():
    """录制一段音频，返回 int16 numpy 数组。"""
    print(f"  准备录音 {DURATION_SEC} 秒...", end="", flush=True)
    # 倒计时
    for i in range(3, 0, -1):
        print(f"\r  {i} 秒后开始录音...", end="", flush=True)
        time.sleep(1)
    print(f"\r  >>> 录音中（{DURATION_SEC}秒）...", end="", flush=True)

    audio = sd.rec(RECORD_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print(" 完成。")
    return audio.flatten()


def save_wav(audio, filepath):
    """保存为 16kHz 16-bit 单声道 WAV。"""
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def main():
    parser = argparse.ArgumentParser(description="录制音频样本")
    parser.add_argument("--class", dest="cls", required=True,
                        choices=["noise", "knock", "doorbell"],
                        help="要录制的类别")
    parser.add_argument("--count", type=int, default=50,
                        help="录制段数（默认50）")
    parser.add_argument("--start", type=int, default=0,
                        help="起始编号（续录时用）")
    args = parser.parse_args()

    out_dir = os.path.join(DATA_DIR, args.cls)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== 录制 [{args.cls}] 类别，共 {args.count} 段 ===")
    print(f"保存目录: {out_dir}")
    print(f"参数: {SAMPLE_RATE}Hz, 16-bit, 单声道, 每段 {DURATION_SEC} 秒")
    print()

    success = 0
    for i in range(args.start, args.start + args.count):
        print(f"[{i+1}/{args.start + args.count}] 第 {i} 段")
        audio = record_one()

        # 播放回放让用户确认（可选）
        # sd.play(audio, SAMPLE_RATE); sd.wait()

        filepath = os.path.join(out_dir, f"{args.cls}_{i:04d}.wav")
        save_wav(audio, filepath)
        print(f"  已保存: {filepath}")

        # 检测是否静音（最大值过小说明没录到）
        peak = np.max(np.abs(audio))
        if peak < 500:
            print(f"  ⚠️ 警告: 音量过小（峰值={peak}），可能没录到声音，请检查麦克风。")

        success += 1
        time.sleep(0.3)

    print(f"\n录制完成！成功 {success} 段。")
    print(f"文件保存在: {out_dir}")


if __name__ == "__main__":
    main()
