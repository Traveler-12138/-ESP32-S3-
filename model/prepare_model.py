"""
为ESP32-S3生成轻量级音频分类TFLite模型，并转换为C数组。

用法（无真实数据时生成随机权重占位模型）:
    python model/prepare_model.py --dummy
用法（用CSV格式特征数据训练）:
    python model/prepare_model.py --data data/features.csv --labels data/labels.csv

输出:
    model/doorbell_knock_classifier.tflite
    main/model_data.c
    main/model_data.h
"""
import argparse
import os
import struct
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 与C代码中 AUDIO_FEATURE_SIZE 保持一致
AUDIO_FEATURE_SIZE = 40
NUM_CLASSES = 3
CLASS_NAMES = ["noise", "knock", "doorbell"]

def build_model(input_size=AUDIO_FEATURE_SIZE, num_classes=NUM_CLASSES):
    """构建一个极小的全连接分类模型。"""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_size,), name="input"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(32, activation="relu", name="dense_2"),
        keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def representative_dataset_gen():
    """用于INT8量化的代表性数据集生成器。"""
    for _ in range(100):
        data = np.random.randn(1, AUDIO_FEATURE_SIZE).astype(np.float32) * 0.5
        yield [data]


def convert_to_tflite(keras_model, out_path, quantize=False):
    """将Keras模型转换为TFLite。"""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    # 不量化时保持纯float32，避免混合量化模型在TFLite Micro上不兼容
    return converter.convert()


def model_to_c_array(tflite_model, out_c, out_h, array_name="g_model_data"):
    """将TFLite字节流转换为C数组源文件。"""
    n = len(tflite_model)
    hex_lines = []
    for i in range(0, n, 12):
        line = ", ".join("0x{:02x}".format(b) for b in tflite_model[i:i+12])
        hex_lines.append("    " + line + ",")
    body = "\n".join(hex_lines)

    with open(out_c, "w", encoding="utf-8") as f:
        f.write("// Auto-generated model data. DO NOT EDIT.\n")
        f.write('#include "model_data.h"\n\n')
        f.write(f"const unsigned char {array_name}[] = {{\n")
        f.write(body + "\n")
        f.write("};\n\n")
        f.write(f"const unsigned int {array_name}_len = {n};\n")

    with open(out_h, "w", encoding="utf-8") as f:
        f.write("// Auto-generated model data. DO NOT EDIT.\n")
        f.write("#pragma once\n\n")
        f.write(f"#define MODEL_DATA_LEN {n}\n")
        f.write(f"extern const unsigned char {array_name}[];\n")
        f.write(f"extern const unsigned int {array_name}_len;\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare TFLite model for ESP32-S3 edge AI.")
    parser.add_argument("--dummy", action="store_true", help="生成随机权重占位模型，仅用于编译验证")
    parser.add_argument("--data", type=str, help="训练特征CSV文件路径 (每行一个样本，{AUDIO_FEATURE_SIZE}列)")
    parser.add_argument("--labels", type=str, help="训练标签CSV文件路径 (0=noise, 1=knock, 2=doorbell)")
    parser.add_argument("--quantize", action="store_true", help="启用INT8量化（推荐部署时使用）")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__), help="TFLite模型输出目录")
    parser.add_argument("--main-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "main"), help="C数组输出目录")
    args = parser.parse_args()

    model = build_model()

    if args.dummy or (args.data is None and args.labels is None):
        print("未提供训练数据，生成随机权重占位模型。")
        # 用随机数据做少量训练，使权重不是全零
        x_dummy = np.random.randn(200, AUDIO_FEATURE_SIZE).astype(np.float32) * 0.5
        y_dummy = np.random.randint(0, NUM_CLASSES, size=(200,))
        model.fit(x_dummy, y_dummy, epochs=5, verbose=0, batch_size=32)
    else:
        x_train = np.loadtxt(args.data, delimiter=",", dtype=np.float32)
        y_train = np.loadtxt(args.labels, delimiter=",", dtype=np.int32)
        if x_train.ndim == 1:
            x_train = x_train.reshape(1, -1)
        if y_train.ndim == 0:
            y_train = y_train.reshape(1)
        print(f"训练数据: {x_train.shape}, 标签: {y_train.shape}")
        model.fit(x_train, y_train, epochs=50, verbose=1, validation_split=0.2, batch_size=32)

    tflite_model = convert_to_tflite(model, os.path.join(args.output_dir, "model.tflite"), quantize=args.quantize)

    tflite_path = os.path.join(args.output_dir, "doorbell_knock_classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite模型已保存: {tflite_path} ({len(tflite_model)} bytes)")

    main_dir = os.path.abspath(args.main_dir)
    os.makedirs(main_dir, exist_ok=True)
    model_to_c_array(tflite_model, os.path.join(main_dir, "model_data.c"), os.path.join(main_dir, "model_data.h"))
    print(f"C数组已生成: {main_dir}/model_data.c, {main_dir}/model_data.h")


if __name__ == "__main__":
    main()
