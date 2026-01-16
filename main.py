# -*- coding: utf-8 -*-
import os
import time
import json
from pathlib import Path

from flask import Flask, request, redirect, render_template, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from PIL import Image, ImageOps  # 可視化保存に使用

# ========================
# 設定（DenseNet121 学習コードに合わせる）
# ========================
IMAGE_SIZE = 200
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}  # 安定運用なら gif は避ける

# ★Kerasモデルではなく、TFLiteを読む
TFLITE_MODEL_PATH = "./model_fp16.tflite"            # 生成した tflite を指定
CLASS_INDEX_PATH = "./class_indices.json"            # train_generator.class_indices のJSON（推奨）

# アプリ初期化
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "catdog_demo_secret"

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# ========================
# ユーティリティ
# ========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_label_map():
    """
    class_indices.json があればそれを使って 0/1 の意味を確定させる。
    無ければ {0:'Cat', 1:'Dog'} として扱う（※環境によって逆になるので注意）。
    """
    if os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)  # 例: {"cats":0,"dogs":1}

        idx_to_raw = {int(v): str(k) for k, v in class_indices.items()}

        def pretty(name: str) -> str:
            n = name.lower()
            if "cat" in n:
                return "Cat"
            if "dog" in n:
                return "Dog"
            return name

        idx_to_label = {idx: pretty(lbl) for idx, lbl in idx_to_raw.items()}

        # 0/1 どちらも存在しないケースへの保険
        if 0 not in idx_to_label:
            idx_to_label[0] = "Class0"
        if 1 not in idx_to_label:
            idx_to_label[1] = "Class1"

        return idx_to_label

    # fallback（※本当は class_indices.json を置くのが安全）
    return {0: "Cat", 1: "Dog"}


def load_rgb_200x200(path: str) -> np.ndarray:
    """
    画像を (200,200,3) float32 [0,1] にして返す
    学習側: ImageDataGenerator(rescale=1/255) + target_size=(200,200)
    に合わせて、RGB化→リサイズ→/255 を行う。
    """
    img = kimage.load_img(
        path,
        color_mode="rgb",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        interpolation="nearest",
    )
    x = kimage.img_to_array(img)  # (200,200,3) float32
    x = (x / 255.0).astype("float32")
    return x


def prepare_input_for_tflite(x_hwc: np.ndarray) -> np.ndarray:
    """
    x_hwc: (200,200,3)
    TFLite入力に合わせて (1,200,200,3) にするだけでOK。
    """
    return np.expand_dims(x_hwc, axis=0)


def save_visual_images(x_hwc: np.ndarray, src_filename: str) -> tuple[str, str]:
    """
    2枚保存してテンプレ側の表示に流す（変数名は既存に合わせる）:
      - vis_gray_url  : 実際にモデルへ入れた画像（200x200, stretch）
      - vis_invert_url: 参考用：縦横比を維持して 200x200 にpadした画像（モデルには入れていない）
    """
    x_uint8 = (x_hwc * 255.0).clip(0, 255).astype(np.uint8)
    img_input = Image.fromarray(x_uint8, mode="RGB")

    # 参考：縦横比維持+pad（※モデル入力ではない）
    src_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(src_filename))
    try:
        raw = Image.open(src_path).convert("RGB")
        img_pad = ImageOps.pad(
            raw,
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.NEAREST,
            color=(0, 0, 0),
        )
    except Exception:
        img_pad = img_input.copy()

    base, _ = os.path.splitext(os.path.basename(src_filename))
    ts = int(time.time())
    input_name = f"{base}_modelinput_{ts}.png"
    pad_name = f"{base}_aspectpad_{ts}.png"

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_name)
    pad_path = os.path.join(app.config["UPLOAD_FOLDER"], pad_name)

    img_input.save(input_path)
    img_pad.save(pad_path)

    input_url = url_for("uploaded_file", filename=input_name)
    pad_url = url_for("uploaded_file", filename=pad_name)
    return input_url, pad_url


# ========================
# TFLite 推論器（起動時1回）
# ========================
if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(
        f"TFLite model not found: {TFLITE_MODEL_PATH}\n"
        "先に model_fp16.tflite を生成して配置してください。"
    )

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = load_label_map()

# 目視デバッグ用（サーバ起動ログで確認）
try:
    print(">> Loaded TFLite:", TFLITE_MODEL_PATH)
    print(">> input_details:", input_details)
    print(">> output_details:", output_details)
    print(">> label_map:", label_map)
except Exception:
    pass


def predict_tflite(x_in: np.ndarray) -> float:
    """
    x_in: (1,200,200,3) float32 を想定（/255 済み）
    戻り値: sigmoid出力（0〜1）を float で返す
    """
    # 入力dtypeに合わせる（fp16モデルでも入出力はfloat32のことが多い）
    in_dtype = input_details[0]["dtype"]
    x = x_in.astype(in_dtype, copy=False)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_details[0]["index"])  # 例: (1,1)
    y = np.asarray(y).reshape(-1)
    if y.size < 1:
        raise ValueError("TFLite output is empty.")
    return float(y[0])


# ========================
# ルーティング
# ========================
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # 入力チェック
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("ファイルがありません")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("許可されていない拡張子です（png, jpg, jpeg, webp）")
            return redirect(request.url)

        # 保存
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # 前処理（RGB 200x200 / 0-1）
        try:
            x_hwc = load_rgb_200x200(filepath)      # (200,200,3)
            x_in = prepare_input_for_tflite(x_hwc)  # (1,200,200,3)
        except Exception as e:
            flash(f"前処理でエラー: {e}")
            return redirect(request.url)

        # 予測（sigmoid 1出力）
        try:
            prob_pos = predict_tflite(x_in)  # 0〜1
        except Exception as e:
            flash(f"予測でエラー: {e}")
            return redirect(request.url)

        pred_idx = 1 if prob_pos >= 0.5 else 0
        pred_label = label_map.get(pred_idx, str(pred_idx))

        # 「確信度」は予測クラス側の確率を表示
        confidence = prob_pos if pred_idx == 1 else (1.0 - prob_pos)

        pred_answer = (
            f"予測: {pred_label}（確信度: {confidence:.3f}）"
            f" / sigmoid出力: {prob_pos:.3f}"
        )

        # 可視化保存（モデル入力画像 + 参考pad画像）
        vis_input_url, vis_pad_url = save_visual_images(x_hwc, filename)

        return render_template(
            "index.html",
            answer=pred_answer,
            image_url=url_for("uploaded_file", filename=filename),  # 元画像
            vis_gray_url=vis_input_url,    # モデル入力（200x200）
            vis_invert_url=vis_pad_url,    # 参考（縦横比維持+pad）
        )

    # GET
    return render_template("index.html", answer="", image_url=None, vis_gray_url=None, vis_invert_url=None)


# ========================
# エントリポイント
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
