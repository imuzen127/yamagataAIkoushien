# -*- coding: utf-8 -*-
"""
地理画像認識システム v2 - 高解像度版
- 最大ズームレベル(Z18)でタイル取得
- 施設ごとのカスタムスケール設定
- 問題のある施設の特別対応
"""

import cv2
import math
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ===== 設定 =====
SCRIPT_DIR = Path(__file__).parent.absolute()
SAMPLE_DIR = SCRIPT_DIR / "data" / "施設画像サンプル（100施設）_20251120" / "施設画像サンプル"
EXCEL_PATH = SCRIPT_DIR / "data" / "施設情報（100施設）_20251031" / "施設情報（100施設）_20251031.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "textbook_images_v2"
MAPPING_FILE = SCRIPT_DIR / "mapping_v2.json"

# デフォルト設定
DEFAULT_ZOOM = 18  # 最大ズームレベル
DEFAULT_SCALE = 2.0  # デフォルト200%

# 施設別カスタム設定 (問題のある施設を調整)
FACILITY_CONFIG = {
    # 45番: 範囲をさらに広げる
    45: {'scale': 4.0, 'zoom': 18},  # 400%に拡大

    # 65,66,67番: オフセットなし
    65: {'scale': 3.5, 'zoom': 18},  # 350%
    66: {'scale': 3.5, 'zoom': 18},  # 350%
    67: {'scale': 2.5, 'zoom': 18},  # 250%

    # 73番 (45と混同された)
    73: {'scale': 2.5, 'zoom': 18},

    # 70番 (65と混同された)
    70: {'scale': 2.0, 'zoom': 18},
}

# ===== 座標変換関数 =====
def latlon2pixel(lat, lon, zoom):
    """緯度経度からピクセル座標への変換"""
    x = ((lon + 180) / 360) * 256 * (2 ** zoom)
    y = ((1 - (math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi)) / 2) * 256 * (2 ** zoom)
    return x, y

# ===== タイル画像取得 =====
def download_tile(zoom, x, y, max_retries=3):
    """国土地理院の航空写真タイルをダウンロード (最大解像度)"""
    url = f"https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{zoom}/{x}/{y}.jpg"
    for attempt in range(max_retries):
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return Image.open(BytesIO(res.content)).convert('RGB')
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Failed to download tile {zoom}/{x}/{y}: {e}")
    return Image.new('RGB', (256, 256), (128, 128, 128))

def generate_gsi_image_v2(lat, lon, zoom, width_px, height_px):
    """指定座標を中心に、最大解像度で画像を生成"""
    center_x, center_y = latlon2pixel(lat, lon, zoom)

    # 必要なタイルの範囲を計算 (余裕を持って取得)
    left = int((center_x - width_px / 2) / 256) - 1
    right = int((center_x + width_px / 2) / 256) + 1
    top = int((center_y - height_px / 2) / 256) - 1
    bottom = int((center_y + height_px / 2) / 256) + 1

    # キャンバス作成
    canvas_w = (right - left + 1) * 256
    canvas_h = (bottom - top + 1) * 256
    canvas = Image.new('RGB', (canvas_w, canvas_h))

    # タイルダウンロードと貼り付け
    for tx in range(left, right + 1):
        for ty in range(top, bottom + 1):
            tile = download_tile(zoom, tx, ty)
            canvas.paste(tile, ((tx - left) * 256, (ty - top) * 256))

    # 中心から正確に切り抜く
    crop_center_x = center_x - (left * 256)
    crop_center_y = center_y - (top * 256)

    left_crop = int(crop_center_x - (width_px / 2))
    top_crop = int(crop_center_y - (height_px / 2))

    cropped = canvas.crop((left_crop, top_crop, left_crop + width_px, top_crop + height_px))
    return cropped

# ===== 画像読み込み =====
def imread_japanese(path):
    """日本語パスに対応した画像読み込み"""
    try:
        pil_img = Image.open(path).convert('RGB')
        return np.array(pil_img)[:, :, ::-1].copy()
    except Exception as e:
        print(f"  Error reading image: {e}")
        return None

# ===== 学習フロー v2 =====
def learning_flow_v2(facility_ids=None):
    """
    教科書画像を高解像度で生成
    facility_ids: 指定した施設のみ生成 (Noneなら全施設)
    """
    print("=" * 60)
    print("学習フロー v2 (高解像度版)")
    print(f"デフォルトズーム: Z{DEFAULT_ZOOM}")
    print(f"デフォルトスケール: {DEFAULT_SCALE * 100}%")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Excel読み込み
    print("\n[1] 施設情報を読み込み中...")
    df = pd.read_excel(EXCEL_PATH)
    df.columns = ['施設番号', '施設名', '都道府県', '市区町村', '緯度', '経度']
    print(f"  {len(df)}件の施設情報を読み込みました")

    # 既存マッピング読み込み (あれば)
    mapping = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

    # 処理対象の施設
    if facility_ids:
        target_ids = facility_ids
        print(f"\n指定施設のみ処理: {target_ids}")
    else:
        target_ids = list(range(1, 101))

    # 各施設を処理
    for idx, row in df.iterrows():
        facility_id = int(row['施設番号'])

        if facility_id not in target_ids:
            continue

        facility_name = str(row['施設名'])
        lat = float(row['緯度'])
        lon = float(row['経度'])

        # カスタム設定を取得
        config = FACILITY_CONFIG.get(facility_id, {})
        zoom = config.get('zoom', DEFAULT_ZOOM)
        scale = config.get('scale', DEFAULT_SCALE)
        offset_lat = config.get('offset_lat', 0)
        offset_lon = config.get('offset_lon', 0)

        # オフセット適用
        lat += offset_lat
        lon += offset_lon

        print(f"\n[{facility_id:03d}] {facility_name[:20]}...")
        print(f"  設定: Z{zoom}, Scale={scale*100}%", end="")
        if offset_lat or offset_lon:
            print(f", Offset=({offset_lat}, {offset_lon})", end="")
        print()

        # サンプル画像読み込み
        sample_path = SAMPLE_DIR / f"map{facility_id:03d}.jpg"
        if not sample_path.exists():
            print(f"  Warning: Sample not found")
            continue

        sample_img = imread_japanese(str(sample_path))
        if sample_img is None:
            continue

        h, w = sample_img.shape[:2]

        # 教科書画像生成 (カスタムスケール)
        out_w = int(w * scale)
        out_h = int(h * scale)
        print(f"  生成中: {out_w}x{out_h} (元{w}x{h})")

        textbook_img = generate_gsi_image_v2(lat, lon, zoom, out_w, out_h)

        # 保存
        textbook_path = OUTPUT_DIR / f"textbook_{facility_id:03d}.png"
        textbook_img.save(textbook_path)
        print(f"  -> 保存: {textbook_path}")

        # マッピング更新
        mapping[str(facility_id)] = {
            'name': facility_name,
            'lat': lat,
            'lon': lon,
            'zoom': zoom,
            'scale': scale,
            'textbook_path': str(textbook_path)
        }

    # マッピング保存
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"\n[完了] マッピング保存: {MAPPING_FILE}")

    return mapping

# ===== 特定施設のみ再生成 =====
def regenerate_facilities(facility_ids):
    """指定した施設の教科書画像のみ再生成"""
    print(f"再生成対象: {facility_ids}")
    return learning_flow_v2(facility_ids=facility_ids)

# ===== メイン =====
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("  全施設生成:     python geo_matcher_v2.py all")
        print("  指定施設のみ:   python geo_matcher_v2.py regenerate 45 65 66 67")
        print("  問題施設のみ:   python geo_matcher_v2.py fix")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "all":
        learning_flow_v2()
    elif mode == "regenerate":
        if len(sys.argv) < 3:
            print("Error: 施設番号を指定してください")
            sys.exit(1)
        ids = [int(x) for x in sys.argv[2:]]
        regenerate_facilities(ids)
    elif mode == "fix":
        # 問題のある施設を再生成
        problem_ids = list(FACILITY_CONFIG.keys())
        print(f"問題施設を再生成: {problem_ids}")
        regenerate_facilities(problem_ids)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
