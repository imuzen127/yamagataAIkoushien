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

# 元のズームレベル（mapping.jsonから取得）
ORIGINAL_ZOOM_LEVELS = {
    1: 15, 2: 17, 3: 18, 4: 18, 5: 17, 6: 16, 7: 16, 8: 17, 9: 16, 10: 18,
    11: 16, 12: 17, 13: 16, 14: 14, 15: 17, 16: 16, 17: 18, 18: 16, 19: 18, 20: 17,
    21: 16, 22: 18, 23: 15, 24: 16, 25: 16, 26: 18, 27: 17, 28: 16, 29: 17, 30: 14,
    31: 17, 32: 17, 33: 16, 34: 14, 35: 17, 36: 15, 37: 16, 38: 15, 39: 17, 40: 18,
    41: 17, 42: 18, 43: 16, 44: 18, 45: 18, 46: 17, 47: 17, 48: 16, 49: 16, 50: 18,
    51: 17, 52: 17, 53: 17, 54: 17, 55: 18, 56: 18, 57: 18, 58: 17, 59: 14, 60: 17,
    61: 18, 62: 18, 63: 17, 64: 18, 65: 15, 66: 15, 67: 17, 68: 16, 69: 15, 70: 17,
    71: 16, 72: 16, 73: 15, 74: 14, 75: 17, 76: 17, 77: 15, 78: 14, 79: 17, 80: 17,
    81: 17, 82: 16, 83: 16, 84: 16, 85: 16, 86: 15, 87: 15, 88: 18, 89: 17, 90: 18,
    91: 17, 92: 14, 93: 18, 94: 16, 95: 15, 96: 15, 97: 15, 98: 16, 99: 17, 100: 14,
}

def calculate_scale(facility_id):
    """元のズームレベルに基づいてスケールを計算"""
    original_zoom = ORIGINAL_ZOOM_LEVELS.get(facility_id, 17)
    # スケール = 2.0 × 2^(18 - 元のズーム)
    scale = DEFAULT_SCALE * (2 ** (DEFAULT_ZOOM - original_zoom))
    return scale

# 手動設定した施設（上書きしない）
MANUAL_FACILITIES = {45, 65, 66, 67, 68, 69}

# 施設別カスタム設定 (手動設定のみ)
FACILITY_CONFIG = {
    # 45番: 手動設定
    45: {'scale': 4.0, 'zoom': 18},  # 400%

    # 65,66,67番: 手動設定
    65: {'scale': 3.5, 'zoom': 18},  # 350%
    66: {'scale': 3.5, 'zoom': 18},  # 350%
    67: {'scale': 2.5, 'zoom': 18},  # 250%

    # 68,69番: 位置が近いため手動調整
    68: {'scale': 2.5, 'zoom': 18, 'offset_lon': -0.002},  # 250%, 西にずらす
    69: {'scale': 2.7, 'zoom': 18, 'offset_lon': -0.0005},  # 270%, 西にずらす
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
        if facility_id in MANUAL_FACILITIES:
            # 手動設定の施設はそのまま使用
            config = FACILITY_CONFIG.get(facility_id, {})
            zoom = config.get('zoom', DEFAULT_ZOOM)
            scale = config.get('scale', DEFAULT_SCALE)
        else:
            # 自動計算：元のズームレベルに基づいてスケールを算出
            zoom = DEFAULT_ZOOM
            scale = calculate_scale(facility_id)

        config = FACILITY_CONFIG.get(facility_id, {})
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
