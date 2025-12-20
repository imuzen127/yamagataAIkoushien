# -*- coding: utf-8 -*-
"""
地理画像認識システム
- 学習フロー: サンプル画像と座標から教科書画像を生成
- 解答フロー: 問題画像と教科書画像をマッチング
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ===== 設定 =====
# スクリプトの場所を基準にパスを設定
SCRIPT_DIR = Path(__file__).parent.absolute()

# デフォルト設定（環境変数で上書き可能）
SAMPLE_DIR = os.environ.get('SAMPLE_DIR', str(SCRIPT_DIR / "data" / "施設画像サンプル（100施設）_20251120" / "施設画像サンプル"))
EXCEL_PATH = os.environ.get('EXCEL_PATH', str(SCRIPT_DIR / "data" / "施設情報（100施設）_20251031" / "施設情報（100施設）_20251031.xlsx"))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', str(SCRIPT_DIR / "textbook_images"))
MAPPING_FILE = os.environ.get('MAPPING_FILE', str(SCRIPT_DIR / "mapping.json"))

# ===== 座標変換関数 =====
def latlon2pixel(lat, lon, zoom):
    """緯度経度からピクセル座標への変換（国土地理院仕様）"""
    L = 85.05112878
    x = ((lon + 180) / 360) * 256 * (2 ** zoom)
    y = ((1 - (math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi)) / 2) * 256 * (2 ** zoom)
    return x, y

def pixel2latlon(x, y, zoom):
    """ピクセル座標から緯度経度への変換"""
    lon = (x / (256 * (2 ** zoom))) * 360 - 180
    n = math.pi - 2 * math.pi * y / (256 * (2 ** zoom))
    lat = math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))
    return lat, lon

# ===== タイル画像取得・結合 =====
def download_tile(zoom, x, y, max_retries=3):
    """国土地理院の航空写真タイルをダウンロード"""
    url = f"https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{zoom}/{x}/{y}.jpg"
    for attempt in range(max_retries):
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return Image.open(BytesIO(res.content)).convert('RGB')
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning: Failed to download tile {zoom}/{x}/{y}: {e}")
    # 失敗した場合は灰色画像を返す
    return Image.new('RGB', (256, 256), (128, 128, 128))

def generate_gsi_image(lat, lon, zoom, width_px, height_px):
    """指定座標を中心に、指定ピクセルサイズ分の画像を生成"""
    # 中心ピクセル座標
    center_x, center_y = latlon2pixel(lat, lon, zoom)

    # 必要なタイルの範囲を計算
    left = int((center_x - width_px / 2) / 256)
    right = int((center_x + width_px / 2) / 256)
    top = int((center_y - height_px / 2) / 256)
    bottom = int((center_y + height_px / 2) / 256)

    # キャンバス作成
    canvas_w = (right - left + 1) * 256
    canvas_h = (bottom - top + 1) * 256
    canvas = Image.new('RGB', (canvas_w, canvas_h))

    # タイルダウンロードと貼り付け
    for tx in range(left, right + 1):
        for ty in range(top, bottom + 1):
            tile = download_tile(zoom, tx, ty)
            canvas.paste(tile, ((tx - left) * 256, (ty - top) * 256))

    # 正確に中心から width x height で切り抜く
    crop_center_x = center_x - (left * 256)
    crop_center_y = center_y - (top * 256)

    left_crop = int(crop_center_x - (width_px / 2))
    top_crop = int(crop_center_y - (height_px / 2))

    return canvas.crop((left_crop, top_crop, left_crop + width_px, top_crop + height_px))

# ===== ズームレベル自動判定 =====
def detect_zoom_level(lat, lon, sample_img_cv, zoom_range=(14, 19)):
    """サンプル画像と最も一致するズームレベルを判定"""
    h, w = sample_img_cv.shape[:2]
    best_score = -1
    best_zoom = 17  # デフォルト

    for z in range(zoom_range[0], zoom_range[1]):
        try:
            # サンプルと同じサイズの画像を地図から生成
            test_img_pil = generate_gsi_image(lat, lon, z, w, h)
            test_img = np.array(test_img_pil)[:, :, ::-1].copy()  # PIL to OpenCV (BGR)

            # テンプレートマッチングでスコア計算
            # サンプル画像が少しずれている可能性を考慮し、少し広い範囲で検索
            if test_img.shape[0] >= sample_img_cv.shape[0] and test_img.shape[1] >= sample_img_cv.shape[1]:
                res = cv2.matchTemplate(test_img, sample_img_cv, cv2.TM_CCOEFF_NORMED)
                score = res.max()
            else:
                # サイズが小さい場合は逆にマッチング
                res = cv2.matchTemplate(sample_img_cv, test_img, cv2.TM_CCOEFF_NORMED)
                score = res.max()

            if score > best_score:
                best_score = score
                best_zoom = z

        except Exception as e:
            print(f"  Warning: Zoom {z} failed: {e}")
            continue

    return best_zoom, best_score

# ===== 画像読み込みヘルパー（日本語パス対応）=====
def imread_japanese(path):
    """日本語パスに対応した画像読み込み"""
    try:
        pil_img = Image.open(path).convert('RGB')
        return np.array(pil_img)[:, :, ::-1].copy()  # RGB to BGR for OpenCV
    except Exception as e:
        print(f"  Error reading image: {e}")
        return None

# ===== 学習フロー =====
def learning_flow():
    """教科書画像を生成する学習フロー"""
    print("=" * 60)
    print("学習フロー開始")
    print("=" * 60)

    # 出力ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Excelから施設情報を読み込み
    print("\n[1] 施設情報を読み込み中...")
    df = pd.read_excel(EXCEL_PATH)
    df.columns = ['施設番号', '施設名', '都道府県', '市区町村', '緯度', '経度']
    print(f"  {len(df)}件の施設情報を読み込みました")

    # マッピング情報を格納
    mapping = {}

    # 各施設を処理
    for idx, row in df.iterrows():
        facility_id = int(row['施設番号'])
        facility_name = str(row['施設名'])
        lat = float(row['緯度'])
        lon = float(row['経度'])

        print(f"\n[{facility_id:03d}/100] 処理中: {facility_name[:20]}...")

        # サンプル画像読み込み（PILを使用して日本語パス対応）
        sample_path = os.path.join(SAMPLE_DIR, f"map{facility_id:03d}.jpg")
        if not os.path.exists(sample_path):
            print(f"  Warning: Sample image not found: {sample_path}")
            continue

        sample_img = imread_japanese(sample_path)
        if sample_img is None:
            print(f"  Warning: Failed to read sample image: {sample_path}")
            continue

        h, w = sample_img.shape[:2]
        print(f"  サンプル画像サイズ: {w}x{h}")

        # ズームレベル自動判定
        print(f"  ズームレベル判定中 (Z14-Z18)...")
        best_zoom, score = detect_zoom_level(lat, lon, sample_img)
        print(f"  -> 判定結果: Zoom={best_zoom}, Score={score:.4f}")

        # 教科書画像生成（200%サイズ）
        print(f"  教科書画像生成中 ({w*2}x{h*2})...")
        textbook_img = generate_gsi_image(lat, lon, best_zoom, w * 2, h * 2)

        # 保存
        textbook_path = os.path.join(OUTPUT_DIR, f"textbook_{facility_id:03d}.png")
        textbook_img.save(textbook_path)
        print(f"  -> 保存: {textbook_path}")

        # マッピング情報保存
        mapping[str(facility_id)] = {
            'name': facility_name,
            'lat': lat,
            'lon': lon,
            'zoom': best_zoom,
            'score': float(score),
            'textbook_path': textbook_path
        }

    # マッピングファイル保存
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"\n[完了] マッピングファイル保存: {MAPPING_FILE}")

    return mapping

# ===== 解答フロー（サンプル画像を使用）=====
def answer_flow_sample(question_dir):
    """問題画像とサンプル画像を直接マッチングして解答を出力"""
    print("=" * 60)
    print("解答フロー開始（サンプル画像モード）")
    print("=" * 60)

    # サンプル画像を読み込み
    print("\n[1] サンプル画像を読み込み中...")
    samples = {}
    for i in range(1, 101):
        sample_path = os.path.join(SAMPLE_DIR, f"map{i:03d}.jpg")
        if os.path.exists(sample_path):
            img = imread_japanese(sample_path)
            if img is not None:
                samples[i] = img
    print(f"  {len(samples)}件のサンプル画像を読み込みました")

    # 問題画像を取得
    print(f"\n[2] 問題画像を読み込み中: {question_dir}")
    question_files = sorted([f for f in os.listdir(question_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  {len(question_files)}件の問題画像を検出")

    # マッチング（複数手法を併用）
    print("\n[3] マッチング実行中...")
    results = []

    for qf in question_files:
        question_path = os.path.join(question_dir, qf)
        question_img = imread_japanese(question_path)

        if question_img is None:
            print(f"  Warning: Failed to read: {qf}")
            results.append(-1)
            continue

        qh, qw = question_img.shape[:2]
        best_match_id = -1
        best_score = -1

        for fid, sample_img in samples.items():
            sh, sw = sample_img.shape[:2]

            try:
                # サンプルと問題のサイズが同じ場合は直接比較
                if sh == qh and sw == qw:
                    # ヒストグラム比較
                    q_hist = cv2.calcHist([question_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    s_hist = cv2.calcHist([sample_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist_score = cv2.compareHist(q_hist, s_hist, cv2.HISTCMP_CORREL)

                    # テンプレートマッチング（直接比較）
                    result = cv2.matchTemplate(sample_img, question_img, cv2.TM_CCOEFF_NORMED)
                    template_score = result.max()

                    # 複合スコア
                    score = 0.3 * hist_score + 0.7 * template_score
                else:
                    # サイズが異なる場合
                    if sh >= qh and sw >= qw:
                        # サンプルが大きい場合
                        result = cv2.matchTemplate(sample_img, question_img, cv2.TM_CCOEFF_NORMED)
                        score = result.max()
                    else:
                        # 問題画像が大きい場合
                        result = cv2.matchTemplate(question_img, sample_img, cv2.TM_CCOEFF_NORMED)
                        score = result.max()

                if score > best_score:
                    best_score = score
                    best_match_id = fid
            except Exception as e:
                continue

        print(f"  {qf} -> 施設番号 {best_match_id} (score={best_score:.4f})")
        results.append(best_match_id)

    # 結果出力
    print("\n[4] 結果出力")
    answer_text = ",".join(map(str, results))
    print(f"\n回答: {answer_text}")

    # ファイルに保存
    answer_path = os.path.join(question_dir, "answer.txt")
    with open(answer_path, 'w', encoding='utf-8') as f:
        f.write(answer_text)
    print(f"保存: {answer_path}")

    return results

# ===== 解答フロー（教科書画像を使用）=====
def answer_flow(question_dir):
    """問題画像と教科書画像をマッチングして解答を出力"""
    print("=" * 60)
    print("解答フロー開始（教科書画像モード）")
    print("=" * 60)

    # マッピング読み込み
    print("\n[1] マッピング情報読み込み中...")
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"  {len(mapping)}件の教科書画像情報を読み込みました")

    # 教科書画像をすべて読み込み
    print("\n[2] 教科書画像を読み込み中...")
    textbooks = {}
    for fid, info in mapping.items():
        path = info['textbook_path']
        if os.path.exists(path):
            img = imread_japanese(path)
            if img is not None:
                textbooks[fid] = img
    print(f"  {len(textbooks)}件の教科書画像を読み込みました")

    # 問題画像を取得
    print(f"\n[3] 問題画像を読み込み中: {question_dir}")
    question_files = sorted([f for f in os.listdir(question_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  {len(question_files)}件の問題画像を検出")

    # マッチング
    print("\n[4] マッチング実行中...")
    results = []

    for qf in question_files:
        question_path = os.path.join(question_dir, qf)
        question_img = imread_japanese(question_path)

        if question_img is None:
            print(f"  Warning: Failed to read: {qf}")
            results.append(-1)
            continue

        best_match_id = -1
        best_score = -1

        for fid, textbook_img in textbooks.items():
            try:
                # テンプレートマッチング
                res = cv2.matchTemplate(textbook_img, question_img, cv2.TM_CCOEFF_NORMED)
                score = res.max()

                if score > best_score:
                    best_score = score
                    best_match_id = int(fid)
            except Exception as e:
                continue

        print(f"  {qf} -> 施設番号 {best_match_id} (score={best_score:.4f})")
        results.append(best_match_id)

    # 結果出力
    print("\n[5] 結果出力")
    answer_text = ",".join(map(str, results))
    print(f"\n回答: {answer_text}")

    # ファイルに保存
    answer_path = os.path.join(question_dir, "answer.txt")
    with open(answer_path, 'w', encoding='utf-8') as f:
        f.write(answer_text)
    print(f"保存: {answer_path}")

    return results

# ===== メイン =====
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("  学習フロー:           python geo_matcher.py learn")
        print("  解答フロー(教科書):   python geo_matcher.py answer <問題画像フォルダ>")
        print("  解答フロー(サンプル): python geo_matcher.py answer-sample <問題画像フォルダ>")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "learn":
        learning_flow()
    elif mode == "answer":
        if len(sys.argv) < 3:
            print("Error: 問題画像フォルダを指定してください")
            sys.exit(1)
        question_dir = sys.argv[2]
        answer_flow(question_dir)
    elif mode == "answer-sample":
        if len(sys.argv) < 3:
            print("Error: 問題画像フォルダを指定してください")
            sys.exit(1)
        question_dir = sys.argv[2]
        answer_flow_sample(question_dir)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
