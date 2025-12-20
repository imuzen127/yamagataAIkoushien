# -*- coding: utf-8 -*-
"""
ハイブリッド画像認識システム
- AIによる候補絞り込み（高速）
- テンプレートマッチングによる最終判定（高精度）
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import os
import json
from pathlib import Path

# ===== 設定 =====
SCRIPT_DIR = Path(__file__).parent.absolute()
SAMPLE_DIR = SCRIPT_DIR / "data" / "施設画像サンプル（100施設）_20251120" / "施設画像サンプル"
TEXTBOOK_DIR = SCRIPT_DIR / "textbook_images"
MODEL_PATH = SCRIPT_DIR / "model.pth"
NUM_CLASSES = 100
CROP_SIZE = 64
INPUT_SIZE = 64

# ===== AIモデル定義 =====
class GeoClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE):
        super(GeoClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===== 画像読み込み =====
def imread_pil(path):
    """PILで画像読み込み（日本語パス対応）"""
    return Image.open(path).convert('RGB')

def pil_to_cv2(pil_img):
    """PIL -> OpenCV変換"""
    return np.array(pil_img)[:, :, ::-1].copy()

# ===== ハイブリッドマッチャー =====
class HybridMatcher:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.samples = {}
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """AIモデルをロード"""
        if MODEL_PATH.exists():
            self.model = GeoClassifier(NUM_CLASSES, INPUT_SIZE).to(self.device)
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"AIモデルロード完了 (精度: {checkpoint['best_acc']:.2f}%)")
            return True
        else:
            print("Warning: AIモデルが見つかりません。テンプレートマッチングのみ使用します。")
            return False

    def load_samples(self):
        """サンプル画像をロード"""
        for i in range(1, NUM_CLASSES + 1):
            sample_path = SAMPLE_DIR / f"map{i:03d}.jpg"
            if sample_path.exists():
                img = imread_pil(sample_path)
                self.samples[i] = pil_to_cv2(img)
        print(f"サンプル画像ロード完了: {len(self.samples)}枚")

    def get_ai_candidates(self, img, top_k=10):
        """AIで上位候補を取得"""
        if self.model is None:
            return list(range(1, NUM_CLASSES + 1))  # モデルがない場合は全候補

        w, h = img.size
        votes = {}

        # 複数クロップで投票
        crop_positions = [
            (w//2 - CROP_SIZE//2, h//2 - CROP_SIZE//2),
            (0, 0), (w - CROP_SIZE, 0),
            (0, h - CROP_SIZE), (w - CROP_SIZE, h - CROP_SIZE),
        ]

        import random
        for _ in range(10):
            x = random.randint(0, max(0, w - CROP_SIZE))
            y = random.randint(0, max(0, h - CROP_SIZE))
            crop_positions.append((x, y))

        for (x, y) in crop_positions:
            x = max(0, min(x, w - CROP_SIZE))
            y = max(0, min(y, h - CROP_SIZE))
            crop = img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
            input_tensor = self.transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                top_probs, top_indices = probs.topk(5)

                for prob, idx in zip(top_probs[0], top_indices[0]):
                    label = idx.item() + 1
                    if label not in votes:
                        votes[label] = 0
                    votes[label] += prob.item()

        # 上位候補を返す
        sorted_candidates = sorted(votes.items(), key=lambda x: -x[1])
        return [c[0] for c in sorted_candidates[:top_k]]

    def template_match(self, question_img_cv, candidate_ids):
        """テンプレートマッチングで最終判定"""
        best_id = -1
        best_score = -1

        qh, qw = question_img_cv.shape[:2]

        for fid in candidate_ids:
            if fid not in self.samples:
                continue

            sample_img = self.samples[fid]
            sh, sw = sample_img.shape[:2]

            try:
                if sh == qh and sw == qw:
                    # 同サイズ: 直接比較
                    result = cv2.matchTemplate(sample_img, question_img_cv, cv2.TM_CCOEFF_NORMED)
                    score = result.max()
                elif sh >= qh and sw >= qw:
                    result = cv2.matchTemplate(sample_img, question_img_cv, cv2.TM_CCOEFF_NORMED)
                    score = result.max()
                else:
                    result = cv2.matchTemplate(question_img_cv, sample_img, cv2.TM_CCOEFF_NORMED)
                    score = result.max()

                if score > best_score:
                    best_score = score
                    best_id = fid

            except Exception:
                continue

        return best_id, best_score

    def predict(self, question_dir, output_file="answer.txt", use_ai=True, top_k=20):
        """問題画像を分類"""
        print("=" * 60)
        print("ハイブリッド推論開始")
        print("=" * 60)

        # 初期化
        if use_ai:
            self.load_model()
        self.load_samples()

        # 問題画像取得
        question_dir = Path(question_dir)
        question_files = sorted([f for f in os.listdir(question_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"問題画像: {len(question_files)}枚")

        results = []

        for qf in question_files:
            img_path = question_dir / qf
            img_pil = imread_pil(img_path)
            img_cv = pil_to_cv2(img_pil)

            if use_ai and self.model is not None:
                # AIで候補絞り込み
                candidates = self.get_ai_candidates(img_pil, top_k=top_k)
            else:
                candidates = list(range(1, NUM_CLASSES + 1))

            # テンプレートマッチングで最終判定
            best_id, score = self.template_match(img_cv, candidates)

            print(f"  {qf} -> 施設番号 {best_id} (score={score:.4f})")
            results.append(best_id)

        # 結果出力
        answer_text = ",".join(map(str, results))
        output_path = question_dir / output_file
        with open(output_path, 'w') as f:
            f.write(answer_text)

        print(f"\n回答: {answer_text}")
        print(f"保存: {output_path}")

        return results

# ===== メイン =====
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python hybrid_matcher.py <問題画像フォルダ> [--no-ai]")
        sys.exit(1)

    question_dir = sys.argv[1]
    use_ai = "--no-ai" not in sys.argv

    matcher = HybridMatcher()
    matcher.predict(question_dir, use_ai=use_ai)
