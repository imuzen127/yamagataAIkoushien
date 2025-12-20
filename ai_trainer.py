# -*- coding: utf-8 -*-
"""
AI画像認識システム - 学習・推論モジュール
- 教科書画像からランダムクロップで学習データ生成
- CNNモデルによる分類学習
- 問題画像の推論
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import json
import random
from pathlib import Path

# ===== 設定 =====
SCRIPT_DIR = Path(__file__).parent.absolute()
TEXTBOOK_DIR = SCRIPT_DIR / "textbook_images"
TRAINING_DIR = SCRIPT_DIR / "training_data"
MODEL_PATH = SCRIPT_DIR / "model.pth"
NUM_CLASSES = 100  # 施設数

# 画像サイズ設定
CROP_SIZE = 64  # 64x64 または 128x128
INPUT_SIZE = 64  # モデル入力サイズ

# ===== データ生成 =====
def generate_training_data(num_crops_per_image=50, crop_size=CROP_SIZE, train_ratio=0.8):
    """教科書画像からランダムクロップで学習データを生成"""
    print("=" * 60)
    print("学習データ生成開始")
    print("=" * 60)

    # 出力ディレクトリ作成
    train_dir = TRAINING_DIR / "train"
    test_dir = TRAINING_DIR / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # ラベルファイル
    train_labels = []
    test_labels = []

    for i in range(1, NUM_CLASSES + 1):
        textbook_path = TEXTBOOK_DIR / f"textbook_{i:03d}.png"
        if not textbook_path.exists():
            print(f"  Warning: {textbook_path} not found")
            continue

        img = Image.open(textbook_path).convert('RGB')
        w, h = img.size

        # クラスごとのサブディレクトリ
        (train_dir / f"{i:03d}").mkdir(exist_ok=True)
        (test_dir / f"{i:03d}").mkdir(exist_ok=True)

        for j in range(num_crops_per_image):
            # ランダムな位置でクロップ
            max_x = w - crop_size
            max_y = h - crop_size
            if max_x <= 0 or max_y <= 0:
                continue

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            crop = img.crop((x, y, x + crop_size, y + crop_size))

            # 学習/テスト分割
            if random.random() < train_ratio:
                crop_path = train_dir / f"{i:03d}" / f"crop_{i:03d}_{j:04d}.png"
                crop.save(crop_path)
                train_labels.append({"file": str(crop_path), "label": i})
            else:
                crop_path = test_dir / f"{i:03d}" / f"crop_{i:03d}_{j:04d}.png"
                crop.save(crop_path)
                test_labels.append({"file": str(crop_path), "label": i})

        if i % 10 == 0:
            print(f"  処理済み: {i}/100")

    # ラベルファイル保存
    with open(TRAINING_DIR / "train_labels.json", 'w') as f:
        json.dump(train_labels, f)
    with open(TRAINING_DIR / "test_labels.json", 'w') as f:
        json.dump(test_labels, f)

    print(f"\n生成完了:")
    print(f"  学習データ: {len(train_labels)}枚")
    print(f"  テストデータ: {len(test_labels)}枚")

    return len(train_labels), len(test_labels)

# ===== データセット =====
class CropDataset(Dataset):
    def __init__(self, label_file, transform=None):
        with open(label_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['file']).convert('RGB')
        label = item['label'] - 1  # 0-indexed

        if self.transform:
            img = self.transform(img)

        return img, label

# ===== CNNモデル =====
class GeoClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE):
        super(GeoClassifier, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # Conv Block 4
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

# ===== 学習 =====
def train_model(epochs=30, batch_size=32, learning_rate=0.001):
    """モデルを学習"""
    print("=" * 60)
    print("モデル学習開始")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    # データ変換
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # データセット
    train_dataset = CropDataset(TRAINING_DIR / "train_labels.json", train_transform)
    test_dataset = CropDataset(TRAINING_DIR / "test_labels.json", test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"学習データ: {len(train_dataset)}枚")
    print(f"テストデータ: {len(test_dataset)}枚")

    # モデル
    model = GeoClassifier(NUM_CLASSES, INPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0.0

    for epoch in range(epochs):
        # 学習
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # 評価
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Test Acc: {test_acc:.2f}%")

        # ベストモデル保存
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, MODEL_PATH)
            print(f"  -> Best model saved (acc: {best_acc:.2f}%)")

        scheduler.step()

    print(f"\n学習完了! 最高精度: {best_acc:.2f}%")
    return best_acc

# ===== 推論 =====
def predict(question_dir, output_file="answer.txt"):
    """問題画像を分類して回答を出力"""
    print("=" * 60)
    print("AI推論開始")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルロード
    model = GeoClassifier(NUM_CLASSES, INPUT_SIZE).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"モデルロード完了 (精度: {checkpoint['best_acc']:.2f}%)")

    # 変換
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 問題画像取得
    question_dir = Path(question_dir)
    question_files = sorted([f for f in os.listdir(question_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"問題画像: {len(question_files)}枚")

    results = []

    for qf in question_files:
        img_path = question_dir / qf
        img = Image.open(img_path).convert('RGB')

        # 複数クロップで投票
        w, h = img.size
        votes = {}

        # 中央と4隅からクロップ
        crop_positions = [
            (w//2 - CROP_SIZE//2, h//2 - CROP_SIZE//2),  # 中央
            (0, 0),  # 左上
            (w - CROP_SIZE, 0),  # 右上
            (0, h - CROP_SIZE),  # 左下
            (w - CROP_SIZE, h - CROP_SIZE),  # 右下
        ]

        # ランダムクロップも追加
        for _ in range(5):
            x = random.randint(0, max(0, w - CROP_SIZE))
            y = random.randint(0, max(0, h - CROP_SIZE))
            crop_positions.append((x, y))

        for (x, y) in crop_positions:
            x = max(0, min(x, w - CROP_SIZE))
            y = max(0, min(y, h - CROP_SIZE))
            crop = img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))

            input_tensor = transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                top_prob, top_idx = probs.topk(3)

                for prob, idx in zip(top_prob[0], top_idx[0]):
                    label = idx.item() + 1
                    if label not in votes:
                        votes[label] = 0
                    votes[label] += prob.item()

        # 最も投票数の多いラベルを選択
        best_label = max(votes, key=votes.get)
        confidence = votes[best_label] / sum(votes.values()) * 100

        print(f"  {qf} -> 施設番号 {best_label} (信頼度: {confidence:.1f}%)")
        results.append(best_label)

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
        print("  データ生成: python ai_trainer.py generate [crops_per_image]")
        print("  学習:       python ai_trainer.py train [epochs]")
        print("  推論:       python ai_trainer.py predict <問題画像フォルダ>")
        print("  全実行:     python ai_trainer.py all")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "generate":
        num_crops = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        generate_training_data(num_crops_per_image=num_crops)

    elif mode == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        train_model(epochs=epochs)

    elif mode == "predict":
        if len(sys.argv) < 3:
            print("Error: 問題画像フォルダを指定してください")
            sys.exit(1)
        predict(sys.argv[2])

    elif mode == "all":
        generate_training_data(num_crops_per_image=50)
        train_model(epochs=30)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
