# -*- coding: utf-8 -*-
"""
AI画像認識システム v2 - 反復学習・反省機構付き
- 大量の学習データ生成
- 反復学習とエラー分析
- 誤分類サンプルの重点学習
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import json
import random
from pathlib import Path
from collections import defaultdict
import time

# ===== 設定 =====
SCRIPT_DIR = Path(__file__).parent.absolute()
TEXTBOOK_DIR = SCRIPT_DIR / "textbook_images"
SAMPLE_DIR = SCRIPT_DIR / "data" / "施設画像サンプル（100施設）_20251120" / "施設画像サンプル"
TRAINING_DIR = SCRIPT_DIR / "training_data_v2"
MODEL_PATH = SCRIPT_DIR / "model_v2.pth"
REPORT_PATH = SCRIPT_DIR / "training_report.json"
NUM_CLASSES = 100

# 画像サイズ設定
CROP_SIZES = [64, 96, 128]  # 複数サイズでクロップ
INPUT_SIZE = 128  # モデル入力サイズ（大きくして情報量増加）

# ===== 改良版データ生成 =====
def generate_training_data_v2(num_crops_per_image=200, train_ratio=0.8):
    """より多くの学習データを生成（複数サイズ、複数ソース）"""
    print("=" * 60)
    print("学習データ生成 v2")
    print("=" * 60)

    train_dir = TRAINING_DIR / "train"
    test_dir = TRAINING_DIR / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_labels = []
    test_labels = []

    for i in range(1, NUM_CLASSES + 1):
        # 教科書画像とサンプル画像の両方を使用
        sources = []

        textbook_path = TEXTBOOK_DIR / f"textbook_{i:03d}.png"
        if textbook_path.exists():
            sources.append(("textbook", Image.open(textbook_path).convert('RGB')))

        sample_path = SAMPLE_DIR / f"map{i:03d}.jpg"
        if sample_path.exists():
            sources.append(("sample", Image.open(sample_path).convert('RGB')))

        if not sources:
            print(f"  Warning: No images for facility {i}")
            continue

        (train_dir / f"{i:03d}").mkdir(exist_ok=True)
        (test_dir / f"{i:03d}").mkdir(exist_ok=True)

        crop_count = 0
        for source_name, img in sources:
            w, h = img.size
            crops_per_source = num_crops_per_image // len(sources)

            for j in range(crops_per_source):
                # ランダムなクロップサイズ
                crop_size = random.choice(CROP_SIZES)

                if w < crop_size or h < crop_size:
                    # 画像が小さい場合はリサイズ
                    crop = img.resize((crop_size, crop_size), Image.LANCZOS)
                else:
                    # ランダム位置でクロップ
                    x = random.randint(0, w - crop_size)
                    y = random.randint(0, h - crop_size)
                    crop = img.crop((x, y, x + crop_size, y + crop_size))

                # 入力サイズにリサイズ
                crop = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

                # 学習/テスト分割
                if random.random() < train_ratio:
                    crop_path = train_dir / f"{i:03d}" / f"{source_name}_{i:03d}_{crop_count:05d}.png"
                    crop.save(crop_path)
                    train_labels.append({"file": str(crop_path), "label": i})
                else:
                    crop_path = test_dir / f"{i:03d}" / f"{source_name}_{i:03d}_{crop_count:05d}.png"
                    crop.save(crop_path)
                    test_labels.append({"file": str(crop_path), "label": i})

                crop_count += 1

        if i % 20 == 0:
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
class CropDatasetV2(Dataset):
    def __init__(self, label_file, transform=None, hard_samples=None):
        with open(label_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        # 難しいサンプルを追加（重点学習用）
        if hard_samples:
            self.data.extend(hard_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['file']).convert('RGB')
        label = item['label'] - 1

        if self.transform:
            img = self.transform(img)

        return img, label

# ===== 改良版CNNモデル（ResNet系）=====
class GeoClassifierV2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(GeoClassifierV2, self).__init__()

        # より深いネットワーク
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===== エラー分析 =====
def analyze_errors(model, test_loader, device):
    """誤分類を分析して弱点を特定"""
    model.eval()
    errors = defaultdict(list)  # label -> [(predicted, file), ...]
    confusion = defaultdict(lambda: defaultdict(int))  # true -> pred -> count

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i, (pred, true) in enumerate(zip(predicted, labels)):
                pred_label = pred.item() + 1
                true_label = true.item() + 1
                confusion[true_label][pred_label] += 1

                if pred_label != true_label:
                    errors[true_label].append(pred_label)

    # 最も誤分類が多いクラスを特定
    error_counts = {label: len(errs) for label, errs in errors.items()}
    sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])

    return sorted_errors, dict(confusion)

# ===== 反復学習 =====
def iterative_train(max_iterations=5, epochs_per_iter=20, target_accuracy=95.0):
    """反復学習：精度が目標に達するまで繰り返す"""
    print("=" * 60, flush=True)
    print("反復学習開始", flush=True)
    print(f"目標精度: {target_accuracy}%", flush=True)
    print("=" * 60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}", flush=True)

    # データ変換（強いaugmentation）
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 学習履歴
    history = {
        'iterations': [],
        'best_accuracy': 0,
        'error_analysis': []
    }

    best_overall_acc = 0
    hard_samples = []  # 難しいサンプル（誤分類されやすい）

    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"反復 {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")

        # データセット
        train_dataset = CropDatasetV2(
            TRAINING_DIR / "train_labels.json",
            train_transform,
            hard_samples if iteration > 0 else None
        )
        test_dataset = CropDatasetV2(
            TRAINING_DIR / "test_labels.json",
            test_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        print(f"学習データ: {len(train_dataset)}枚")
        print(f"テストデータ: {len(test_dataset)}枚")

        # モデル（前回のモデルを継続、または新規）
        model = GeoClassifierV2(NUM_CLASSES).to(device)

        if iteration > 0 and MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("前回のモデルを継続学習")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_iter)

        best_acc = 0

        for epoch in range(epochs_per_iter):
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

            print(f"  Epoch [{epoch+1}/{epochs_per_iter}] "
                  f"Loss: {train_loss/len(train_loader):.4f} "
                  f"Train: {train_acc:.2f}% "
                  f"Test: {test_acc:.2f}%", flush=True)

            if test_acc > best_acc:
                best_acc = test_acc
                if test_acc > best_overall_acc:
                    best_overall_acc = test_acc
                    torch.save({
                        'iteration': iteration,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }, MODEL_PATH)
                    print(f"    -> ベストモデル保存 (acc: {best_acc:.2f}%)")

            scheduler.step()

        # エラー分析
        print(f"\n[エラー分析]")
        error_list, confusion = analyze_errors(model, test_loader, device)

        if error_list:
            print(f"  誤分類が多いクラス (Top 10):")
            for label, count in error_list[:10]:
                print(f"    施設{label}: {count}件の誤分類")

            # 難しいサンプルを特定して次回の学習に追加
            weak_classes = [label for label, count in error_list[:20] if count > 0]
            hard_samples = []
            with open(TRAINING_DIR / "train_labels.json", 'r') as f:
                all_train = json.load(f)
            for item in all_train:
                if item['label'] in weak_classes:
                    # 弱いクラスのサンプルを複製
                    hard_samples.extend([item] * 3)

            print(f"  次回の重点学習サンプル: {len(hard_samples)}枚追加")

        history['iterations'].append({
            'iteration': iteration + 1,
            'best_accuracy': best_acc,
            'weak_classes': error_list[:10] if error_list else []
        })

        # 目標精度に到達したら終了
        if best_acc >= target_accuracy:
            print(f"\n目標精度 {target_accuracy}% に到達しました！")
            break

    # レポート保存
    history['best_accuracy'] = best_overall_acc
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n学習完了!")
    print(f"最終精度: {best_overall_acc:.2f}%")
    print(f"レポート: {REPORT_PATH}")

    return best_overall_acc

# ===== AI推論（単独）=====
def predict_ai_only(question_dir, output_file="answer.txt"):
    """AIのみで推論（ハイブリッドなし）"""
    print("=" * 60)
    print("AI推論（単独モード）")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルロード
    model = GeoClassifierV2(NUM_CLASSES).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"モデルロード完了 (精度: {checkpoint['best_acc']:.2f}%)")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TTA（Test Time Augmentation）用
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

    # 問題画像取得
    question_dir = Path(question_dir)
    question_files = sorted([f for f in os.listdir(question_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"問題画像: {len(question_files)}枚")

    results = []

    for qf in question_files:
        img_path = question_dir / qf
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # 複数クロップ + TTA で投票
        votes = defaultdict(float)
        crop_size = min(w, h, INPUT_SIZE)

        # クロップ位置
        positions = [
            (w//2 - crop_size//2, h//2 - crop_size//2),  # 中央
            (0, 0), (w - crop_size, 0),  # 上
            (0, h - crop_size), (w - crop_size, h - crop_size),  # 下
        ]

        # ランダムクロップ追加
        for _ in range(10):
            x = random.randint(0, max(0, w - crop_size))
            y = random.randint(0, max(0, h - crop_size))
            positions.append((x, y))

        for (x, y) in positions:
            x = max(0, min(x, w - crop_size))
            y = max(0, min(y, h - crop_size))

            if crop_size < w or crop_size < h:
                crop = img.crop((x, y, x + crop_size, y + crop_size))
            else:
                crop = img

            # TTA
            for tta_tf in tta_transforms:
                input_tensor = tta_tf(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    top_probs, top_indices = probs.topk(5)

                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        label = idx.item() + 1
                        votes[label] += prob.item()

        # 最も投票数の多いラベル
        best_label = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[best_label] / total_votes * 100

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
        print("  データ生成:   python ai_trainer_v2.py generate [crops_per_image]")
        print("  反復学習:     python ai_trainer_v2.py train [iterations] [epochs] [target_acc]")
        print("  推論(AIのみ): python ai_trainer_v2.py predict <問題画像フォルダ>")
        print("  全実行:       python ai_trainer_v2.py all")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "generate":
        num_crops = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        generate_training_data_v2(num_crops_per_image=num_crops)

    elif mode == "train":
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        target = float(sys.argv[4]) if len(sys.argv) > 4 else 95.0
        iterative_train(max_iterations=iterations, epochs_per_iter=epochs, target_accuracy=target)

    elif mode == "predict":
        if len(sys.argv) < 3:
            print("Error: 問題画像フォルダを指定してください")
            sys.exit(1)
        predict_ai_only(sys.argv[2])

    elif mode == "all":
        generate_training_data_v2(num_crops_per_image=200)
        iterative_train(max_iterations=5, epochs_per_iter=30, target_accuracy=95.0)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
