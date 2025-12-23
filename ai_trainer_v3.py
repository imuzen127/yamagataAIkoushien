# -*- coding: utf-8 -*-
"""
AI画像認識システム v3 - 改善版
- 入力サイズ拡大 (128→224)
- クロップサイズ拡大
- 特徴抽出にResNet活用オプション
- 苦手クラス重点学習
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import json
import random
from pathlib import Path
from collections import defaultdict
import time
from datetime import datetime, timezone, timedelta

# ===== 設定 =====
SCRIPT_DIR = Path(__file__).parent.absolute()
TEXTBOOK_DIR = SCRIPT_DIR / "textbook_images_v2"  # 高解像度版を使用
SAMPLE_DIR = SCRIPT_DIR / "data" / "施設画像サンプル（100施設）_20251120" / "施設画像サンプル"
TRAINING_DIR = SCRIPT_DIR / "training_data_v3"
MODEL_PATH = SCRIPT_DIR / "model_v3.pth"
REPORT_PATH = SCRIPT_DIR / "training_report_v3.json"
NUM_CLASSES = 100

# v3改善: サイズ拡大
CROP_SIZES = [128, 192, 256]  # より大きなクロップ
INPUT_SIZE = 224  # ResNet標準サイズ

# ===== 改良版データ生成 =====
def generate_training_data_v3(num_crops_per_image=300, train_ratio=0.8):
    """v3: より大きなクロップサイズで学習データ生成"""
    print("=" * 60)
    print("学習データ生成 v3 (拡大サイズ版)")
    print(f"クロップサイズ: {CROP_SIZES}")
    print(f"入力サイズ: {INPUT_SIZE}x{INPUT_SIZE}")
    print("=" * 60)

    train_dir = TRAINING_DIR / "train"
    test_dir = TRAINING_DIR / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_labels = []
    test_labels = []

    for i in range(1, NUM_CLASSES + 1):
        sources = []

        # 教科書画像 (512x512)
        textbook_path = TEXTBOOK_DIR / f"textbook_{i:03d}.png"
        if textbook_path.exists():
            sources.append(("textbook", Image.open(textbook_path).convert('RGB')))

        # サンプル画像 (256x256)
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
                    # 画像が小さい場合は可能な最大サイズでクロップ
                    actual_crop = min(w, h)
                    x = random.randint(0, max(0, w - actual_crop))
                    y = random.randint(0, max(0, h - actual_crop))
                    crop = img.crop((x, y, x + actual_crop, y + actual_crop))
                else:
                    x = random.randint(0, w - crop_size)
                    y = random.randint(0, h - crop_size)
                    crop = img.crop((x, y, x + crop_size, y + crop_size))

                # 入力サイズにリサイズ
                crop = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

                # 学習/テスト分割 (相対パスで保存)
                if random.random() < train_ratio:
                    crop_path = train_dir / f"{i:03d}" / f"{source_name}_{i:03d}_{crop_count:05d}.png"
                    crop.save(crop_path)
                    rel_path = f"train/{i:03d}/{source_name}_{i:03d}_{crop_count:05d}.png"
                    train_labels.append({"file": rel_path, "label": i})
                else:
                    crop_path = test_dir / f"{i:03d}" / f"{source_name}_{i:03d}_{crop_count:05d}.png"
                    crop.save(crop_path)
                    rel_path = f"test/{i:03d}/{source_name}_{i:03d}_{crop_count:05d}.png"
                    test_labels.append({"file": rel_path, "label": i})

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


# ===== 部分再生成 (特定施設のみ) =====
def regenerate_facilities(facility_ids, num_crops_per_image=300, train_ratio=0.8):
    """特定施設の訓練データのみ再生成"""
    print("=" * 60)
    print(f"部分再生成: 施設 {facility_ids}")
    print("=" * 60)

    train_dir = TRAINING_DIR / "train"
    test_dir = TRAINING_DIR / "test"

    # 既存のラベルファイル読み込み
    train_labels_path = TRAINING_DIR / "train_labels.json"
    test_labels_path = TRAINING_DIR / "test_labels.json"

    if train_labels_path.exists():
        with open(train_labels_path, 'r') as f:
            train_labels = json.load(f)
    else:
        train_labels = []

    if test_labels_path.exists():
        with open(test_labels_path, 'r') as f:
            test_labels = json.load(f)
    else:
        test_labels = []

    # 対象施設の既存データを削除
    train_labels = [x for x in train_labels if x['label'] not in facility_ids]
    test_labels = [x for x in test_labels if x['label'] not in facility_ids]

    for i in facility_ids:
        # 既存ファイル削除
        train_subdir = train_dir / f"{i:03d}"
        test_subdir = test_dir / f"{i:03d}"
        if train_subdir.exists():
            for f in train_subdir.glob("*"):
                f.unlink()
        if test_subdir.exists():
            for f in test_subdir.glob("*"):
                f.unlink()

        sources = []

        # 教科書画像
        textbook_path = TEXTBOOK_DIR / f"textbook_{i:03d}.png"
        if textbook_path.exists():
            sources.append(("textbook", Image.open(textbook_path).convert('RGB')))

        # サンプル画像
        sample_path = SAMPLE_DIR / f"map{i:03d}.jpg"
        if sample_path.exists():
            sources.append(("sample", Image.open(sample_path).convert('RGB')))

        if not sources:
            print(f"  Warning: No images for facility {i}")
            continue

        train_subdir.mkdir(parents=True, exist_ok=True)
        test_subdir.mkdir(parents=True, exist_ok=True)

        crop_count = 0
        for source_name, img in sources:
            w, h = img.size
            crops_per_source = num_crops_per_image // len(sources)

            for j in range(crops_per_source):
                crop_size = random.choice(CROP_SIZES)

                if w < crop_size or h < crop_size:
                    actual_crop = min(w, h)
                    x = random.randint(0, max(0, w - actual_crop))
                    y = random.randint(0, max(0, h - actual_crop))
                    crop = img.crop((x, y, x + actual_crop, y + actual_crop))
                else:
                    x = random.randint(0, w - crop_size)
                    y = random.randint(0, h - crop_size)
                    crop = img.crop((x, y, x + crop_size, y + crop_size))

                crop = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

                if random.random() < train_ratio:
                    crop_path = train_subdir / f"{source_name}_{i:03d}_{crop_count:05d}.png"
                    crop.save(crop_path)
                    rel_path = f"train/{i:03d}/{source_name}_{i:03d}_{crop_count:05d}.png"
                    train_labels.append({"file": rel_path, "label": i})
                else:
                    crop_path = test_subdir / f"{source_name}_{i:03d}_{crop_count:05d}.png"
                    crop.save(crop_path)
                    rel_path = f"test/{i:03d}/{source_name}_{i:03d}_{crop_count:05d}.png"
                    test_labels.append({"file": rel_path, "label": i})

                crop_count += 1

        print(f"  施設{i}: {crop_count}枚生成")

    # ラベルファイル保存
    with open(train_labels_path, 'w') as f:
        json.dump(train_labels, f)
    with open(test_labels_path, 'w') as f:
        json.dump(test_labels, f)

    print(f"\n再生成完了:")
    print(f"  学習データ: {len(train_labels)}枚")
    print(f"  テストデータ: {len(test_labels)}枚")

    return len(train_labels), len(test_labels)


# ===== ファインチューニング =====
def finetune_v3(base_model_path, facility_ids, epochs=10, batch_size=16, output_suffix="ft"):
    """既存モデルを特定施設でファインチューニング"""
    print("=" * 60)
    print(f"ファインチューニング")
    print(f"ベースモデル: {base_model_path}")
    print(f"対象施設: {facility_ids}")
    print(f"エポック: {epochs}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    # ベースモデルロード
    checkpoint = torch.load(base_model_path, map_location=device)
    model_type = checkpoint.get('model_type', 'custom_cnn')
    base_acc = checkpoint.get('best_acc', 0)

    if model_type == 'resnet18':
        model = ResNetFeatureExtractor(NUM_CLASSES).to(device)
    else:
        model = GeoClassifierV3(NUM_CLASSES).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"モデルロード完了 (精度: {base_acc:.2f}%, タイプ: {model_type})")

    # データ変換
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # データセット
    train_dataset = CropDatasetV3(TRAINING_DIR / "train_labels.json", train_transform)
    test_dataset = CropDatasetV3(TRAINING_DIR / "test_labels.json", test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"学習データ: {len(train_dataset)}枚")
    print(f"テストデータ: {len(test_dataset)}枚")

    # 低い学習率でファインチューニング
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = base_acc
    jst = timezone(timedelta(hours=9))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
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

        scheduler.step()
        train_acc = 100.0 * train_correct / train_total

        # テスト
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

        now_jst = datetime.now(jst).strftime('%H:%M:%S')
        print(f"  [{now_jst}] Epoch [{epoch+1}/{epochs}] "
              f"Loss: {train_loss/len(train_loader):.4f} "
              f"Train: {train_acc:.2f}% "
              f"Test: {test_acc:.2f}%", flush=True)

        # ベストモデル保存
        if test_acc > best_acc:
            best_acc = test_acc
            # 新しいモデルファイル名
            base_name = Path(base_model_path).stem
            new_model_path = SCRIPT_DIR / f"{base_name}_{output_suffix}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'model_type': model_type,
                'finetuned_facilities': facility_ids,
            }, new_model_path)
            print(f"  -> ベスト更新! {new_model_path} (精度: {best_acc:.2f}%)")

    print(f"\nファインチューニング完了")
    print(f"  元精度: {base_acc:.2f}%")
    print(f"  最終精度: {best_acc:.2f}%")

    return best_acc


# ===== データセット =====
class CropDatasetV3(Dataset):
    def __init__(self, label_file, transform=None, hard_samples=None):
        with open(label_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.base_dir = Path(label_file).parent  # training_data_v3フォルダ
        if hard_samples:
            self.data.extend(hard_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 相対パスをベースディレクトリからの絶対パスに変換
        file_path = self.base_dir / item['file']
        img = Image.open(file_path).convert('RGB')
        label = item['label'] - 1

        if self.transform:
            img = self.transform(img)

        return img, label


# ===== カスタムCNN v3 (より深く、224対応) =====
class GeoClassifierV3(nn.Module):
    """カスタムCNN - 224x224入力対応"""
    def __init__(self, num_classes=NUM_CLASSES):
        super(GeoClassifierV3, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5: 14 -> 7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===== ResNet特徴抽出モデル (オプション) =====
class ResNetFeatureExtractor(nn.Module):
    """ResNet18の特徴抽出能力を活用"""
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=False):
        super(ResNetFeatureExtractor, self).__init__()

        # ResNet18をベースに（軽量で高速）
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # バックボーンを凍結するかどうか
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 最終層を置き換え
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ===== エラー分析 =====
def analyze_errors(model, test_loader, device):
    """誤分類を分析して弱点を特定"""
    model.eval()
    errors = defaultdict(list)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for pred, true in zip(predicted, labels):
                true_label = true.item() + 1
                pred_label = pred.item() + 1
                class_total[true_label] += 1

                if pred_label == true_label:
                    class_correct[true_label] += 1
                else:
                    errors[true_label].append(pred_label)

    # クラスごとの精度計算
    class_accuracy = {}
    for label in range(1, NUM_CLASSES + 1):
        if class_total[label] > 0:
            class_accuracy[label] = class_correct[label] / class_total[label] * 100
        else:
            class_accuracy[label] = 0

    # 精度が低いクラスを特定
    weak_classes = sorted(class_accuracy.items(), key=lambda x: x[1])

    return weak_classes, dict(errors)


# ===== 反復学習 v3 =====
def iterative_train_v3(max_iterations=10, epochs_per_iter=30, target_accuracy=95.0,
                       use_resnet=False, batch_size=32):
    """v3: 改善版反復学習"""
    print("=" * 60)
    print("反復学習 v3 開始")
    print(f"目標精度: {target_accuracy}%")
    print(f"モデル: {'ResNet18' if use_resnet else 'Custom CNN'}")
    print(f"入力サイズ: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"バッチサイズ: {batch_size}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    # データ変換
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    history = {
        'iterations': [],
        'best_accuracy': 0,
        'model_type': 'resnet18' if use_resnet else 'custom_cnn',
        'input_size': INPUT_SIZE
    }

    best_overall_acc = 0
    hard_samples = []

    for iteration in range(max_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"反復 {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")

        # データセット
        train_dataset = CropDatasetV3(
            TRAINING_DIR / "train_labels.json",
            train_transform,
            hard_samples if iteration > 0 else None
        )
        test_dataset = CropDatasetV3(
            TRAINING_DIR / "test_labels.json",
            test_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f"学習データ: {len(train_dataset)}枚")
        print(f"テストデータ: {len(test_dataset)}枚")
        print("モデル初期化中...", flush=True)

        # モデル選択
        if use_resnet:
            model = ResNetFeatureExtractor(NUM_CLASSES, freeze_backbone=(iteration == 0)).to(device)
        else:
            model = GeoClassifierV3(NUM_CLASSES).to(device)

        # 前回のモデルを継続
        if iteration > 0 and MODEL_PATH.exists():
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("前回のモデルを継続学習")
            except:
                print("新規モデルで開始")

        criterion = nn.CrossEntropyLoss()

        # ResNetの場合は最終層のみ高い学習率
        if use_resnet:
            optimizer = optim.AdamW([
                {'params': model.backbone.fc.parameters(), 'lr': 0.001},
                {'params': [p for n, p in model.backbone.named_parameters() if 'fc' not in n], 'lr': 0.0001}
            ], weight_decay=0.01)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_iter)

        best_acc = 0
        total_batches = len(train_loader)
        print(f"学習開始 (バッチ数: {total_batches})", flush=True)

        for epoch in range(epochs_per_iter):
            # 学習
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx == 0:
                    print(f"  Epoch {epoch+1}: 最初のバッチ処理中...", flush=True)
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

                # 進捗表示 (100バッチごと)
                if (batch_idx + 1) % 100 == 0:
                    print(f"    バッチ {batch_idx+1}/{total_batches}", flush=True)

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

            # 日本時間 (JST = UTC+9)
            jst = timezone(timedelta(hours=9))
            now_jst = datetime.now(jst).strftime('%H:%M:%S')
            print(f"  [{now_jst}] Epoch [{epoch+1}/{epochs_per_iter}] "
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
                        'model_type': 'resnet18' if use_resnet else 'custom_cnn'
                    }, MODEL_PATH)
                    print(f"    -> ベストモデル保存 (acc: {best_acc:.2f}%)")

            scheduler.step()

        # エラー分析
        print(f"\n[エラー分析]")
        weak_classes, error_details = analyze_errors(model, test_loader, device)

        print(f"  精度が低いクラス (Top 10):")
        for label, acc in weak_classes[:10]:
            print(f"    施設{label}: {acc:.1f}%")

        # 苦手クラスの重点学習データ作成
        very_weak = [label for label, acc in weak_classes if acc < 50]
        if very_weak:
            hard_samples = []
            with open(TRAINING_DIR / "train_labels.json", 'r') as f:
                all_train = json.load(f)
            for item in all_train:
                if item['label'] in very_weak:
                    # 苦手クラスを3倍に
                    hard_samples.extend([item] * 3)
            print(f"  次回の重点学習サンプル: {len(hard_samples)}枚 ({len(very_weak)}クラス)")

        iter_time = time.time() - iter_start
        history['iterations'].append({
            'iteration': iteration + 1,
            'best_accuracy': best_acc,
            'weak_classes': weak_classes[:10],
            'time_seconds': iter_time
        })

        # レポート保存（反復ごと）
        history['best_accuracy'] = best_overall_acc
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"  反復時間: {iter_time/60:.1f}分")

        # 目標達成チェック
        if best_acc >= target_accuracy:
            print(f"\n目標精度 {target_accuracy}% に到達!")
            break

    print(f"\n{'='*60}")
    print(f"学習完了!")
    print(f"最終精度: {best_overall_acc:.2f}%")
    print(f"レポート: {REPORT_PATH}")
    print(f"{'='*60}")

    return best_overall_acc


# ===== AI推論 =====
def predict_v3(question_dir, output_file="answer.txt", use_resnet=False, model_path=None):
    """v3推論"""
    print("=" * 60)
    print("AI推論 v3")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルロード
    load_path = Path(model_path) if model_path else MODEL_PATH
    print(f"モデルファイル: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)
    model_type = checkpoint.get('model_type', 'custom_cnn')

    if model_type == 'resnet18':
        model = ResNetFeatureExtractor(NUM_CLASSES).to(device)
    else:
        model = GeoClassifierV3(NUM_CLASSES).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"モデルロード完了 (精度: {checkpoint['best_acc']:.2f}%, タイプ: {model_type})")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TTA用
    tta_transforms = [
        transform,
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

    question_dir = Path(question_dir)
    # 数字順でソート (q1, q2, ... q10 の順になるように)
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    question_files = sorted([f for f in os.listdir(question_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))], key=natural_sort_key)
    print(f"問題画像: {len(question_files)}枚")

    results = []

    for qf in question_files:
        img_path = question_dir / qf
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        votes = defaultdict(float)
        crop_size = min(w, h, INPUT_SIZE)

        # クロップ位置
        positions = [
            (w//2 - crop_size//2, h//2 - crop_size//2),
            (0, 0), (w - crop_size, 0),
            (0, h - crop_size), (w - crop_size, h - crop_size),
        ]

        for _ in range(15):
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

            for tta_tf in tta_transforms:
                input_tensor = tta_tf(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    top_probs, top_indices = probs.topk(5)

                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        label = idx.item() + 1
                        votes[label] += prob.item()

        best_label = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[best_label] / total_votes * 100

        print(f"  {qf} -> 施設番号 {best_label} (信頼度: {confidence:.1f}%)")
        results.append(best_label)

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
        print("  データ生成:     python ai_trainer_v3.py generate [crops_per_image]")
        print("  部分再生成:     python ai_trainer_v3.py regenerate <施設番号...>")
        print("  学習(カスタム): python ai_trainer_v3.py train [iterations] [epochs] [target_acc] [batch_size]")
        print("  学習(ResNet):   python ai_trainer_v3.py train_resnet [iterations] [epochs] [target_acc] [batch_size]")
        print("  ファインチューニング: python ai_trainer_v3.py finetune <モデル> <epochs> <施設番号...>")
        print("  推論:           python ai_trainer_v3.py predict <問題画像フォルダ> [モデルファイル]")
        print("  全実行:         python ai_trainer_v3.py all")
        print("")
        print("  例: python ai_trainer_v3.py train 10 30 95 16  (バッチサイズ16)")
        print("  例: python ai_trainer_v3.py regenerate 68 69")
        print("  例: python ai_trainer_v3.py finetune model_v3m89.pth 10 68 69")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "generate":
        num_crops = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        generate_training_data_v3(num_crops_per_image=num_crops)

    elif mode == "regenerate":
        if len(sys.argv) < 3:
            print("Error: 施設番号を指定してください")
            sys.exit(1)
        facility_ids = [int(x) for x in sys.argv[2:]]
        regenerate_facilities(facility_ids)

    elif mode == "train":
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        target = float(sys.argv[4]) if len(sys.argv) > 4 else 95.0
        batch = int(sys.argv[5]) if len(sys.argv) > 5 else 16  # デフォルト16に変更
        iterative_train_v3(max_iterations=iterations, epochs_per_iter=epochs,
                          target_accuracy=target, use_resnet=False, batch_size=batch)

    elif mode == "train_resnet":
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        target = float(sys.argv[4]) if len(sys.argv) > 4 else 95.0
        batch = int(sys.argv[5]) if len(sys.argv) > 5 else 16  # デフォルト16に変更
        iterative_train_v3(max_iterations=iterations, epochs_per_iter=epochs,
                          target_accuracy=target, use_resnet=True, batch_size=batch)

    elif mode == "finetune":
        # 形式: finetune <モデル> <エポック数> <施設番号...>
        if len(sys.argv) < 5:
            print("Error: モデルファイル、エポック数、施設番号を指定してください")
            print("  例: python ai_trainer_v3.py finetune model_v3m89.pth 10 68 69")
            sys.exit(1)
        model_file = sys.argv[2]
        epochs = int(sys.argv[3])
        facility_ids = [int(x) for x in sys.argv[4:]]
        finetune_v3(model_file, facility_ids, epochs=epochs)

    elif mode == "predict":
        if len(sys.argv) < 3:
            print("Error: 問題画像フォルダを指定してください")
            sys.exit(1)
        model_file = sys.argv[3] if len(sys.argv) > 3 else None
        predict_v3(sys.argv[2], model_path=model_file)

    elif mode == "all":
        generate_training_data_v3(num_crops_per_image=300)
        iterative_train_v3(max_iterations=10, epochs_per_iter=30, target_accuracy=95.0, batch_size=16)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
