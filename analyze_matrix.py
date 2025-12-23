# -*- coding: utf-8 -*-
"""
混同行列分析スクリプト - 施設65と70の誤分類分析
GPU搭載PCで実行してください
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ai_trainer_v3.pyからモデルクラスをインポート
from ai_trainer_v3 import GeoClassifierV3, NUM_CLASSES, INPUT_SIZE

# ===== 設定 =====
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / "model_v3m109.pth"
TEST_DIR = SCRIPT_DIR / "training_data_v3" / "test"

# 分析対象クラス
TARGET_CLASSES = [65, 70]


def analyze_confusion(model_path=MODEL_PATH):
    print("=" * 60)
    print("混同行列分析 - 施設65 vs 70")
    print(f"モデル: {model_path}")
    print("=" * 60)

    # CUDA使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    # モデルロード
    checkpoint = torch.load(model_path, map_location=device)
    model = GeoClassifierV3(NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"モデルロード完了 (精度: {checkpoint['best_acc']:.2f}%)")

    # データ変換 (テスト用)
    test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ImageFolderでテストデータ読み込み
    test_dataset = ImageFolder(str(TEST_DIR), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"テストデータ: {len(test_dataset)}枚")
    print(f"クラス数: {len(test_dataset.classes)}")

    # クラス名からインデックスへのマッピング確認
    class_to_idx = test_dataset.class_to_idx
    print(f"クラスマッピング例: 065 -> {class_to_idx.get('065', 'N/A')}, 070 -> {class_to_idx.get('070', 'N/A')}")

    # 全クラスの混同行列を計算
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    print("\n分析中...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for true, pred in zip(labels, predicted):
                confusion[true.item()][pred.item()] += 1

    # ターゲットクラスのインデックス (ImageFolderはフォルダ名順でソートするので 065->64, 070->69)
    idx_65 = class_to_idx.get('065', 64)
    idx_70 = class_to_idx.get('070', 69)

    # 2x2のクロス集計
    print("\n" + "=" * 60)
    print(f"Target Classes: 65 vs 70")
    print("-" * 60)

    actual_65_pred_65 = confusion[idx_65][idx_65]
    actual_65_pred_70 = confusion[idx_65][idx_70]
    actual_70_pred_70 = confusion[idx_70][idx_70]
    actual_70_pred_65 = confusion[idx_70][idx_65]

    print(f"Actual 65 -> Predicted 65:  {actual_65_pred_65} (正解)")
    print(f"Actual 65 -> Predicted 70:  {actual_65_pred_70} (誤答)")
    print("-" * 60)
    print(f"Actual 70 -> Predicted 70:  {actual_70_pred_70} (正解)")
    print(f"Actual 70 -> Predicted 65:  {actual_70_pred_65} (誤答)")
    print("-" * 60)

    # 精度計算
    total_65 = sum(confusion[idx_65])
    total_70 = sum(confusion[idx_70])

    if total_65 > 0:
        acc_65 = 100.0 * actual_65_pred_65 / total_65
        print(f"Class 65 Accuracy: {acc_65:.2f}% ({actual_65_pred_65}/{total_65})")

    if total_70 > 0:
        acc_70 = 100.0 * actual_70_pred_70 / total_70
        print(f"Class 70 Accuracy: {acc_70:.2f}% ({actual_70_pred_70}/{total_70})")

    combined_correct = actual_65_pred_65 + actual_70_pred_70
    combined_total = total_65 + total_70
    combined_acc = 100.0 * combined_correct / combined_total if combined_total > 0 else 0
    print(f"Accuracy for these classes: {combined_acc:.2f}%")
    print("=" * 60)

    # その他の誤分類も表示
    print("\n" + "=" * 60)
    print("その他の誤分類 (上位5件)")
    print("-" * 60)

    for target_idx, target_class in [(idx_65, 65), (idx_70, 70)]:
        errors = []
        for pred_idx in range(NUM_CLASSES):
            if pred_idx != target_idx and confusion[target_idx][pred_idx] > 0:
                errors.append((pred_idx + 1, confusion[target_idx][pred_idx]))
        errors.sort(key=lambda x: -x[1])

        print(f"\nActual {target_class} の誤分類先:")
        if errors:
            for pred_class, count in errors[:5]:
                print(f"  -> Predicted {pred_class}: {count}件")
        else:
            print("  (誤分類なし)")

    # 全体のヒートマップ保存
    print("\n" + "=" * 60)
    print("全体の混同行列ヒートマップを作成中...")

    fig, ax = plt.subplots(figsize=(20, 16))

    # 対数スケールで表示（0は避ける）
    confusion_log = np.log1p(confusion)

    sns.heatmap(confusion_log, cmap='Blues', ax=ax,
                xticklabels=range(1, NUM_CLASSES + 1),
                yticklabels=range(1, NUM_CLASSES + 1))

    ax.set_title(f'Confusion Matrix (log scale)\n'
                 f'Model: {Path(model_path).name} (Acc: {checkpoint["best_acc"]:.2f}%)')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')

    # 65と70をハイライト
    ax.axhline(y=idx_65, color='red', linewidth=2)
    ax.axhline(y=idx_65 + 1, color='red', linewidth=2)
    ax.axhline(y=idx_70, color='red', linewidth=2)
    ax.axhline(y=idx_70 + 1, color='red', linewidth=2)
    ax.axvline(x=idx_65, color='red', linewidth=2)
    ax.axvline(x=idx_65 + 1, color='red', linewidth=2)
    ax.axvline(x=idx_70, color='red', linewidth=2)
    ax.axvline(x=idx_70 + 1, color='red', linewidth=2)

    plt.tight_layout()
    output_path = SCRIPT_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    print(f"ヒートマップ保存: {output_path}")
    plt.close()

    # 65と70だけの2x2ヒートマップも保存
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    matrix_2x2 = np.array([
        [actual_65_pred_65, actual_65_pred_70],
        [actual_70_pred_65, actual_70_pred_70]
    ])

    sns.heatmap(matrix_2x2, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 65', 'Pred 70'],
                yticklabels=['Actual 65', 'Actual 70'],
                ax=ax2)

    ax2.set_title(f'Confusion Matrix: Class 65 vs 70\n'
                  f'Model: {Path(model_path).name}')

    plt.tight_layout()
    output_path_2x2 = SCRIPT_DIR / "confusion_matrix_65_70.png"
    plt.savefig(output_path_2x2, dpi=150)
    print(f"2x2ヒートマップ保存: {output_path_2x2}")
    plt.close()

    print("\n分析完了!")
    return confusion


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        model_path = MODEL_PATH

    analyze_confusion(model_path)
