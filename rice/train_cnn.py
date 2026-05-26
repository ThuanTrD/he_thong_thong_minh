"""
=============================================================================
PIPELINE HYBRID CNN + FUZZY (XAI) — BƯỚC 2: CNN INFERENCE MODULE
=============================================================================
Dataset  : Severity-Based Rice Leaf Diseases Dataset (Kaggle)
Backbone : EfficientNet-B0 (pretrained ImageNet, fine-tuned)
Output   :
  - checkpoints/best_model.pt          ← trọng số tốt nhất
  - checkpoints/last_model.pt          ← trọng số epoch cuối
  - outputs/val_cnn_scores.csv         ← softmax scores cho Fuzzy layer
  - outputs/class_metadata.json        ← tên class + index mapping
  - outputs/training_report.txt        ← báo cáo kết quả train
=============================================================================
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Train CNN for Rice Disease Classification")
    parser.add_argument("--data_dir",    type=str,   default="./dataset",
                        help="Root folder of dataset (contains train/val or class folders)")
    parser.add_argument("--output_dir",  type=str,   default="./outputs",
                        help="Folder to save CSV, JSON, report")
    parser.add_argument("--ckpt_dir",    type=str,   default="./checkpoints",
                        help="Folder to save .pt weight files")
    parser.add_argument("--backbone",    type=str,   default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "mobilenet_v3_small"],
                        help="CNN backbone architecture")
    parser.add_argument("--img_size",    type=int,   default=224)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--val_split",   type=float, default=0.2,
                        help="Validation ratio if no separate val folder")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Epochs to freeze backbone (only train head)")
    parser.add_argument("--use_amp",     action="store_true", default=True,
                        help="Use Automatic Mixed Precision (faster on GPU)")
    return parser.parse_args()


# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
#  DATA TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(img_size: int):
    """
    Train: augmentation mạnh để chống overfit trên dataset nhỏ
    Val  : chỉ resize + normalize (không augment)
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2),   # giả lập che khuất lá
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ─────────────────────────────────────────────
#  DATASET LOADER
# ─────────────────────────────────────────────
def load_datasets(data_dir: str, val_split: float, train_tf, val_tf, seed: int):
    """
    Hỗ trợ 2 cấu trúc:
      A) data_dir/train/  và  data_dir/val/   (đã chia sẵn)
      B) data_dir/class_name/...              (tự split)
    """
    train_path = Path(data_dir) / "train"
    val_path   = Path(data_dir) / "val"

    if train_path.exists() and val_path.exists():
        print("[DATA] Phát hiện thư mục train/val riêng biệt.")
        train_ds = datasets.ImageFolder(str(train_path), transform=train_tf)
        val_ds   = datasets.ImageFolder(str(val_path),   transform=val_tf)
    else:
        print(f"[DATA] Không có split sẵn. Auto-split {int((1-val_split)*100)}/{int(val_split*100)} từ: {data_dir}")
        full_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
        n_val   = int(len(full_ds) * val_split)
        n_train = len(full_ds) - n_val
        train_ds_raw, val_ds_raw = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
        # Áp transform val riêng
        val_ds_raw.dataset = datasets.ImageFolder(str(data_dir), transform=val_tf)
        train_ds = train_ds_raw
        val_ds   = val_ds_raw

    classes = (train_ds.dataset.classes
               if hasattr(train_ds, 'dataset')
               else train_ds.classes)

    print(f"[DATA] Classes ({len(classes)}): {classes}")
    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds, classes


# ─────────────────────────────────────────────
#  MODEL BUILDER
# ─────────────────────────────────────────────
def build_model(backbone: str, num_classes: int) -> nn.Module:
    """
    Xây dựng model với pretrained ImageNet weights.
    Thay classifier head thành num_classes.
    """
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone}")

    print(f"[MODEL] Backbone: {backbone} | Num classes: {num_classes}")
    return model


def freeze_backbone(model: nn.Module, backbone: str):
    """Đóng băng tất cả layer trừ classifier head."""
    for name, param in model.named_parameters():
        if backbone == "efficientnet_b0" and "classifier" in name:
            param.requires_grad = True
        elif backbone == "resnet50" and "fc" in name:
            param.requires_grad = True
        elif backbone == "mobilenet_v3_small" and "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("[MODEL] Backbone đã bị freeze (chỉ train head)")


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
    print("[MODEL] Mở freeze toàn bộ backbone")


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    avg_loss = total_loss / total
    return avg_loss, acc, all_preds, all_labels


# ─────────────────────────────────────────────
#  CNN SCORE EXPORT (Input cho Fuzzy Layer)
# ─────────────────────────────────────────────
@torch.no_grad()
def export_cnn_scores(model, loader, classes, device, output_path, use_amp=True):
    """
    Export softmax probability scores của từng ảnh trên val set.
    Đây là INPUT chính cho Fuzzy Reasoning Layer (Bước 3).

    Output CSV schema:
      image_path | true_label | pred_label | <class_0_score> | ... | <class_N_score>
    """
    model.eval()
    softmax = nn.Softmax(dim=1)

    rows = []
    img_paths = []

    # Lấy paths ảnh từ dataset
    if hasattr(loader.dataset, 'dataset'):
        # Subset từ random_split
        indices = loader.dataset.indices
        samples = [loader.dataset.dataset.samples[i] for i in indices]
    else:
        samples = loader.dataset.samples

    img_paths = [Path(s[0]).name for s in samples]
    true_labels_all = [s[1] for s in samples]

    idx = 0
    for images, labels in loader:
        images = images.to(device)
        with autocast(enabled=use_amp):
            logits = model(images)
        probs = softmax(logits).cpu().numpy()
        preds = probs.argmax(axis=1)

        for b in range(len(labels)):
            row = {
                "image_path"  : img_paths[idx] if idx < len(img_paths) else f"sample_{idx}",
                "true_label"  : classes[labels[b].item()],
                "pred_label"  : classes[preds[b]],
                "correct"     : int(labels[b].item() == preds[b]),
            }
            # Thêm score từng class (đây là input cho Fuzzy)
            for c_idx, c_name in enumerate(classes):
                row[f"score_{c_name}"] = round(float(probs[b, c_idx]), 6)
            rows.append(row)
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[EXPORT] CNN scores đã lưu: {output_path} ({len(df)} records)")
    return df


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    args = get_args()
    set_seed(args.seed)

    # Tạo thư mục output
    os.makedirs(args.output_dir,  exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Sử dụng: {device}")
    if device.type == "cuda":
        print(f"         GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────
    train_tf, val_tf = get_transforms(args.img_size)
    train_ds, val_ds, classes = load_datasets(
        args.data_dir, args.val_split, train_tf, val_tf, args.seed
    )
    num_classes = len(classes)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # ── Model ────────────────────────────────
    model = build_model(args.backbone, num_classes).to(device)

    # ── Loss + Optimizer + Scheduler ─────────
    # Label smoothing giúp model calibrated hơn (quan trọng cho Fuzzy)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=args.use_amp)

    # ── Training ─────────────────────────────
    best_val_acc = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"  BẮT ĐẦU TRAINING: {args.backbone.upper()}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  Freeze backbone {args.freeze_epochs} epochs đầu")
    print(f"{'='*60}\n")

    # Phase 1: Freeze backbone — chỉ train head
    freeze_backbone(model, args.backbone)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 5, weight_decay=args.weight_decay
    )

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone sau freeze_epochs
        if epoch == args.freeze_epochs + 1:
            unfreeze_all(model)
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-6
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.use_amp
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, args.use_amp
        )
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:>3}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.0f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })

        # Lưu best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save({
                "epoch"       : epoch,
                "backbone"    : args.backbone,
                "num_classes" : num_classes,
                "classes"     : classes,
                "state_dict"  : model.state_dict(),
                "val_acc"     : val_acc,
                "val_loss"    : val_loss,
                "img_size"    : args.img_size,
            }, best_ckpt)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    # Lưu last model
    last_ckpt = os.path.join(args.ckpt_dir, "last_model.pt")
    torch.save({
        "epoch"       : args.epochs,
        "backbone"    : args.backbone,
        "num_classes" : num_classes,
        "classes"     : classes,
        "state_dict"  : model.state_dict(),
        "val_acc"     : val_acc,
        "img_size"    : args.img_size,
    }, last_ckpt)

    # ── Load best model để export ─────────────
    print(f"\n[EVAL] Load best model để đánh giá và export...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # Classification report
    _, final_acc, final_preds, final_labels = validate(
        model, val_loader, criterion, device, args.use_amp
    )
    report = classification_report(final_labels, final_preds,
                                   target_names=classes, digits=4)
    cm = confusion_matrix(final_labels, final_preds)

    print(f"\n[RESULT] Best Val Accuracy: {best_val_acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # ── Export CNN Scores cho Fuzzy Layer ─────
    scores_path = os.path.join(args.output_dir, "val_cnn_scores.csv")
    export_cnn_scores(model, val_loader, classes, device, scores_path, args.use_amp)

    # ── Export class metadata ─────────────────
    metadata = {
        "backbone"      : args.backbone,
        "num_classes"   : num_classes,
        "classes"       : classes,
        "class_to_idx"  : {c: i for i, c in enumerate(classes)},
        "img_size"      : args.img_size,
        "best_val_acc"  : round(best_val_acc, 6),
        "train_date"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fuzzy_input_columns": [f"score_{c}" for c in classes],
        "note": (
            "Dùng cột 'score_*' làm input cho Fuzzy Reasoning Layer. "
            "Đây là softmax probability, range [0, 1], tổng = 1."
        )
    }
    meta_path = os.path.join(args.output_dir, "class_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[EXPORT] Metadata lưu: {meta_path}")

    # ── Training history CSV ──────────────────
    hist_path = os.path.join(args.output_dir, "training_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)

    # ── Training report text ──────────────────
    report_lines = [
        "=" * 60,
        "  TRAINING REPORT — CNN RICE DISEASE CLASSIFIER",
        "=" * 60,
        f"Backbone      : {args.backbone}",
        f"Dataset dir   : {args.data_dir}",
        f"Num classes   : {num_classes}",
        f"Classes       : {classes}",
        f"Image size    : {args.img_size}x{args.img_size}",
        f"Epochs        : {args.epochs}",
        f"Batch size    : {args.batch_size}",
        f"Learning rate : {args.lr}",
        f"Best val acc  : {best_val_acc:.4f}",
        f"Train date    : {metadata['train_date']}",
        "",
        "Confusion Matrix:",
        str(cm),
        "",
        "Classification Report:",
        report,
        "=" * 60,
        "OUTPUT FILES:",
        f"  Trọng số tốt nhất : {best_ckpt}",
        f"  Trọng số epoch cuối: {last_ckpt}",
        f"  CNN scores (Fuzzy input): {scores_path}",
        f"  Class metadata    : {meta_path}",
        f"  Training history  : {hist_path}",
        "=" * 60,
    ]
    report_path = os.path.join(args.output_dir, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[EXPORT] Report lưu: {report_path}")

    print(f"\n{'='*60}")
    print(f"  TRAINING HOÀN TẤT!")
    print(f"  Best Val Accuracy : {best_val_acc:.4f}")
    print(f"  Trọng số          : {best_ckpt}")
    print(f"  CNN Scores (→Fuzzy): {scores_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
