import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, f1_score, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import csv
import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

results_dir = "DenseNet121_CLAHE_Results_SubjectLevel"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}/")

class CLAHETransform:

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img.convert("RGB"))          # PIL -> numpy uint8, HxWx3
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),  # <-- CLAHE applied here, before ToTensor
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# Paths and parameters
# ─────────────────────────────────────────────
# Root produced by split_and_augment_5fold.py:
#   DATA_ROOT/fold_1/train/{TRE,NonTRE}
#   DATA_ROOT/fold_1/val/{TRE,NonTRE}
#   DATA_ROOT/fold_1/test/{TRE,NonTRE}
#   DATA_ROOT/fold_2/...  ... fold_5/...
DATA_ROOT  = r"C:\FYP_MRIDataset\Final_Data_Used\Subject_Level_5Fold"
batch_size = 8
num_epochs = 10
k_folds    = 5

# ─────────────────────────────────────────────
# Helper: extract patient ID from filename
# e.g. "G002_slice1.png" -> "G002"
#      "G002_slice1_rotL.png" -> "G002"  (augmented files, only ever in train)
# ─────────────────────────────────────────────
def get_patient_id(filepath):
    fname = os.path.basename(filepath)
    return fname.split("_")[0]

# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────
def create_model(num_classes):
    # DenseNet121 (2017): each layer receives feature maps from all
    # preceding layers (dense connectivity), which encourages feature
    # reuse and tends to work well on smaller datasets — a solid
    # comparison point against the newer architectures.
    model = models.densenet121(pretrained=True)
    # DenseNet's head is a single Linear layer (model.classifier),
    # NOT a Sequential block like EfficientNet/ConvNeXt:
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────
# Subject-level metrics helper
# Positive class = "TRE". Computes:
#   Test Accuracy, F1-Score, Sensitivity (Recall for TRE), Specificity
#   (Recall for NonTRE), NPV, AUC (using mean predicted TRE-probability
#   per patient as the score).
# ─────────────────────────────────────────────────────
def compute_subject_metrics(y_true, y_pred, y_score, positive_idx, negative_idx):
    """
    y_true, y_pred : lists/arrays of class indices (one entry per patient)
    y_score        : list/array of predicted probability of the POSITIVE
                      class (TRE) per patient, used only for AUC
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label=positive_idx, average="binary")

    cm = confusion_matrix(y_true, y_pred, labels=[negative_idx, positive_idx])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # Recall (TRE)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # Recall (NonTRE)
    npv         = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    y_true_bin = (y_true == positive_idx).astype(int)
    try:
        auc = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        # AUC undefined if only one class is present in y_true for this fold
        auc = float("nan")

    return {
        "Test Accuracy": acc,
        "F1-Score":      f1,
        "Sensitivity":   sensitivity,
        "Specificity":   specificity,
        "NPV":           npv,
        "AUC":           auc,
    }


# ─────────────────────────────────────────────
# Storage for aggregated results (subject-level only)
# ─────────────────────────────────────────────
fold_val_results          = []   # best val acc per fold
fold_subject_test_results = []   # subject-level test acc per fold
fold_metrics_list         = []   # list of per-fold metrics dicts (Acc/F1/Sens/Spec/NPV/AUC)

all_subject_y_true  = []   # accumulated SUBJECT-level test labels across all folds
all_subject_y_pred  = []   # accumulated SUBJECT-level test predictions across all folds
all_subject_y_score = []   # accumulated SUBJECT-level predicted TRE-probability across all folds

class_names = None    # set from the first fold's ImageFolder, must match across folds
positive_idx = None   # index of "TRE" within class_names
negative_idx = None   # index of "NonTRE" within class_names


# ─────────────────────────────────────────────
# 5-Fold Cross-Validation  (fixed, subject-level, non-overlapping test sets)
# ─────────────────────────────────────────────
for fold in range(k_folds):
    print(f"\n{'='*40}")
    print(f"  FOLD {fold+1}/{k_folds}")
    print(f"{'='*40}\n")

    fold_dir  = os.path.join(DATA_ROOT, f"fold_{fold+1}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir   = os.path.join(fold_dir, "val")
    test_dir  = os.path.join(fold_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=transform)

    if class_names is None:
        class_names = train_dataset.classes  # e.g. ['NonTRE', 'TRE']
        positive_idx = class_names.index("TRE")
        negative_idx = class_names.index("NonTRE")
        print(f"Classes: {class_names}  (positive='TRE' -> idx {positive_idx}, "
              f"negative='NonTRE' -> idx {negative_idx})")
    else:
        assert train_dataset.classes == class_names, \
            "Class order differs between folds — check folder names match across fold_X dirs."

    print(f"  Train: {len(train_dataset)} (incl. augmented) | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    # shuffle=False on test is required so we can zip predictions back to
    # test_dataset.samples in order for subject-level aggregation.
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ── Model & optimiser ─────────
    model     = create_model(len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_acc = 0.0

    train_losses, val_losses         = [], []
    train_accuracies, val_accuracies = [], []

    # ── Training loop ─────────────
    for epoch in range(num_epochs):
        # -- Train --
        model.train()
        running_loss = correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds  = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # -- Validate --
        model.eval()
        v_loss = v_correct = v_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                v_loss    += loss.item() * images.size(0)
                _, preds   = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"  Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(results_dir, f"DenseNet121_CLAHE_Best_Fold{fold+1}.pth"))

    fold_val_results.append(best_val_acc)
    print(f"\n  Best Val Acc for Fold {fold+1}: {best_val_acc:.4f}")

    # ── Save training curves  (unchanged) ──────
    epochs_range = range(1, num_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, train_losses, label="Train Loss")
    axes[0].plot(epochs_range, val_losses,   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fold {fold+1} — Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, train_accuracies, label="Train Acc")
    axes[1].plot(epochs_range, val_accuracies,   label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Fold {fold+1} — Accuracy")
    axes[1].legend()

    plt.tight_layout()
    curve_path = os.path.join(results_dir, f"Fold{fold+1}_TrainingCurves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"  Training curves saved to {curve_path}")

    # ── Run test set inference (raw predictions, used only for subject-level aggregation below) ──
    model.load_state_dict(
        torch.load(os.path.join(results_dir, f"DenseNet121_CLAHE_Best_Fold{fold+1}.pth"))
    )
    model.eval()

    fold_y_true, fold_y_pred = [], []
    fold_y_probs = []  # softmax probs, needed for subject-level tie-break

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            fold_y_true.extend(labels.cpu().numpy())
            fold_y_pred.extend(preds.cpu().numpy())
            fold_y_probs.extend(probs.cpu().numpy())

    # ─────────────────────────────────────────────────────
    # NEW: SUBJECT-LEVEL aggregation for this fold
    # test_dataset.samples is in the same order as test_loader (shuffle=False),
    # so index i in fold_y_pred corresponds to test_dataset.samples[i].
    # ─────────────────────────────────────────────
    patient_true  = {}                 # patient_id -> true label (constant per patient)
    patient_preds = defaultdict(list)  # patient_id -> list of predicted labels (per slice)
    patient_probs = defaultdict(list)  # patient_id -> list of softmax prob vectors

    for (filepath, true_label), pred_label, probs in zip(test_dataset.samples,
                                                            fold_y_pred,
                                                            fold_y_probs):
        pid = get_patient_id(filepath)
        patient_true[pid] = true_label
        patient_preds[pid].append(pred_label)
        patient_probs[pid].append(probs)

    subject_y_true, subject_y_pred, subject_y_score, subject_ids = [], [], [], []
    for pid, preds_list in patient_preds.items():
        vote_counts = Counter(preds_list)
        top_count = max(vote_counts.values())
        tied_classes = [cls for cls, cnt in vote_counts.items() if cnt == top_count]

        if len(tied_classes) == 1:
            final_pred = tied_classes[0]
        else:
            # Tie-break: pick the class with the higher summed softmax confidence
            # across this patient's slices.
            summed_conf = np.sum(patient_probs[pid], axis=0)  # shape (num_classes,)
            final_pred = int(np.argmax(summed_conf))

        # Patient-level score for AUC: mean predicted probability of the
        # positive class (TRE) across that patient's slices.
        mean_probs = np.mean(patient_probs[pid], axis=0)  # shape (num_classes,)
        final_score = float(mean_probs[positive_idx])

        subject_y_true.append(patient_true[pid])
        subject_y_pred.append(final_pred)
        subject_y_score.append(final_score)
        subject_ids.append(pid)

    n_test_patients = len(subject_ids)
    print(f"\n  Subject-level test patients this fold: {n_test_patients} "
          f"(expected 12 = 6 TRE + 6 NonTRE)")

    subject_test_acc = np.sum(np.array(subject_y_true) == np.array(subject_y_pred)) / n_test_patients
    print(f"  Test Acc for Fold {fold+1} (SUBJECT-level): {subject_test_acc:.4f}")
    fold_subject_test_results.append(subject_test_acc)

    all_subject_y_true.extend(subject_y_true)
    all_subject_y_pred.extend(subject_y_pred)
    all_subject_y_score.extend(subject_y_score)

    # ── Test Accuracy / F1 / Sensitivity / Specificity / NPV / AUC (subject-level) ──
    fold_metrics = compute_subject_metrics(subject_y_true, subject_y_pred, subject_y_score,
                                            positive_idx, negative_idx)
    fold_metrics_list.append(fold_metrics)
    print(f"\n  Fold {fold+1} Subject-level Metrics:")
    for m_name, m_val in fold_metrics.items():
        print(f"    {m_name:<15}: {m_val:.4f}")

    # Per-fold SUBJECT-level confusion matrix (should be 12 total: 6 TRE, 6 NonTRE)
    cm_subject = confusion_matrix(subject_y_true, subject_y_pred,
                                   labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_subject, annot=True, fmt="d", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fold {fold+1} — Confusion Matrix (Subject-level, n={n_test_patients})")
    plt.tight_layout()
    cm_subj_path = os.path.join(results_dir, f"Fold{fold+1}_ConfusionMatrix_SubjectLevel.png")
    plt.savefig(cm_subj_path, dpi=150)
    plt.close()
    print(f"  Subject-level confusion matrix saved to {cm_subj_path}")

    print(f"\n  Classification Report — Fold {fold+1} Test Slice (subject-level):")
    print(classification_report(subject_y_true, subject_y_pred,
                                target_names=class_names,
                                labels=list(range(len(class_names)))))

# ─────────────────────────────────────────────
# K-Fold Summary  (subject-level)
# ─────────────────────────────────────────────
mean_val_acc     = np.mean(fold_val_results)
std_val_acc      = np.std(fold_val_results)
mean_subject_acc = np.mean(fold_subject_test_results)
std_subject_acc  = np.std(fold_subject_test_results)

print("\n" + "="*50)
print("       K-FOLD CROSS-VALIDATION SUMMARY (subject-level)")
print("="*50)
print(f"  {'Fold':<8} {'Best Val Acc':>14} {'Subject Test Acc':>18}")
print(f"  {'-'*42}")
for i in range(k_folds):
    print(f"  Fold {i+1:<3}  {fold_val_results[i]:>14.4f}  {fold_subject_test_results[i]:>18.4f}")
print(f"  {'-'*42}")
print(f"  {'Mean':<8} {mean_val_acc:>14.4f}  {mean_subject_acc:>18.4f}")
print(f"  {'Std Dev':<8} {std_val_acc:>14.4f}  {std_subject_acc:>18.4f}")
print("="*50)

# ─────────────────────────────────────────────
# Test Accuracy / F1-Score / Sensitivity / Specificity / NPV / AUC
# — per fold, mean ± std across folds, and overall (all 60 patients pooled)
# ─────────────────────────────────────────────
metric_names = ["Test Accuracy", "F1-Score", "Sensitivity", "Specificity", "NPV", "AUC"]

overall_metrics = compute_subject_metrics(all_subject_y_true, all_subject_y_pred,
                                           all_subject_y_score, positive_idx, negative_idx)

print("\n" + "="*90)
print("       SUBJECT-LEVEL METRICS SUMMARY")
print("="*90)
header = f"  {'Fold':<10}" + "".join(f"{m:>14}" for m in metric_names)
print(header)
print("  " + "-"*88)
for i, fm in enumerate(fold_metrics_list):
    row = f"  Fold {i+1:<5}" + "".join(f"{fm[m]:>14.4f}" for m in metric_names)
    print(row)
print("  " + "-"*88)
mean_row = f"  {'Mean':<10}" + "".join(
    f"{np.nanmean([fm[m] for fm in fold_metrics_list]):>14.4f}" for m in metric_names)
std_row  = f"  {'Std Dev':<10}" + "".join(
    f"{np.nanstd([fm[m] for fm in fold_metrics_list]):>14.4f}" for m in metric_names)
print(mean_row)
print(std_row)
print("  " + "-"*88)
overall_row = f"  {'Overall':<10}" + "".join(f"{overall_metrics[m]:>14.4f}" for m in metric_names)
print(overall_row)
print("  (Overall = all 60 test patients pooled together, one prediction each)")
print("="*90)

# Save the same table to CSV for the report
metrics_csv_path = os.path.join(results_dir, "Subject_Level_Metrics.csv")
with open(metrics_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold"] + metric_names)
    for i, fm in enumerate(fold_metrics_list):
        writer.writerow([f"Fold {i+1}"] + [fm[m] for m in metric_names])
    writer.writerow(["Mean"] + [np.nanmean([fm[m] for fm in fold_metrics_list]) for m in metric_names])
    writer.writerow(["Std Dev"] + [np.nanstd([fm[m] for fm in fold_metrics_list]) for m in metric_names])
    writer.writerow(["Overall (pooled)"] + [overall_metrics[m] for m in metric_names])
print(f"\nSubject-level metrics table saved to {metrics_csv_path}")

# ─────────────────────────────────────────────
# Aggregated SUBJECT-level confusion matrix across all 5 folds.
# Every one of the 60 patients (30 TRE + 30 NonTRE) appears exactly once
# across the 5 folds' test sets, so this is a full-dataset subject-level
# confusion matrix (60 total: 30 TRE, 30 NonTRE).
# ─────────────────────────────────────────────
print("\nGenerating aggregated SUBJECT-level confusion matrix across all folds...")
cm_all_subject = confusion_matrix(all_subject_y_true, all_subject_y_pred,
                                   labels=list(range(len(class_names))))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_all_subject, annot=True, fmt="d", ax=ax,
            xticklabels=class_names, yticklabels=class_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Aggregated Confusion Matrix — All Folds "
             f"(Subject-level, n={len(all_subject_y_true)})")
plt.tight_layout()
agg_cm_subj_path = os.path.join(results_dir, "Aggregated_ConfusionMatrix_SubjectLevel.png")
plt.savefig(agg_cm_subj_path, dpi=150)
plt.close()
print(f"Aggregated subject-level confusion matrix saved → {agg_cm_subj_path}")

print("\nAggregated Classification Report — subject-level (all folds):")
print(classification_report(all_subject_y_true, all_subject_y_pred,
                            target_names=class_names,
                            labels=list(range(len(class_names)))))

# ─────────────────────────────────────────────
# Fold accuracy bar chart  (subject-level val vs test)
# ─────────────────────────────────────────────
x      = np.arange(k_folds)
width  = 0.35
labels = [f"Fold {i+1}" for i in range(k_folds)]

fig, ax = plt.subplots(figsize=(10, 5))
bars_val  = ax.bar(x - width/2, fold_val_results,          width, label="Best Val Acc",        color='steelblue',  edgecolor='black')
bars_test = ax.bar(x + width/2, fold_subject_test_results, width, label="Subject Test Acc",    color='darkorange', edgecolor='black')

ax.axhline(mean_val_acc,     color='steelblue',  linestyle='--', alpha=0.7, label=f"Mean Val  = {mean_val_acc:.4f}")
ax.axhline(mean_subject_acc, color='darkorange', linestyle='--', alpha=0.7, label=f"Mean Test = {mean_subject_acc:.4f}")

for bar, val in zip(bars_val,  fold_val_results):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars_test, fold_subject_test_results):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("DenseNet121 (CLAHE) — Per-Fold Validation & Subject-level Test Accuracy")
ax.legend()
plt.tight_layout()
summary_path = os.path.join(results_dir, "Fold_Accuracy_Summary.png")
plt.savefig(summary_path, dpi=150)
plt.close()
print(f"Fold accuracy summary chart saved to {summary_path}")

print(f"\nAll results saved in '{results_dir}/' folder. Done!")