import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def evaluate_biometrics(labels, scores, far_target=1e-3):
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[eer_index]
    threshold_eer = thresholds[eer_index]
    
    f = interp1d(fpr, tpr, kind='linear')
    tar_at_target = f(far_target)
    
    pred_labels = [1 if s >= threshold_eer else 0 for s in scores]
    cm = confusion_matrix(labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*40)
    print("BIOMETRIC EVALUATION REPORT")
    print("="*40)
    print(f"1. Stability Metric (AUC):        {roc_auc:.4f}  (>0.90 is Pro)")
    print(f"2. Balance Metric (EER):          {eer*100:.2f}%   (Lower is Better)")
    print(f"   -> EER Threshold:              {threshold_eer:.4f}")
    print("-" * 40)
    print(f"3. High-Security Metric (TAR @ FAR={far_target}):")
    print(f"   -> {tar_at_target*100:.2f}% of Genuine users are accepted")
    print(f"      when only 1 in {int(1/far_target)} Imposters gets in.")
    print("-" * 40)
    print(f"4. Confusion Matrix (at EER Threshold):")
    print(f"   [ TP (Valid Matches): {tp:<5} | FN (False Rejects): {fn:<5} ]")
    print(f"   [ FP (False Accepts): {fp:<5} | TN (Valid Blocks):  {tn:<5} ]")
    print("="*40)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    #ROC Curve (Standard)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[0].scatter(eer, 1-eer, color='red', label=f'EER = {eer:.2f}', zorder=5)
    axes[0].set_title("A. ROC Curve (Stability)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(fpr, tpr, color='green', lw=2)
    axes[1].set_xlim([0.0, 0.05])
    axes[1].set_ylim([0.6, 1.0])
    axes[1].axvline(far_target, color='purple', linestyle=':', label=f'FAR target {far_target}')
    axes[1].scatter(far_target, tar_at_target, color='purple')
    axes[1].set_title(f"B. Security View (Zoom @ {far_target} FAR)")
    axes[1].set_xlabel("False Positive Rate (Imposters)")
    axes[1].set_ylabel("True Verification Rate (Legits)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Match', 'Match'])
    disp.plot(cmap='Blues', ax=axes[2], colorbar=False)
    axes[2].set_title(f"C. Confusion Matrix @ EER")
    
    plt.tight_layout()
    plt.savefig('biometric_report_card.png')
    print("\nSaved charts to 'biometric_report_card.png'")
    
    return roc_auc, eer, tar_at_target
