import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
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
    
    # Calculate additional metrics
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    
    # FAR and FRR (False Acceptance Rate and False Rejection Rate)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Specificity and Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n" + "="*50)
    print("BIOMETRIC EVALUATION REPORT")
    print("="*50)
    print("📊 STABILITY & DISCRIMINATION METRICS:")
    print(f"  1. AUC (Area Under Curve):       {roc_auc:.4f}  (>0.90 is excellent)")
    print(f"  2. EER (Equal Error Rate):       {eer*100:.2f}%   (Lower is better)")
    print(f"     -> Threshold at EER:         {threshold_eer:.4f}")
    print("-" * 50)
    print("✓ CLASSIFICATION PERFORMANCE:")
    print(f"  3. Accuracy:                     {accuracy:.4f}  (Overall correctness)")
    print(f"  4. Precision:                    {precision:.4f}  (Valid matches among positives)")
    print(f"  5. Recall (Sensitivity):         {recall:.4f}  (Genuine matches detected)")
    print(f"  6. F1-Score:                     {f1:.4f}  (Harmonic mean of P & R)")
    print("-" * 50)
    print("🔐 BIOMETRIC-SPECIFIC RATES:")
    print(f"  7. FAR (False Acceptance):       {far*100:.4f}%  (Imposters incorrectly accepted)")
    print(f"  8. FRR (False Rejection):        {frr*100:.4f}%  (Genuine users rejected)")
    print(f"  9. Specificity:                  {specificity:.4f}  (True negatives correctly identified)")
    print(f"  10. Sensitivity (True Pos Rate): {sensitivity:.4f}  (True positives correctly identified)")
    print("-" * 50)
    print(f"11. High-Security Metric (TAR @ FAR={far_target}):")
    print(f"    -> {tar_at_target*100:.2f}% of Genuine users accepted")
    print(f"       when only 1 in {int(1/far_target)} Imposters gets in")
    print("-" * 50)
    print(f"12. Confusion Matrix (at EER Threshold):")
    print(f"    [ TP (Valid Matches): {tp:<5} | FN (False Rejects): {fn:<5} ]")
    print(f"    [ FP (False Accepts): {fp:<5} | TN (Valid Blocks):  {tn:<5} ]")
    print("="*50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ROC Curve (Standard)
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random')
    axes[0, 0].scatter(eer, 1-eer, color='red', s=100, label=f'EER = {eer*100:.2f}%', zorder=5)
    axes[0, 0].set_title("A. ROC Curve (Model Discrimination)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("False Positive Rate (FPR)")
    axes[0, 0].set_ylabel("True Positive Rate (TPR)")
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(alpha=0.3)

    # Zoomed ROC for security view
    axes[0, 1].plot(fpr, tpr, color='green', lw=2)
    axes[0, 1].set_xlim([0.0, 0.05])
    axes[0, 1].set_ylim([0.6, 1.0])
    axes[0, 1].axvline(far_target, color='purple', linestyle=':', linewidth=2, label=f'FAR target {far_target}')
    axes[0, 1].scatter(far_target, tar_at_target, color='purple', s=100, zorder=5)
    axes[0, 1].set_title(f"B. Security View (FAR={far_target})", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Match (0)', 'Match (1)'])
    disp.plot(cmap='Blues', ax=axes[1, 0], colorbar=True)
    axes[1, 0].set_title(f"C. Confusion Matrix @ EER Threshold", fontsize=12, fontweight='bold')
    
    # Performance Metrics Bar Chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity']
    metrics_values = [accuracy, precision, recall, f1, specificity, sensitivity]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_ylabel("Score", fontsize=10)
    axes[1, 1].set_title("D. Classification Metrics Performance", fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('biometric_report_card.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved comprehensive report to 'biometric_report_card.png'")
    
    return {
        'roc_auc': roc_auc,
        'eer': eer,
        'tar_at_target': tar_at_target,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'far': far,
        'frr': frr,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'threshold_eer': threshold_eer,
        'confusion_matrix': cm
    }
