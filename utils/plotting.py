import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def save_confusion_matrix(labels, preds, class_names, output_dir, epoch=None):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = "Confusion Matrix"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title)
    filename = "confusion_matrix.png" if epoch is None else f"confusion_matrix_epoch_{epoch}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def save_roc_curve(labels, preds, class_names, output_dir, epoch=None):
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'Receiver Operating Characteristic'
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title)
    plt.legend(loc="lower right")
    filename = "roc_curve.png" if epoch is None else f"roc_curve_epoch_{epoch}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
