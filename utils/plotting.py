import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def save_confusion_matrix(labels, preds, class_names, output_dir, epoch=None, acc=None):
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Purples", xticklabels=class_names, yticklabels=class_names, cbar=False)
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    title = "Confusion Matrix"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title)
    
    filename = "confusion_matrix.png" if epoch is None else f"confusion_matrix_epoch_{epoch}_acc_{acc:.4f}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    # Keep only the best top 3 confusion matrices based on accuracy
    saved_files = sorted([f for f in os.listdir(output_dir) if f.startswith("confusion_matrix_epoch_")], key=lambda x: float(x.split('_acc_')[-1].split('.png')[0]), reverse=True)
    for file in saved_files[3:]:
        os.remove(os.path.join(output_dir, file))

def save_roc_curve(labels, preds, class_names, output_dir, epoch=None, acc=None):
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
    filename = "roc_curve.png" if epoch is None else f"roc_curve_epoch_{epoch}_acc_{acc:.4f}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    # Keep only the best top 3 ROC curves based on accuracy
    saved_files = sorted([f for f in os.listdir(output_dir) if f.startswith("roc_curve_epoch_")], key=lambda x: float(x.split('_acc_')[-1].split('.png')[0]), reverse=True)
    for file in saved_files[3:]:
        os.remove(os.path.join(output_dir, file))

def tr_plot(tr_data, start_epoch, output_dir):
    # Plot the training and validation data
    tacc = tr_data['accuracy']
    tloss = tr_data['loss']
    vacc = tr_data['val_accuracy']
    vloss = tr_data['val_loss']
    Epoch_count = len(tacc)
    Epochs = list(range(start_epoch + 1, start_epoch + Epoch_count + 1))
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'logs', f"training_plot.png"))
    plt.close()