import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import requests
from matplotlib import font_manager
from scipy import interp
from sklearn.utils import resample

def save_confusion_matrix_ori(labels, preds, class_names, output_dir, epoch=None, acc=None):
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".0f", cmap="Purples", xticklabels=class_names, yticklabels=class_names, cbar=False)
    
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

def save_confusion_matrix(labels, preds, class_names, output_dir, epoch=None, acc=None):
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Create an annotation matrix with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            count = cm[i, j]
            percent = cm_normalized[i, j] * 100
            annot[i, j] = f'{percent:.1f}%\n({count})'
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=annot, fmt='', cmap="Purples", xticklabels=class_names, yticklabels=class_names, cbar=True)
    # Customize the color bar
    cbar = plt.gca().collections[0].colorbar  # Get the color bar from the current Axes
    ticks = np.linspace(0, 100, 6)  # Define the ticks
    cbar.set_ticks(ticks)  # Set specific ticks
    cbar.set_ticklabels([f'{int(t)}%' for t in ticks]) # Format tick labels as percentages
        
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    title = "Confusion Matrix"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title)
    
    # Save the figure
    filename = "confusion_matrix.png" if epoch is None else f"confusion_matrix_epoch_{epoch}_acc_{acc:.4f}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    # Keep only the top 3 confusion matrices based on accuracy
    saved_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("confusion_matrix_epoch_")],
        key=lambda x: float(x.split('_acc_')[-1].split('.png')[0]),
        reverse=True
    )
    for file in saved_files[3:]:
        os.remove(os.path.join(output_dir, file))

def save_roc_curve(labels, positive_risk, class_names, output_dir, epoch=None, acc=None):
    # Apply ggplot style
    plt.style.use('ggplot')

    # Download the font file
    font_url = 'https://github.com/tommyngx/style/blob/main/Poppins.ttf?raw=true'
    font_path = 'Poppins.ttf'
    response = requests.get(font_url)
    with open(font_path, 'wb') as f:
        f.write(response.content)

    # Load the font
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    # Binarize labels for disease detection
    labels = np.array(labels)
    labels = np.where(labels > 1, 1, 0)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, positive_risk, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Bootstrap for confidence interval
    bootstrapped_scores = []
    for i in range(1000):
        indices = resample(np.arange(len(labels)), replace=True)
        if len(np.unique(labels[indices])) < 2:
            continue
        score = auc(*roc_curve(labels[indices], positive_risk[indices])[:2])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    plt.figure(figsize=(10, 9))
    plt.plot(fpr, tpr, color='darkred', lw=2, label=f'AUC: {roc_auc*100:.0f}% ({confidence_lower*100:.0f}% - {confidence_upper*100:.0f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('100 - Specificity (%)', fontproperties=prop, fontsize=16)
    plt.ylabel('Sensitivity (%)', fontproperties=prop, fontsize=16)
    plt.xticks(np.arange(0, 1.1, step=0.1), labels=[f'{int(x*100)}%' for x in np.arange(0, 1.1, step=0.1)], fontsize=15)
    plt.yticks(np.arange(0, 1.1, step=0.1), labels=[f'{int(y*100)}%' for y in np.arange(0, 1.1, step=0.1)], fontsize=15)
    title = 'Receiver Operating Characteristic'
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title, fontproperties=prop, fontsize=18)
    
    # Customize legend
    legend = plt.legend(loc="lower right", prop=prop, fontsize=18)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    plt.gcf().set_facecolor('white')  # Set the background color outside the plot area to white
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
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