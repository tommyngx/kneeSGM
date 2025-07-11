# "Opportunities don't happen, you create them" — Chris Grosser

The process of using the image classification model is carried out as follows.

## 1. Data Preparation

- Organize your dataset in the following folder structure:
  ```
  data/
    train/
      class_1/
        img1.jpg
        img2.jpg
        ...
      class_2/
        ...
      ...
    val/
      class_1/
        ...
      class_2/
        ...
      ...
    test/
      ...
  ```
- Each subfolder represents a class label.
- Ensure a balanced number of images per class for optimal training.
- You may use the preprocessing scripts in `data/preprocess.py` for resizing, augmentation, or normalization.

## 2. Environment Setup

- Python >= 3.8 is required.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Main libraries include: `torch`, `torchvision`, `albumentations`, `opencv-python`, `matplotlib`, `tqdm`, etc.

## 3. Model Training

- Edit the configuration file in `config/default.yaml` as needed.
- Start training with:
  ```bash
  python run/train.py --config config/default.yaml --model_name resnet50
  ```
- The `--model_name` argument can be changed to other supported architectures (e.g., resnet18, resnet50, efficientnet, ...).
- Model checkpoints will be saved in the `checkpoints/` directory.

## 4. Model Evaluation and Testing

- To evaluate a trained model, run:
  ```bash
  python run/test_all.py --config config/default.yaml --model_name resnet50 --model_path checkpoints/resnet50_epoch_10.pth
  ```
- The script will output accuracy, confusion matrix, and other evaluation metrics.

## 5. Visualization

- For visualizing model predictions and feature localization, use:
  ```bash
  python run/plot_model.py \
      --config config/default.yaml \
      --model_name resnet50 \
      --model_path checkpoints/resnet50_epoch_10.pth \
      --yolo_model_path yolov8n.pt \
      --output_path gradcam_yolo_plot.png
  ```
- The script will generate images showing the original, GradCAM heatmaps, and detected features for qualitative analysis.

## 6. Notes

- Ensure all file paths for configuration, checkpoints, and models are correct.
- Pretrained weights for auxiliary models (if used) should be downloaded in advance.
- If using custom fonts for visualization, make sure the font file is available in your working directory.

## 7. Contact & Contribution

- For questions or contributions, please contact via email or open an issue on this GitHub repository.
