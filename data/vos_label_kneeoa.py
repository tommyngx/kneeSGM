import os
import pandas as pd
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml
import shutil
import requests

def parse_arguments():
    parser = argparse.ArgumentParser(description="Label knee osteoarthritis images based on CSV data.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing knee OA data.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing cropped knee images.')
    parser.add_argument('--config_file', type=str, default='config/default.yaml', help='Path to the YAML configuration file.')
    return parser.parse_args()

def load_csv_data(csv_file):
    return pd.read_csv(csv_file).fillna("N/A")

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_image_info(image_name):
    age = image_name[:2]
    code_id = image_name[2:4]
    sex = image_name[4]
    id_ = int(image_name[5:9])
    location = image_name[9]
    kl_score = image_name[10]
    return age, code_id, sex, id_, location, kl_score

def download_font(font_url, font_path):
    if not os.path.exists(font_path):
        response = requests.get(font_url)
        with open(font_path, 'wb') as f:
            f.write(response.content)

def draw_text_on_image(image, text):
    draw = ImageDraw.Draw(image, "RGBA")
    font_url = 'https://github.com/tommyngx/style/blob/main/arial.ttf?raw=true'
    font_path = 'arial.ttf'
    download_font(font_url, font_path)
    font = ImageFont.truetype(font_path, 17)  # Increase font size
    text_position = (10, 10)
    text_bbox = draw.textbbox(text_position, text, font=font)
    background_position = (text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5)
    draw.rectangle(background_position, fill=(0, 0, 0, int(255 * 0.2)))
    draw.text(text_position, text, font=font, fill="white")
    return image

def label_image(image_path, info, output_folder):
    image = Image.open(image_path)
    image = image.resize((600, 600))  # Resize image to 600x600
    sex_label = "Male" if info['Sex'] else "Female"
    label = f"ID: {info['ID']}\nSex: {sex_label}\nAge: {info['Age']}\n"
    if info['Location'] == 'R':
        if info['Gai Trái'] != 0 and info['Gai Trái'] != "N/A":
            label += f"Gai Trái: {info['Gai Trái']}\n"
        if info['Số Gai Lớn Trái'] != 0 and info['Số Gai Lớn Trái'] != "N/A":
            label += f"Số Gai Lớn Trái: {info['Số Gai Lớn Trái']}\n"
        if info['Vị Trí Gai Lớn Trái'] != 0 and info['Vị Trí Gai Lớn Trái'] != "N/A":
            label += f"Vị Trí Gai Lớn Trái: {info['Vị Trí Gai Lớn Trái']}\n"
        if info['Số Gai Nhỏ Trái'] != 0 and info['Số Gai Nhỏ Trái'] != "N/A":
            label += f"Số Gai Nhỏ Trái: {info['Số Gai Nhỏ Trái']}\n"
        if info['Vị Trí Gai Nhỏ Trái'] != 0 and info['Vị Trí Gai Nhỏ Trái'] != "N/A":
            label += f"Vị Trí Gai Nhỏ Trái: {info['Vị Trí Gai Nhỏ Trái']}\n"
        if info['Hẹp Khớp Trái'] != 0 and info['Hẹp Khớp Trái'] != "N/A":
            label += f"Hẹp Khớp Trái: {info['Hẹp Khớp Trái']}\n"
        if info['Vị Trí Khớp Trái'] != 0 and info['Vị Trí Khớp Trái'] != "N/A":
            label += f"Vị Trí Khớp Trái: {info['Vị Trí Khớp Trái']}\n"
        if info['Xơ Sụn Trái'] != 0 and info['Xơ Sụn Trái'] != "N/A":
            label += f"Xơ Sụn Trái: {info['Xơ Sụn Trái']}\n"
        if info['Vị Trí Xơ Sụn Trái'] != 0 and info['Vị Trí Xơ Sụn Trái'] != "N/A":
            label += f"Vị Trí Xơ Sụn Trái: {info['Vị Trí Xơ Sụn Trái']}\n"
        label += f"KL Trái: {info['KL Trái']}\n"
    else:
        if info['Gai Phải'] != 0 and info['Gai Phải'] != "N/A":
            label += f"Gai Phải: {info['Gai Phải']}\n"
        if info['Gai Lớn Phải'] != 0 and info['Gai Lớn Phải'] != "N/A":
            label += f"Gai Lớn Phải: {info['Gai Lớn Phải']}\n"
        if info['Vị Trí Gai Lớn Phải'] != 0 and info['Vị Trí Gai Lớn Phải'] != "N/A":
            label += f"Vị Trí Gai Lớn Phải: {info['Vị Trí Gai Lớn Phải']}\n"
        if info['Gai Nhỏ Phải'] != 0 and info['Gai Nhỏ Phải'] != "N/A":
            label += f"Gai Nhỏ Phải: {info['Gai Nhỏ Phải']}\n"
        if info['Vị Trí Gai Nhỏ Phải'] != 0 and info['Vị Trí Gai Nhỏ Phải'] != "N/A":
            label += f"Vị Trí Gai Nhỏ Phải: {info['Vị Trí Gai Nhỏ Phải']}\n"
        if info['Hẹp Khớp Phải'] != 0 and info['Hẹp Khớp Phải'] != "N/A":
            label += f"Hẹp Khớp Phải: {info['Hẹp Khớp Phải']}\n"
        if info['Vị Trí Khớp Phải'] != 0 and info['Vị Trí Khớp Phải'] != "N/A":
            label += f"Vị Trí Khớp Phải: {info['Vị Trí Khớp Phải']}\n"
        if info['Xơ Sụn Phải'] != 0 and info['Xơ Sụn Phải'] != "N/A":
            label += f"Xơ Sụn Phải: {info['Xơ Sụn Phải']}\n"
        if info['Vị Trí Xơ Sụn Phải'] != 0 and info['Vị Trí Xơ Sụn Phải'] != "N/A":
            label += f"Vị Trí Xơ Sụn Phải: {info['Vị Trí Xơ Sụn Phải']}\n"
        label += f"KL Phải: {info['KL Phải']}\n"
    
    image = draw_text_on_image(image, label)
    output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.png', '_labels.png'))
    image.save(output_path)

def main():
    args = parse_arguments()
    csv_data = load_csv_data(args.csv_file)
    config = load_config(args.config_file)
    output_folder = os.path.join(config['output_dir'], os.path.basename(args.image_folder) + '_labels')
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(args.image_folder):
        for dir_name in dirs:
            os.makedirs(os.path.join(output_folder, dir_name), exist_ok=True)
        for file in tqdm(files):
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                age, code_id, sex, id_, location, kl_score = get_image_info(file)
                info = csv_data[csv_data['ID'] == id_].iloc[0].to_dict()
                info['Location'] = location
                label_image(image_path, info, os.path.join(output_folder, os.path.relpath(root, args.image_folder)))

if __name__ == "__main__":
    main()
