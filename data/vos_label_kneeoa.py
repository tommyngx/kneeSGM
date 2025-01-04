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
    font = ImageFont.truetype(font_path, 16)
    text_position = (10, 10)
    text_size = draw.textsize(text, font=font)
    background_position = (text_position[0] - 5, text_position[1] - 5, text_position[0] + text_size[0] + 5, text_position[1] + text_size[1] + 5)
    draw.rectangle(background_position, fill=(0, 0, 0, int(255 * 0.2)))
    draw.text(text_position, text, font=font, fill="white")
    return image

def label_image(image_path, info, output_folder):
    image = Image.open(image_path)
    sex_label = "Male" if info['Sex'] else "Female"
    label = f"ID: {info['ID']}\nSex: {sex_label}\nAge: {info['Age']}\n"
    if info['Location'] == 'L':
        label += f"Gai Trái: {info['Gai Trái']}\nSố Gai Lớn Trái: {info['Số Gai Lớn Trái']}\nVị Trí Gai Lớn Trái: {info['Vị Trí Gai Lớn Trái']}\n"
        label += f"Số Gai Nhỏ Trái: {info['Số Gai Nhỏ Trái']}\nVị Trí Gai Nhỏ Trái: {info['Vị Trí Gai Nhỏ Trái']}\nHẹp Khớp Trái: {info['Hẹp Khớp Trái']}\n"
        label += f"Vị Trí Khớp Trái: {info['Vị Trí Khớp Trái']}\nXơ Sụn Trái: {info['Xơ Sụn Trái']}\nVị Trí Xơ Sụn Trái: {info['Vị Trí Xơ Sụn Trái']}\nKL Trái: {info['KL Trái']}\n"
    else:
        label += f"Gai Phải: {info['Gai Phải']}\nGai Lớn Phải: {info['Gai Lớn Phải']}\nVị Trí Gai Lớn Phải: {info['Vị Trí Gai Lớn Phải']}\n"
        label += f"Gai Nhỏ Phải: {info['Gai Nhỏ Phải']}\nVị Trí Gai Nhỏ Phải: {info['Vị Trí Gai Nhỏ Phải']}\nHẹp Khớp Phải: {info['Hẹp Khớp Phải']}\n"
        label += f"Vị Trí Khớp Phải: {info['Vị Trí Khớp Phải']}\nXơ Sụn Phải: {info['Xơ Sụn Phải']}\nVị Trí Xơ Sụn Phải: {info['Vị Trí Xơ Sụn Phải']}\nKL Phải: {info['KL Phải']}\n"
    
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
