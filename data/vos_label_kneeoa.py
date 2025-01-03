import os
import pandas as pd
import argparse
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Label knee osteoarthritis images based on CSV data.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing knee OA data.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing cropped knee images.')
    return parser.parse_args()

def load_csv_data(csv_file):
    return pd.read_csv(csv_file)

def get_image_info(image_name):
    age = image_name[:2]
    code_id = image_name[2:6]
    sex = image_name[6]
    id_ = image_name[7:11]
    location = image_name[11]
    kl_score = image_name[12]
    return age, code_id, sex, id_, location, kl_score

def label_image(image_path, info):
    image = Image.open(image_path)
    label = f"ID: {info['ID']}\nSex: {info['Sex']}\nAge: {info['Age']}\n"
    if info['Location'] == 'L':
        label += f"Gai Trái: {info['Gai Trái']}\nSố Gai Lớn Trái: {info['Số Gai Lớn Trái']}\nVị Trí Gai Lớn Trái: {info['Vị Trí Gai Lớn Trái']}\n"
        label += f"Số Gai Nhỏ Trái: {info['Số Gai Nhỏ Trái']}\nVị Trí Gai Nhỏ Trái: {info['Vị Trí Gai Nhỏ Trái']}\nHẹp Khớp Trái: {info['Hẹp Khớp Trái']}\n"
        label += f"Vị Trí Khớp Trái: {info['Vị Trí Khớp Trái']}\nXơ Sụn Trái: {info['Xơ Sụn Trái']}\nVị Trí Xơ Sụn Trái: {info['Vị Trí Xơ Sụn Trái']}\nKL Trái: {info['KL Trái']}\n"
    else:
        label += f"Gai Phải: {info['Gai Phải']}\nGai Lớn Phải: {info['Gai Lớn Phải']}\nVị Trí Gai Lớn Phải: {info['Vị Trí Gai Lớn Phải']}\n"
        label += f"Gai Nhỏ Phải: {info['Gai Nhỏ Phải']}\nVị Trí Gai Nhỏ Phải: {info['Vị Trí Gai Nhỏ Phải']}\nHẹp Khớp Phải: {info['Hẹp Khớp Phải']}\n"
        label += f"Vị Trí Khớp Phải: {info['Vị Trí Khớp Phải']}\nXơ Sụn Phải: {info['Xơ Sụn Phải']}\nVị Trí Xơ Sụn Phải: {info['Vị Trí Xơ Sụn Phải']}\nKL Phải: {info['KL Phải']}\n"
    
    output_path = image_path.replace('.png', '_labels.png')
    image.save(output_path)
    with open(output_path.replace('.png', '.txt'), 'w') as f:
        f.write(label)

def main():
    args = parse_arguments()
    csv_data = load_csv_data(args.csv_file)
    
    for root, _, files in os.walk(args.image_folder):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                age, code_id, sex, id_, location, kl_score = get_image_info(file)
                info = csv_data[csv_data['ID'] == int(id_)].iloc[0].to_dict()
                info['Location'] = location
                label_image(image_path, info)

if __name__ == "__main__":
    main()
