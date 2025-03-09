import os
import json
from pathlib import Path


def generate_dataset_json(data_dir, output_json_file):
    dataset = {}
    data_path = Path(data_dir)

    for i, sub_dir in enumerate(sorted(data_path.iterdir()), 1):
        if sub_dir.is_dir():
            dataset[f'data{i}'] = []

            for label, cls_dir in enumerate(sorted(sub_dir.iterdir())):
                if cls_dir.is_dir():
                    for img_file in cls_dir.glob('*.[jp][pn][g]'):
                        img_path = str(img_file).replace('\\', '/')
                        dataset[f'data{i}'].append({
                            'image': img_path,
                            'label': label
                        })

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Generated JSON: {output_json_file}")


def generate_test_dataset_json(data_dir, output_json_file):
    dataset = {'test': []}

    for top_folder in sorted(os.listdir(data_dir)):
        top_folder_path = os.path.join(data_dir, top_folder)
        if os.path.isdir(top_folder_path):
            subfolders = sorted(os.listdir(top_folder_path))
            for subfolder in subfolders:
                subfolder_path = os.path.join(top_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    label = subfolders.index(subfolder)
                    for img_name in os.listdir(subfolder_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(subfolder_path, img_name).replace('\\', '/')
                            dataset['test'].append({'image': img_path, 'label': label})

    with open(output_json_file, 'w') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Test JSON saved: {output_json_file}")


def generate_datasetn_json(data_dir, output_json_file):
    dataset = {'data': []}

    for top_folder in sorted(os.listdir(data_dir)):
        top_folder_path = os.path.join(data_dir, top_folder)
        if os.path.isdir(top_folder_path):
            subfolders = sorted(os.listdir(top_folder_path))
            for subfolder in subfolders:
                subfolder_path = os.path.join(top_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    label = subfolders.index(subfolder)
                    for img_name in os.listdir(subfolder_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(subfolder_path, img_name).replace('\\', '/')
                            dataset['data'].append({'image': img_path, 'label': label})

    with open(output_json_file, 'w') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Dataset JSON saved: {output_json_file}")


if __name__ == "__main__":
    data_directory = r'data\plaque\train'
    output_json = r'json\combined.json'
    generate_dataset_json(data_directory, output_json)

    test_directory = r'data\plaque\test'
    test_json = r'json\test.json'
    generate_test_dataset_json(test_directory, test_json)

    data1_directory = r'data\plaque\train'
    output1_json = r'json\data.json'
    generate_datasetn_json(data1_directory, output1_json)
