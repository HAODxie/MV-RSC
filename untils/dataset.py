import json
from torch.utils.data import Dataset
from PIL import Image


class GrayscaleDataset(Dataset):
    def __init__(self, json_path, img_size, data_key, transform=None):
        with open(json_path) as f:
            self.data = json.load(f)[data_key]
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image']
        label = self.data[idx]['label']
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, label


class GrayscaleDatasenm(Dataset):
    def __init__(self, json_path, img_size, transform=None):
        with open(json_path) as f:
            data_dict = json.load(f)
            self.data = data_dict['data']
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image']
        label = self.data[idx]['label']

        try:
            img = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        if self.transform:
            img = self.transform(img)

        return img, label