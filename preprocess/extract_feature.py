import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from PIL import Image
from model.model import ResNet50


root_dir = "data/coco"
dir_list = ['train', 'gallery', 'val2017']
# train_dir = os.path.join(root_dir, 'train')
# gallery_dir = os.path.join(root_dir, 'gallery')
# val_dir = os.path.join(root_dir, 'val2017')

# os.makedirs(os.path.join(train_dir, 'features'), exist_ok=True)
# os.makedirs(os.path.join(gallery_dir, 'features'), exist_ok=True)
# os.makedirs(os.path.join(val_dir, 'features'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50(pretrained=True)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for dir_name in dir_list:
    data_dir = os.path.join(root_dir, dir_name)
    img_dir = os.path.join(data_dir, 'images')
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
    os.makedirs(os.path.join(data_dir, 'features'), exist_ok=True)
    features_dict = {}
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        img_id = int(os.path.basename(img_path).split('.')[0])
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            feature = model(img_tensor)
        feature = feature.squeeze(0)
        features_dict[img_id] = feature.cpu()
    feature_path = os.path.join(data_dir, 'features', 'resnet50_features.pt')
    torch.save(features_dict, feature_path)
    print("Done with {}".format(dir_name))
