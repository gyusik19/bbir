import torch
import torch.nn.functional as F
import os
import json
import fasttext, fasttext.util
import math

from torchvision import datasets, transforms
from pycocotools.coco import COCO
from PIL import Image

word2vec = None

def one_hot_embedding(label, num_classes):
    '''
    Embedding labels to one-hot form.

    Parameters:
    - labels: (LongTensor) class labels, sized [N,].
    - num_classes: (int) number of classes.

    Returns:
    - (tensor) encoded labels, sized [N, #classes].
    '''
    y = torch.eye(num_classes) 
    return y[label]

def convert_query_to_tensor(query, num_dim, mode='one-hot', embedding_fn=None):
    """
    convert query to tensor
    """
    width, height = 31, 31
    obj = query['layout']
    query_tensor = torch.zeros((width, height, num_dim), dtype=torch.float32)
    if mode == 'one-hot':
        cat_embedding = embedding_fn(obj['label'], num_dim)
    else:
        cat_embedding = torch.tensor(embedding_fn(obj['category']))
    x, y, w, h = obj['bbox']
    x = x * width
    y = y * height
    w = w * width
    h = h * height

    min_x, min_y, max_x, max_y = int(x - w / 2), int(y - h / 2), math.ceil(x + w / 2), math.ceil(y + h / 2)
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, width)
    max_y = min(max_y, height)

    query_tensor[min_y:max_y, min_x:max_x, :] = cat_embedding
    query_tensor = query_tensor.permute(2, 0, 1)
    # rescaled_tensor = F.interpolate(query_tensor.permute(2, 0, 1).unsqueeze(0), size=(31, 31), mode='bilinear', align_corners=False).squeeze(0)
    return query_tensor

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root='../data/coco', mode='train'):
        self.root = root
        self.dir = os.path.join(root, mode)
        self.queries_path = os.path.join(self.dir, 'query.json')
        with open(self.queries_path, 'r') as f:
            self.queries = json.load(f)
    

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        img_id = self.queries[index]['img_id']
        layouts = self.queries[index]['layout']
        W, H = self.queries[index]['W'], self.queries[index]['H']
        return img_id, layouts, (W, H)

def collate_fn(batch):
    img_ids = [item[0] for item in batch]
    layouts = [item[1] for item in batch]
    sizes = [item[2] for item in batch]
    return img_ids, layouts, sizes


if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CocoDataset(root='./data/coco', mode='val2017')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for i, (img_id, layouts, sizes) in enumerate(dataloader):
        print(img_id, layouts, sizes)