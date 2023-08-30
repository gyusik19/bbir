import torch
import torch.nn.functional as F
import os
import json

from torchvision import datasets, transforms
from pycocotools.coco import COCO
from PIL import Image

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

def convert_query_to_tensor(query, num_classes, mode='one-hot'):
    """
    convert query to tensor
    """
    width, height = 31, 31
    obj = query['layout']
    query_tensor = torch.zeros((width, height, num_classes), dtype=torch.float32)
    if mode == 'one-hot':
        label_vector = one_hot_embedding(obj['label'], num_classes)
    else:
        NotImplementedError
    x, y, w, h = obj['bbox']
    x, y, w, h = int(x * width), int(y * height), int(w * width), int(h * height)
    min_x, min_y, max_x, max_y = x - w // 2, y - h // 2, x + w // 2, y + h // 2
    query_tensor[min_y:max_y, min_x:max_x, :] = label_vector
    query_tensor = query_tensor.permute(2, 0, 1)
    # rescaled_tensor = F.interpolate(query_tensor.permute(2, 0, 1).unsqueeze(0), size=(31, 31), mode='bilinear', align_corners=False).squeeze(0)
    return query_tensor

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root='../data/coco', transform=None, mode='train'):
        self.root = root
        self.train_dir = os.path.join(root, mode)
        self.gallery_dir = os.path.join(root, 'gallery')

        # with open(os.path.join(root, 'annotations', 'instances_train2017.json'), 'r') as f:
        #     self.coco_ann = json.load(f)
        
        # self.id_to_name = {category['id']: category['name'] for category in self.coco_ann['categories']}
        with open(os.path.join(root, 'labels.txt'), 'r') as f:
            labels = f.read().split('\n')
            self.name_to_id = {label: i for i, label in enumerate(labels)}

        self.num_classes = len(self.name_to_id)
        self.transform = transform
        self.mode = mode
        self.train_img_path = os.path.join(self.train_dir, 'images')
        self.gallery_img_path = os.path.join(self.gallery_dir, 'images')
        
        self.train_queries_path = os.path.join(self.train_dir, 'query.json')
        with open(self.train_queries_path, 'r') as f:
            self.train_queries = json.load(f)
    

    def __len__(self):
        return len(self.train_queries)

    def __getitem__(self, index):
        canvas_tensor = convert_query_to_tensor(self.train_queries[index], self.num_classes)
        item = self.train_queries[index]['layout']
        img_id = self.train_queries[index]['img_id']
        query = (item['category'], item['bbox'], item['area'], img_id)
        image_name = "{:012}".format(self.train_queries[index]['img_id']) + '.jpg'
        image = Image.open(os.path.join(self.train_img_path, image_name))
        if self.transform:
            # the image sample can have channels of 1 or 4, we want to convert them to 3
            if image.mode == 'L':
                image = image.convert('RGB')
            image = self.transform(image)

        return image, canvas_tensor, query

def collate_fn(batch):
    images = [item[0] for item in batch]
    canvas_tensors = [item[1] for item in batch]
    queries = [item[2] for item in batch]

    # Stack images and canvas_tensors
    images = torch.stack(images)
    canvas_tensors = torch.stack(canvas_tensors)

    # For 'bbox' in queries, we want to stack along the second dimension
    bboxes = [query[1] for query in queries]
    categories = [query[0] for query in queries]
    areas = [query[2] for query in queries]
    img_ids = [query[3] for query in queries]

    return images, canvas_tensors, (categories, bboxes, areas, img_ids)


if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CocoDataset(root='./data/coco', mode='train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for i, (image, canvas_tensor, query) in enumerate(dataloader):
        print(image.shape, canvas_tensor.shape)
        break