import torch
import os
import glob
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

root_dir = "data/coco"
train_dir = os.path.join(root_dir, 'train')
gallery_dir = os.path.join(root_dir, 'gallery')
val_dir = os.path.join(root_dir, 'val2017')

os.makedirs(os.path.join(train_dir, 'features'), exist_ok=True)
os.makedirs(os.path.join(gallery_dir, 'features'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
