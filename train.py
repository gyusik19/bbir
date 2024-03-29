import argparse
import collections
import torch
import torchvision.transforms as transforms
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

from model.model import *
from data_loader.data_loaders import CocoDataset, collate_fn
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    logger.info(config)
    # setup data_loader instances
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = config['data_loader']['args']['batch_size']
    train_set = CocoDataset(root='./data/coco', mode='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=4, collate_fn=collate_fn)
    train_features = torch.load('./data/coco/train/features/resnet50_features.pt')

    valid_set = CocoDataset(root='./data/coco', mode='val2017')
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=4, collate_fn=collate_fn)
    valid_features = torch.load('./data/coco/val2017/features/resnet50_features.pt')

    embed_dim = config['arch']['args']['embed_dim']
    # build model architecture, then print to console
    model = FeatureSynthesisModel(embed_dim=embed_dim)
    backbone_model = ResNet50()
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    backbone_model = backbone_model.to(device)
    backbone_model.eval()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, backbone_model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      train_features=train_features,
                      valid_data_loader=valid_loader,
                      valid_features=valid_features,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default='./saved/models/clip/1016_155245/checkpoint-epoch28.pth', type=str,
    #                   help='path to latest checkpoint (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str)

    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
