import numpy as np
import torch
import torch.nn.functional as F
import fasttext, fasttext.util
import clip
import math
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from pycocotools.coco import COCO
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, create_bbox_mask
from utils import convert_query_to_tensor, one_hot_embedding, AP_at_k, clip_embedding_fn, coco_clip_embedding_fn


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, backbone_model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, train_features, valid_data_loader=None, valid_features=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.backbone_model = backbone_model
        self.train_features = train_features
        self.valid_features = valid_features

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.embed_mode =  self.config['arch']['args']['embed_mode']
        self.num_dim = self.config['arch']['args']['embed_dim']
        if self.embed_mode == 'one-hot':
            self.embed_fn = one_hot_embedding
        elif self.embed_mode == 'word2vec':
            fasttext.util.download_model('en', if_exists='ignore')
            word2vec = fasttext.load_model('cc.en.300.bin')
            self.embed_fn = word2vec.get_word_vector
        elif self.embed_mode == 'clip':
            self.embed_fn = clip_embedding_fn(device=self.device)
        elif self.embed_mode == 'coco_clip':
            self.embed_fn = coco_clip_embedding_fn()
        else:
            raise ValueError('Invalid embedding mode')

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (img_ids, layouts, sizes) in enumerate(tqdm(self.data_loader)):
            batch_size = len(img_ids)
            self.optimizer.zero_grad()
            target_feature = [self.train_features[img_id] for img_id in img_ids]
            target_feature = torch.stack(target_feature).to(self.device)
            canvas_queries = [convert_query_to_tensor(layouts[i], self.num_dim, mode=self.embed_mode, embedding_fn=self.embed_fn)[0] for i in range(batch_size)]
            canvas_queries = torch.stack(canvas_queries).to(self.device)
            output_feature = self.model(canvas_queries)

            # Create a mask tensor for the entire batch
            masks = torch.stack([create_bbox_mask(layouts[i][0]['bbox'], 7) for i in range(batch_size)]).to(self.device)
            masks = masks.unsqueeze(1)  # Add channel dimension: [batch_size, 1, 7, 7]

            # Multiply the feature tensors with the batched mask tensor
            output_feature = output_feature * masks
            target_feature = target_feature * masks
            

            loss = self.criterion(output_feature, target_feature)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output_feature, target_feature))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(imgs.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        # if self.do_validation:
            # val_log = self._valid_epoch(epoch)
            # log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (img_ids, layouts, _) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
