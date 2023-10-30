import argparse
import torch
import numpy as np
import fasttext, fasttext.util

from sklearn.neighbors import NearestNeighbors
import faiss
from tqdm import tqdm

from data_loader.data_loaders import CocoDataset, collate_fn
from model.model import *
from parse_config import ConfigParser
from utils import prepare_device, convert_query_to_tensor, create_bbox_mask, one_hot_embedding, create_bbox_mask 
from utils import BatchKNearestNeighbor
from utils import is_relevant, AP_at_k, ndcg_at_k, compute_rel_score, coco_clip_embedding_fn, clip_embedding_fn

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import data_loader.data_loaders as module_data
import time
from pycocotools.coco import COCO



def main(config):
    logger = config.get_logger('test')

    batch_size = config['data_loader']['args']['batch_size']
    batch_size = 4

    eval_set = CocoDataset(root='./data/coco', mode='gallery')
    data_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size,shuffle=False, num_workers=4, collate_fn=collate_fn)
    eval_features = torch.load('./data/coco/gallery/features/resnet50_features.pt')
    
    # to numpy
    img_id_list = np.array(list(eval_features.keys()))
    feature_list = list(eval_features.values())
    feature_list = torch.stack(feature_list)

    embed_dim = config['arch']['args']['embed_dim']
    embed_mode = config['arch']['args']['embed_mode']
    
    # build model architecture, then print to console
    model = FeatureSynthesisModel(embed_dim=embed_dim)
    backbone_model = ResNet50()
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    logger.info(f'masking : {config["masking"]}, normalize : {config["normalize"]}')
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if embed_mode == 'one-hot':
        embed_fn = one_hot_embedding
    elif embed_mode == 'word2vec':
        fasttext.util.download_model('en', if_exists='ignore')
        word2vec = fasttext.load_model('cc.en.300.bin')
        embed_fn = word2vec.get_word_vector
    elif embed_mode == 'clip':
        embed_fn = clip_embedding_fn(device=device)
    elif embed_mode == 'coco_clip':
        embed_fn = coco_clip_embedding_fn()
    else:
        raise ValueError('Invalid embedding mode')

    # testing 
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    all_APs_1 = []
    all_APs_10 = []
    all_APs_50 = []

    ndcg_1s = []
    ndcg_10s = []
    ndcg_50s = []

    coco = COCO('./data/coco/annotations/instances_train2017.json')
    nbrs = BatchKNearestNeighbor(batch_size=5000, device=device, mask=config["masking"])
    nbrs.fit(feature_list.view(feature_list.shape[0], -1).cpu().numpy())
    with torch.no_grad():
        for i, (img_ids, layouts, sizes) in enumerate(tqdm(data_loader)):
            batch_size = len(img_ids)
            all_indices = []
            all_output_features = []
            for j in range(batch_size):
                canvas_queries = convert_query_to_tensor(layouts[j], embed_dim, mode=embed_mode, embedding_fn=embed_fn) # [num_obj, num_dim, width, height]
                canvas_queries = canvas_queries.to(device)
                output_feature = model(canvas_queries) # [num_obj, 2048, 7, 7]
                masks = torch.stack([create_bbox_mask(layouts[j][k]['bbox'], 7) for k in range(len(layouts[j]))]).unsqueeze(1).to(device)
                # normalize
                if config['normalize']:
                    output_feature = output_feature / (torch.norm(output_feature, dim=1, keepdim=True) + 1e-10)
                output_feature = output_feature * masks
                # max pooling
                output_feature, _ = torch.max(output_feature, dim=0)
                output_feature = output_feature.view(1, -1)
                all_output_features.append(output_feature.cpu().numpy())
            all_output_features = np.concatenate(all_output_features, axis=0)

            start_time = time.time()
            indices, distance = nbrs.predict(all_output_features, k=50)
            print(f'knn time for {i*batch_size + j}: {time.time() - start_time}')

            # all_output_features = np.concatenate(all_output_features, axis=0)
            
            for j in range(batch_size):    
                ranked_list = img_id_list[indices[j]]
                relevance_scores = [compute_rel_score(layouts[j], ranked_list[k], coco) for k in range(len(ranked_list))]

                AP_1 = AP_at_k(ranked_list, layouts[j], 1, coco)
                AP_10 = AP_at_k(ranked_list, layouts[j], 10, coco)
                AP_50 = AP_at_k(ranked_list, layouts[j], 50, coco)
                all_APs_1.append(AP_1)
                all_APs_10.append(AP_10)
                all_APs_50.append(AP_50)
            
                ndcg_1 = ndcg_at_k(relevance_scores, 1)
                ndcg_10 = ndcg_at_k(relevance_scores, 10)
                ndcg_50 = ndcg_at_k(relevance_scores, 50)
                ndcg_1s.append(ndcg_1)
                ndcg_10s.append(ndcg_10)
                ndcg_50s.append(ndcg_50)

                print(f'img: {img_ids[j]}, num obj: {len(layouts[j])}, AP@1: {AP_1}, AP@10: {AP_10}, ndcg@1: {ndcg_1}, ndcg@10: {ndcg_10}')
                
                

    mAP_1 = sum(all_APs_1) / len(all_APs_1)
    mAP_10 = sum(all_APs_10) / len(all_APs_10)
    mAP_50 = sum(all_APs_50) / len(all_APs_50)

    ndcg_1 = sum(ndcg_1s) / len(ndcg_1s)
    ndcg_10 = sum(ndcg_10s) / len(ndcg_10s)
    ndcg_50 = sum(ndcg_50s) / len(ndcg_50s)

            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    
    log = {'loss': total_loss / n_samples}
    log.update({
        'mAP@1': mAP_1,
        'mAP@10': mAP_10,
        'mAP@50': mAP_50,
        'ndcg@1': ndcg_1,
        'ndcg@10': ndcg_10,
        'ndcg@50': ndcg_50
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default='./saved/models/word2vec/0831_154110/checkpoint-epoch28.pth', type=str,
    #                   help='path to latest checkpoint (default: None)')
    args.add_argument('-r', '--resume', default='./saved/models/clip_template/1016_200332/checkpoint-epoch50.pth', type=str)
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
