import torch
import numpy as np
import math
import fasttext, fasttext.util
import clip
import copy

def embedding_fn(mode='one-hot'):
    func = None
    if mode == 'one-hot':
        func = one_hot_embedding
    elif mode == 'word2vec':
        fasttext.util.download_model('en', if_exists='ignore')
        word2vec = fasttext.load_model('cc.en.300.bin')
        func = word2vec.get_word_vector
    else:
        raise ValueError('Invalid embedding mode')
    return func

def clip_embedding_fn(device='cuda'):
    model, preprocess = clip.load("RN50")
    def func(text):
        text = clip.tokenize(text)
        text = text.to(device)
        text_features = model.encode_text(text)
        return text_features
    return func

def coco_clip_embedding_fn():
    class_embeddings = np.load('./data/coco/coco_embeddings_rn50.npy', allow_pickle=True).item()
    def func(text):
        return class_embeddings[text]
    return func

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

def convert_query_to_tensor(queries, num_dim, mode='one-hot', embedding_fn=None):
    """
    convert query to tensor
    """
    width, height = 31, 31
    query_tensor_list = []
    for obj in queries:
        query_tensor = torch.zeros((width, height, num_dim), dtype=torch.float32)
        if mode == 'one-hot':
            cat_embedding = embedding_fn(obj['label'], num_dim)
        elif mode == 'word2vec':
            cat_embedding = torch.tensor(embedding_fn(obj['category']))
        elif mode == 'clip':
            cat_embedding = torch.tensor(embedding_fn(obj['category']))[0]
        elif mode == 'coco_clip':
            cat_embedding = torch.tensor(embedding_fn(obj['category']))[0]
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
        query_tensor_list.append(query_tensor)
    query_tensor_list = torch.stack(query_tensor_list)
    # rescaled_tensor = F.interpolate(query_tensor.permute(2, 0, 1).unsqueeze(0), size=(31, 31), mode='bilinear', align_corners=False).squeeze(0)
    return query_tensor_list # [num_obj, num_dim, width, height]