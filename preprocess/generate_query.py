import json
import os
import random
import copy
from pycocotools.coco import COCO

ROOT_PATH = '../data/coco'

def convert_bbox(bbox, width, height):
    min_x, min_y, w, h = bbox
    max_x = min_x + w
    max_y = min_y + h
    x, y = (min_x + max_x) / 2, (min_y + max_y) / 2
    x, y = x / width, y / height
    w, h = w / width, h / height
    return [x, y, w, h]

    
def generate_queries_from_coco(ann_file, mode='train'):
    '''
    generate queries from coco dataset

    Parameters:
    - annotation file path
    - mode: train or val or gallery
    
    Returns:
    - query list
    '''
    with open(ann_file, 'r') as f:
        tmp_ann = json.load(f)
    
    id_to_name = {category['id']: category['name'] for category in tmp_ann['categories']}
    with open(os.path.join(ROOT_PATH, 'labels.txt'), 'r') as f:
            labels = f.read().split('\n')
            name_to_id = {label: i for i, label in enumerate(labels)}
    queries = []
    coco_ann = COCO(ann_file)
    query_file_path = os.path.join(ROOT_PATH, mode, 'query.json')
    target_imgs = os.listdir(os.path.join(ROOT_PATH, mode, 'images'))
    # sample random 5000 images from target images
    target_imgs = random.sample(target_imgs, 5000)
    cnt = 0
    for img in target_imgs:
        img_id = int(img.split('.')[0])
        matching_image = coco_ann.loadImgs(img_id)[0]
        W = matching_image['width']
        H = matching_image['height']
        coco_ann.getAnnIds(imgIds=img_id)
        anns = coco_ann.loadAnns(coco_ann.getAnnIds(imgIds=img_id))
        """
        TODO :
        - exclude categories not frequently used
        - exclude categories with small area
        """
        tmp = []
        for ann in anns:
            category = id_to_name[ann['category_id']]
            bbox = convert_bbox(ann['bbox'], W, H)
            x, y, w, h = bbox

            # exclude small object regions 10% of image
            if w * h < 0.1:
                continue
            img_query = dict()
            img_query['layout'] = []
            img_query['img_id'] = img_id
            tmp.append({'label': name_to_id[category],'category': category, 'bbox': [x, y, w, h], 'area': w * h})
        if len(tmp) == 0:
            continue
        tmp = sorted(tmp, key=lambda x: x['area'], reverse=True)
        tmp = tmp[:6] if len(tmp) > 6 else tmp

        img_query = dict()
        img_query['layout'] = []
        img_query['img_id'] = img_id
        img_query['W'], img_query['H'] = W, H
        if mode != 'train':
            history = []
            for i in range(len(tmp)):
                num_elements = random.randint(1, len(tmp))
                sampled = random.sample(tmp, num_elements)
                sampled = sorted(sampled, key=lambda x: x['area'], reverse=True)
                if sampled not in history:
                    cnt+=1
                    history.append(sampled)
                    img_query['layout'] = sampled
                    queries.append(copy.deepcopy(img_query))
        
        if mode == 'train':
            img_query['layout'] = []
            for obj in tmp:
                cnt+=1
                img_query['layout'] = [obj]
                queries.append(copy.deepcopy(img_query))

    print(f'generate {cnt} queries')
    with open(query_file_path, 'w') as f:
        json.dump(queries, f, indent=4)

generate_queries_from_coco(f'{ROOT_PATH}/annotations/instances_train2017.json', mode='gallery')