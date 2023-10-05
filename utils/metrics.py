from pycocotools.coco import COCO
import math
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def AP_at_k(ranked_list, layouts, k, coco):
    # Limit the ranked list to top-K items
    ranked_list = ranked_list[:k]
    
    relevant_positions = [idx for idx, img in enumerate(ranked_list) if is_relevant(layouts, img, coco=coco)]
    precisions = [sum(1 for pos in relevant_positions if pos <= i) / (i+1) for i in relevant_positions]
    
    AP = sum(precisions) / len(relevant_positions) if len(relevant_positions) > 0 else 0
    return AP

def is_relevant(query_layouts, ranked_img_id, coco):
    img = coco.loadImgs([ranked_img_id])[0]
    ann = coco.loadAnns(coco.getAnnIds(imgIds=[ranked_img_id]))
    ranked_img_layouts = [ann[i] for i in range(len(ann))]
    H, W = img['height'], img['width']
    num_bounding_boxes = len(query_layouts)
    rel_score = 0
    for query_layout in query_layouts:
        # calculate miou
        x1, y1, w1, h1 = query_layout['bbox']
        x1, y1, w1, h1 = x1 * W, y1 * H, w1 * W, h1 * H
        min_x1, min_y1, max_x1, max_y1 = int(x1 - w1 / 2), int(y1 - h1 / 2), math.ceil(x1 + w1 / 2), math.ceil(y1 + h1 / 2)
        query_cat = query_layout['category']
        max_iou = 0
        for ranked_img_layout in ranked_img_layouts:
            if coco.getCatIds(catNms=[query_cat])[0] != ranked_img_layout['category_id']:
                continue
            x2, y2, w2, h2 = ranked_img_layout['bbox']
            min_x2, min_y2, max_x2, max_y2 = int(x2), int(y2), math.ceil(x2 + w2), math.ceil(y2 + h2)
            
            # compute miou
            min_x = max(min_x1, min_x2)
            min_y = max(min_y1, min_y2)
            max_x = min(max_x1, max_x2)
            max_y = min(max_y1, max_y2)
            intersection = max(0, max_x - min_x) * max(0, max_y - min_y)
            union = w1 * h1 + w2 * h2 - intersection
            miou = intersection / union
            max_iou = max(max_iou, miou)
        rel_score += max_iou
    rel_score /= num_bounding_boxes
    return rel_score > 0.3

            