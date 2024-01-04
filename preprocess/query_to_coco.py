import json
import os

category_name_to_id = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14,
        'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31,
        'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
        'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56,
        'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
        'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87,
        'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90, 'banner': 92, 'blanket': 93, 'branch': 94, 'bridge': 95, 'building-other': 96, 'bush': 97, 'cabinet': 98, 'cage': 99, 'cardboard': 100,
        'carpet': 101, 'ceiling-other': 102, 'ceiling-tile': 103, 'cloth': 104, 'clothes': 105, 'clouds': 106, 'counter': 107, 'cupboard': 108, 'curtain': 109, 'desk-stuff': 110, 'dirt': 111,
        'door-stuff': 112, 'fence': 113, 'floor-marble': 114, 'floor-other': 115, 'floor-stone': 116, 'floor-tile': 117, 'floor-wood': 118, 'flower': 119, 'fog': 120, 'food-other': 121, 'fruit': 122,
        'furniture-other': 123, 'grass': 124, 'gravel': 125, 'ground-other': 126, 'hill': 127, 'house': 128, 'leaves': 129, 'light': 130, 'mat': 131, 'metal': 132, 'mirror-stuff': 133, 'moss': 134,
        'mountain': 135, 'mud': 136, 'napkin': 137, 'net': 138, 'paper': 139, 'pavement': 140, 'pillow': 141, 'plant-other': 142, 'plastic': 143, 'platform': 144, 'playingfield': 145, 'railing': 146,
        'railroad': 147, 'river': 148, 'road': 149, 'rock': 150, 'roof': 151, 'rug': 152, 'salad': 153, 'sand': 154, 'sea': 155, 'shelf': 156, 'sky-other': 157, 'skyscraper': 158, 'snow': 159,
        'solid-other': 160, 'stairs': 161, 'stone': 162, 'straw': 163, 'structural-other': 164, 'table': 165, 'tent': 166, 'textile-other': 167, 'towel': 168, 'tree': 169, 'vegetable': 170,
        'wall-brick': 171, 'wall-concrete': 172, 'wall-other': 173, 'wall-panel': 174, 'wall-stone': 175, 'wall-tile': 176, 'wall-wood': 177, 'water-other': 178, 'waterdrops': 179,
        'window-blind': 180, 'window-other': 181, 'wood': 182, 'other': 183, '__image__': 0, '__null__': 184}

def convert_bbox_to_coco_format(bbox, width, height):
    center_x, center_y, bbox_w, bbox_h = bbox
    min_x = (center_x - bbox_w / 2) * width
    min_y = (center_y - bbox_h / 2) * height
    width = bbox_w * width
    height = bbox_h * height
    return [min_x, min_y, width, height]

# Prepare COCO formatted data
coco_annotations = []
coco_images = []

root_path = '../data/coco'
query_path = os.path.join(root_path, 'gallery', 'tmp.json')
input_data = json.load(open(query_path, 'r'))
for i, item in enumerate(input_data):
    image_id = item["img_id"]
    query_id = i
    width = item["W"]
    height = item["H"]
    for obj in item["layout"]:
        category_id = category_name_to_id[obj["category"]]
        bbox = convert_bbox_to_coco_format(obj["bbox"], width, height)
        area = obj["area"] * width * height
        annotation = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area
        }
        coco_annotations.append(annotation)

    coco_images.append({
        "id": image_id,
        "query_id": i,
        "file_name": '{0:012d}.jpg'.format(image_id),  # e.g. '000000000001.jpg
        "width": width,
        "height": height
    })

# Final COCO formatted dictionary
coco_format = {
    "annotations": coco_annotations,
    "images": coco_images
    # "categories": []  # This should be a list of all your categories
}

# Convert to JSON
coco_json = json.dumps(coco_format, indent=4)

# Output to a file
with open('coco_annotations.json', 'w') as f:
    f.write(coco_json)

print("COCO annotations have been written to 'coco_annotations.json'")
