import json
import copy

ANNOTATIONS_FILE_PATH = 'data/lvis/annotations/lvis_v1_train+coco_mask_v1.json'
ANNOTATIONS_FILE_PATH_AGNOSTIC = 'data/lvis/annotations/lvis_v1_train+coco_mask_v1_base.json'

with open(ANNOTATIONS_FILE_PATH, 'r') as f:
    annotations = json.load(f)

catid2freq = copy.deepcopy({x['id']: x['frequency'] for x in annotations['categories']})

new_categories = list()
oldid2newid, cat_idx = dict(), 1
for cat in annotations['categories']:
    if cat['frequency'] == 'r':
        continue
    oldid2newid[cat['id']] = cat_idx
    cat['id'] = cat_idx
    new_categories.append(cat)
    cat_idx += 1

new_annotations = list()
for ann in annotations['annotations']:
    if catid2freq[ann['category_id']] == 'r':
        continue
    ann['category_id'] = oldid2newid[ann['category_id']]
    new_annotations.append(ann)


new_images = list()
for img in annotations['images']:
    new_not_exhaustive_category_ids = list()
    for cat_idx in img['not_exhaustive_category_ids']:
        if catid2freq[cat_idx] != 'r':
            new_not_exhaustive_category_ids.append(cat_idx)
    img['not_exhaustive_category_ids'] = new_not_exhaustive_category_ids
    new_images.append(img)

annotations['annotations'] = new_annotations
annotations['categories'] = new_categories
annotations['images'] = new_images
with open(ANNOTATIONS_FILE_PATH_AGNOSTIC, 'w') as f:
    json.dump(annotations, f)

