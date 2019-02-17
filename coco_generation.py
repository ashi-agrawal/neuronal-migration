import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
import pycococreator

root = Path('Desktop/Annotated_Neuron_Nuclei')
IMAGE_DIR = root / 'Images'
MASK_DIR = root / 'Masks'

# filter for jpeg images
coco_output = {
    "images": [],
    "annotations": []
}
image_id = 0
segmentation_id = 0
tolerance = 2

for root, _, files in os.walk(IMAGE_DIR):
    image_files = files
    # go through each image
    for image_filename in image_files:
        image = Image.open(IMAGE_DIR/image_filename)
        image_info = create_image_info(image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        
        # filter for associated png annotations
        for root, _, files2 in os.walk(MASK_DIR):
            annotation_files = files2
            
            # go through each associated annotation
            for annotation_filename in annotation_files:
                category_info = {'id':0,'is_crowd': 0}
                binary_mask = np.asarray(Image.open(MASK_DIR/annotation_filename))
                annotation_info = create_annotation_info(segmentation_id, image_id, category_info, binary_mask, image.size, tolerance)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1
    image_id = image_id + 1
print(coco_output["annotations"])

data = {}
data['annotations'] = coco_output["annotations"]

with open('coco_output_annotations.txt', 'w') as outfile:
    json.dump(data, outfile)
