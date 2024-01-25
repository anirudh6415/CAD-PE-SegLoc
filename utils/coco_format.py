import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
def get_bounding_boxes(mask_path):
    mask = mask_path
    x = np.zeros_like(mask)
    x[mask > 0] = 1
    x = x.astype(np.float32)

    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[x > 0] = (255, 255, 255)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bounding_boxes.append((x, y, x + w, y + h))
    bounding_boxes = np.array(bounding_boxes)
    return bounding_boxes, contours

def coco_format(image_file_names, mask_file_names):
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    for i, image_file_name in enumerate(image_file_names):
        image = np.load(image_file_name)
        #print(image.dtype)
        image_info = {
            "id": i + 1,
            "file_name": os.path.basename(image_file_name),
            "height": image.shape[0],
            "width": image.shape[1]
        }
        coco_dataset["images"].append(image_info)

        mask = np.load(mask_file_names[i])
        #print(mask.dtype)
        bounding_box_list, contour_points_list = get_bounding_boxes(mask)
        for j in range(len(contour_points_list)):
            
            contour_points = contour_points_list[j]
            #print(contour_points.dtype)
            bounding_box = bounding_box_list[j]
            #print(bounding_box.dtype)

            annotation_info = {
                "id": len(coco_dataset["annotations"]) + 1,
                "image_id": i + 1,
                "category_id": 1,  # Replace with the category ID for your object
                "segmentation": [contour_points.flatten().tolist()],
                "area": int(np.sum(mask[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]])),
                "bbox": bounding_box.tolist(),
                "iscrowd": 0
            }
            # x= int(np.sum(mask[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]))
            # print(x)

            coco_dataset["annotations"].append(annotation_info)

    category_info = {
        "id": 1,  # Replace with the category ID for your object
        "name": "PE"  # Replace with the name of your object
    }
    coco_dataset["categories"].append(category_info)
    #print(coco_dataset["annotations"])
#     # Convert uint64 to int
#     coco_dataset["images"] = [img.astype(int) for img in coco_dataset["images"]]
#     coco_dataset["annotations"] = [ann.astype(int) for ann in coco_dataset["annotations"]]

    with open("coco_dataset.json", "w") as f:
        json.dump(coco_dataset, f)