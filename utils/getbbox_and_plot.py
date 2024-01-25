import cv2
import numpy as np
import matplotlib.pyplot as plt
def get_bounding_boxes(mask_path):
    """
    Get bounding boxes from mask file.

    Args:
        mask_path (str): Path to the mask file.

    Returns:
        np.ndarray: Array containing bounding box coordinates in the format (x, y, x+w, y+h).
    """
    mask = np.load(mask_path)
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
    return bounding_boxes


def plot_bounding_boxes(image_path, bounding_boxes):
    """
    Plot bounding boxes on the image.

    Args:
        image_path (str): Path to the image file.
        bounding_boxes (np.ndarray): Array containing bounding box coordinates in the format (x, y, x+w, y+h).
    """
    org_img = np.load(image_path)
    normalized_img = (org_img - np.min(org_img)) / (np.max(org_img) - np.min(org_img))*255
    normalized_img = normalized_img.astype(np.uint8)
    rgb_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)

    bbox_img = rgb_img.copy()
    for bbox in bounding_boxes:
        x, y, x_plus_w, y_plus_h = bbox
        cv2.rectangle(bbox_img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)

    plt.imshow(bbox_img)
    plt.axis('off')
    plt.show()
    return rgb_img