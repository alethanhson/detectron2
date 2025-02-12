import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import logging
import cv2
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "models/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.DEVICE = "cpu"
    return cfg

cfg = setup_detectron2()
predictor = DefaultPredictor(cfg)

def convert_mask_to_pixel_points(pred_masks, epsilon=0.01, min_area=200):
    """
    Làm mịn và giảm tối đa số điểm trong polygon khi chuyển đổi mask từ Detectron2 sang Labelme.

    Args:
        pred_masks (numpy.array): Mảng mask đầu ra từ mô hình.
        epsilon (float): Độ làm mượt (Giá trị càng lớn, số điểm càng ít).
        min_area (int): Diện tích nhỏ nhất của polygon hợp lệ.

    Returns:
        List[dict]: Danh sách polygon với điểm pixel (x, y).
    """
    pixel_polygons = []
    
    for mask in pred_masks:
        # Áp dụng GaussianBlur mạnh hơn để làm mịn răng cưa
        mask = cv2.GaussianBlur(mask.astype(np.uint8), (5, 5), 0)

        # Dùng morphology để loại bỏ nhiễu nhỏ
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Tìm contour, dùng CHAIN_APPROX_SIMPLE để giảm số điểm thừa
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Làm mịn và giảm số điểm xuống tối thiểu
            smoothed_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

            # Chuyển contour thành danh sách điểm pixel (x, y)
            pixel_polygon = [(int(point[0][0]), int(point[0][1])) for point in smoothed_contour]

            # Kiểm tra polygon có hợp lệ không
            if len(pixel_polygon) < 3:
                continue  # Bỏ qua polygon không đủ điểm

            shapely_polygon = Polygon(pixel_polygon)
            if not shapely_polygon.is_valid or shapely_polygon.area < min_area:
                continue  # Bỏ qua polygon nhỏ hoặc không hợp lệ

            pixel_polygons.append({
                "label": "0",  
                "points": pixel_polygon  
            })

    return pixel_polygons

def convert_detectron2_to_labelme(image_name, image_array, pred_masks):
    """
    Chuyển đổi đầu ra của Detectron2 sang JSON định dạng Labelme.

    Args:
        image_name (str): Tên của hình ảnh.
        image_array (numpy.array): Mảng hình ảnh đầu vào.
        pred_masks (numpy.array): Mảng mask nhị phân.

    Returns:
        dict: JSON chứa dữ liệu dạng Labelme.
    """
    height, width = image_array.shape[:2]

    # Chuyển ảnh sang base64 để nhúng vào JSON
    pil_image = Image.fromarray(image_array)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Khởi tạo cấu trúc JSON của LabelMe
    labelme_json = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imageData": image_data,
        "imagePath": image_name,
        "imageHeight": int(height),
        "imageWidth": int(width)
    }

    # Chuyển đổi mask thành pixel points với làm mịn
    pixel_polygons = convert_mask_to_pixel_points(pred_masks)

    for obj in pixel_polygons:
        shape_data = {
            "label": obj["label"],
            "points": obj["points"],
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        }
        labelme_json["shapes"].append(shape_data)

    return labelme_json

# Load image
image_path = "image_54_1724652404.jpg"
image = cv2.imread(image_path)

if image is None:
    logger.error(f"Could not read image: {image_path}")
else:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    outputs = predictor(image)
    
    pred_masks = outputs["instances"].pred_masks.to("cpu").numpy()  # Extract masks

    labelme_json = convert_detectron2_to_labelme(image_path, img_rgb, pred_masks)

    json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_path)[0]}.json")
    with open(json_path, "w") as f:
        json.dump(labelme_json, f, indent=4)

    print(f"Labelme JSON saved to: {json_path}")
