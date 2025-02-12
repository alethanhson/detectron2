import os
import cv2
import json
import base64
import logging
from shapely.geometry import Polygon
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import shutil

# Cấu hình logging
logging.basicConfig(
    filename="process.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

error_logger = logging.getLogger("error")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler("error.log")
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
error_logger.addHandler(error_handler)

# Thư mục chứa ảnh gốc và thư mục lưu JSON đầu ra
DATA_DIR = "images"
OUTPUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_detectron2():
    """Cấu hình Detectron2"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "models/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.DEVICE = "cpu"
    return cfg

cfg = setup_detectron2()
predictor = DefaultPredictor(cfg)

def encode_image_to_base64(image_path):
    """Chuyển đổi hình ảnh thành base64 để lưu vào JSON."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        error_logger.error(f"Lỗi mã hóa base64 {image_path}: {e}")
        return None

def convert_mask_to_pixel_points(pred_masks, pred_classes, epsilon=0.01, min_area=200):
    """
    Chuyển đổi mask Detectron2 thành điểm polygon và lấy nhãn AI.

    Args:
        pred_masks (numpy.array): Danh sách mask từ mô hình.
        pred_classes (numpy.array): Danh sách lớp dự đoán của mô hình.

    Returns:
        List[dict]: Danh sách các đối tượng với label và tọa độ polygon.
    """
    pixel_polygons = []
    try:
        for mask, class_idx in zip(pred_masks, pred_classes):
            contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                smoothed_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
                pixel_polygon = [(int(point[0][0]), int(point[0][1])) for point in smoothed_contour]
                if len(pixel_polygon) < 3:
                    continue
                shapely_polygon = Polygon(pixel_polygon)
                if shapely_polygon.area < min_area:
                    continue

                pixel_polygons.append({
                    "label": str(class_idx),
                    "points": pixel_polygon
                })

    except Exception as e:
        error_logger.error(f"Lỗi xử lý polygon: {e}")

    return pixel_polygons

def process_images():
    """
    Chạy Detectron2 trên tất cả ảnh trong `images/`, tạo dữ liệu LabelMe JSON cùng thư mục với ảnh.
    """
    for image_name in os.listdir(DATA_DIR):
        try:
            if not image_name.endswith((".jpg", ".png")):
                continue

            image_path = os.path.join(DATA_DIR, image_name)
            image_basename = os.path.splitext(image_name)[0]
            json_path = os.path.join(OUTPUT_DIR, f"{image_basename}.json")

            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Lỗi đọc ảnh {image_name}")

            height, width = image.shape[:2]

            # Chuyển đổi ảnh sang base64
            image_data_base64 = encode_image_to_base64(image_path)
            if image_data_base64 is None:
                raise ValueError(f"Lỗi base64 {image_name}")

            # Chạy mô hình Detectron2
            outputs = predictor(image)

            # Lấy masks & classes từ mô hình AI
            pred_masks = outputs["instances"].pred_masks.to("cpu").numpy()
            pred_classes = outputs["instances"].pred_classes.to("cpu").numpy()

            # Tạo JSON theo format của LabelMe
            labelme_json = {
                "version": "5.2.1",
                "shapes": convert_mask_to_pixel_points(pred_masks, pred_classes),
                "imagePath": image_name,
                "imageData": image_data_base64,
                "imageHeight": int(height),
                "imageWidth": int(width),
            }

            # Lưu JSON vào thư mục đầu ra
            with open(json_path, "w") as f:
                json.dump(labelme_json, f, indent=4)

            # Di chuyển ảnh sang thư mục `outputs/` bằng `shutil.move()`
            new_image_path = os.path.join(OUTPUT_DIR, image_name)
            shutil.move(image_path, new_image_path)

            logging.info(f"✔ Xử lý xong: {image_name} → {json_path}")
            print(f"✔ Xử lý xong: {image_name} → {json_path}")

        except Exception as e:
            error_logger.error(f"Lỗi xử lý ảnh {image_name}: {e}")
            print(f"❌ Lỗi xử lý ảnh {image_name}, xem error.log để biết chi tiết")