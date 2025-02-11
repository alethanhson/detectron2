import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image
import pycocotools.mask as mask_util

def convert_detectron2_to_labelme(image_name, image_array, pred_boxes, pred_classes, pred_masks):
    """
    Chuyển đổi output từ Detectron2 sang định dạng JSON của LabelMe.

    - image_name: Tên file ảnh gốc.
    - image_array: Mảng NumPy chứa ảnh gốc.
    - pred_boxes: Danh sách bounding boxes của các đối tượng.
    - pred_classes: Danh sách class labels của các đối tượng.
    - pred_masks: Danh sách binary mask của các đối tượng.
    
    Trả về: JSON theo format LabelMe.
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
        "imagePath": image_name,
        # "imageData": image_data,
        "imageHeight": height,
        "imageWidth": width
    }

    # Xử lý từng đối tượng được dự đoán
    for i, (box, cls, mask) in enumerate(zip(pred_boxes, pred_classes, pred_masks)):
        x1, y1, x2, y2 = box
        bbox = [x1, y1, x2 - x1, y2 - y1]

        # Chuyển đổi binary mask sang danh sách tọa độ điểm ảnh (polygon)
        mask = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask)
        contours = mask_util.decode(rle).astype(np.uint8)

        # Lấy tọa độ của các pixel có giá trị > 0 (tức là thuộc vùng đối tượng)
        points = np.column_stack(np.where(contours > 0))
        points = points[:, ::-1].tolist()  # Đổi từ (y, x) sang (x, y)

        # Nếu không có điểm nào, bỏ qua object này
        if not points:
            continue

        shape_data = {
            "label": str(cls),  # Lưu nhãn theo class index
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        }

        labelme_json["shapes"].append(shape_data)

    return labelme_json
