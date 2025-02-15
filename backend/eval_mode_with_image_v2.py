import os
import cv2
import numpy as np
from tqdm import tqdm

def compute_statistics_from_folder(model, input_folder, score_thresh=0.5):
    """
    Tính toán thống kê từ inference trên tất cả các ảnh trong một folder.
    
    Args:
        model: Một đối tượng model (ví dụ: DefaultPredictor) đã được thiết lập để chạy inference.
        input_folder: Đường dẫn tới folder chứa ảnh đầu vào.
        score_thresh: Ngưỡng confidence (đã được set trong config của model).
        
    Trả về một dictionary chứa:
        - total_images: Số ảnh được xử lý.
        - detection_rate: Tỷ lệ ảnh có dự đoán (ít nhất 1 đối tượng).
        - average_solidity: Giá trị solidity trung bình của tất cả các mask dự đoán.
    """
    # Lấy danh sách file ảnh từ folder (theo đuôi ảnh phổ biến)
    image_files = [os.path.join(input_folder, f) 
                   for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = len(image_files)
    if total_images == 0:
        print("Không tìm thấy ảnh trong folder!")
        return None

    detected_images = 0  # số ảnh có ít nhất 1 dự đoán
    solidity_list = []   # danh sách solidity của các mask dự đoán

    for image_path in tqdm(image_files, desc="Processing images"):
        img = cv2.imread(image_path)
        if img is None:
            continue
        # Chuyển đổi ảnh từ BGR sang RGB (model của Detectron2 nhận RGB)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Chạy inference với model
        outputs = model(image_rgb)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) > 0:
            detected_images += 1
            # Nếu có mask dự đoán, tính solidity cho từng mask
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()  # shape: (N, H, W) kiểu boolean
                for mask in masks:
                    area = np.sum(mask)  # số pixel của mask
                    # Chuyển mask về dạng uint8 để sử dụng cv2.findContours
                    mask_uint8 = (mask.astype(np.uint8)) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Lấy contour có diện tích lớn nhất
                        largest_contour = max(contours, key=cv2.contourArea)
                        convex_hull = cv2.convexHull(largest_contour)
                        convex_area = cv2.contourArea(convex_hull)
                        if convex_area > 0:
                            solidity = area / convex_area
                            solidity_list.append(solidity)
    
    detection_rate = detected_images / total_images
    average_solidity = np.mean(solidity_list) if solidity_list else 0.0

    stats = {
        "total_images": total_images,
        "detection_rate": detection_rate,
        "average_solidity": average_solidity
    }
    return stats

# --- Ví dụ sử dụng:
# Giả sử bạn đã tạo đối tượng DefaultPredictor cho mô hình của mình
# Ví dụ, cấu hình model instance segmentation đã được thiết lập:
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = "models/mask_rcnn_R_101_FPN_3x/time_2/model_final.pth"  # Cập nhật đường dẫn checkpoint của bạn
cfg.MODEL.WEIGHTS = "models/model_final_detect.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.MASK_ON = True
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
predictor = DefaultPredictor(cfg)

# Đường dẫn tới folder chứa ảnh test
input_folder = "images"  # Thay đổi đường dẫn folder của bạn

stats = compute_statistics_from_folder(predictor, input_folder, score_thresh=0.5)
if stats:
    print("=== Thống kê kết quả ===")
    print(f"Số lượng ảnh phân tích: {stats['total_images']}")
    print(f"Tỷ lệ nhận diện đối tượng: {stats['detection_rate']*100:.2f}%")
    print(f"Tỷ lệ bo viền (average solidity): {stats['average_solidity']:.4f}")
