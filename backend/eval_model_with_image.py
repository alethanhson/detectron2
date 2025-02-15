import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def save_mask_with_original_image_from_folder(model_old, model_new, input_folder, output_folder="outputs", num_samples=5):
    """
    Chạy inference trên một số ảnh trong input_folder và lưu kết quả mask overlay từ model_old và model_new.
    
    Args:
        model_old: Đối tượng model (ví dụ: DefaultPredictor) của model cũ.
        model_new: Đối tượng model của model mới.
        input_folder: Đường dẫn tới folder chứa các ảnh đầu vào.
        output_folder: Folder lưu kết quả (sẽ xóa và tạo mới nếu đã tồn tại).
        num_samples: Số lượng mẫu ảnh đầu tiên cần xử lý.
    """
    # Lấy danh sách file ảnh từ folder (theo đuôi ảnh phổ biến)
    image_files = [os.path.join(input_folder, f) 
                   for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files = image_files[:num_samples]
    
    # Xóa folder output nếu đã tồn tại, sau đó tạo lại
    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    def overlay_random_color_mask(image, instances, alpha=0.8):
        """
        Overlay mask với màu ngẫu nhiên lên ảnh gốc.
        Args:
            image: Ảnh gốc (RGB).
            instances: Đối tượng instances từ output của model, chứa pred_masks.
            alpha: Độ trong suốt của overlay (0.8 nghĩa là 80% màu overlay).
        Returns:
            overlay: Ảnh sau khi overlay mask.
        """
        overlay = image.copy()
        # Kiểm tra nếu có mask dự đoán
        if len(instances) > 0 and instances.has("pred_masks"):
            masks = instances.pred_masks.numpy()  # shape: (num_masks, H, W) kiểu boolean
            num_masks = masks.shape[0]
            # Tạo danh sách màu ngẫu nhiên (sáng, để dễ phân biệt)
            colors = [
                [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
                for _ in range(num_masks)
            ]
            for idx, mask in enumerate(masks):
                # Với mỗi mask, overlay màu lên những pixel tương ứng
                overlay[mask] = (alpha * np.array(colors[idx]) + (1 - alpha) * overlay[mask]).astype(np.uint8)
        return overlay

    for image_path in tqdm(image_files, desc="Processing images"):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể load ảnh: {image_path}")
            continue
        # Chuyển đổi từ BGR (OpenCV) sang RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Dự đoán từ model cũ
        outputs_old = model_old(image_rgb)
        instances_old = outputs_old["instances"].to("cpu")
        mask_overlay_old = overlay_random_color_mask(image_rgb, instances_old, alpha=0.8)
        
        # Dự đoán từ model mới
        outputs_new = model_new(image_rgb)
        instances_new = outputs_new["instances"].to("cpu")
        mask_overlay_new = overlay_random_color_mask(image_rgb, instances_new, alpha=0.8)
        
        # Lấy tên file ảnh gốc
        filename = os.path.basename(image_path)
        old_output_path = os.path.join(output_folder, f"old_mask_overlay_{filename}")
        new_output_path = os.path.join(output_folder, f"new_mask_overlay_{filename}")
        
        # Lưu ảnh kết quả (cv2.imwrite yêu cầu ảnh ở định dạng BGR)
        cv2.imwrite(old_output_path, cv2.cvtColor(mask_overlay_old, cv2.COLOR_RGB2BGR))
        cv2.imwrite(new_output_path, cv2.cvtColor(mask_overlay_new, cv2.COLOR_RGB2BGR))
        
        print(f"📌 Đã lưu ảnh mask segmentation: {old_output_path} & {new_output_path}")

# --- Ví dụ sử dụng:
# Giả sử bạn đã cấu hình model_old và model_new sử dụng DefaultPredictor với config instance segmentation.

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Cấu hình cho model cũ
cfg_old = get_cfg()
cfg_old.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg_old.MODEL.WEIGHTS = "models/model_final_detect.pth"  # Cập nhật checkpoint model cũ
cfg_old.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_old.MODEL.MASK_ON = True
cfg_old.MODEL.ROI_HEADS.NUM_CLASSES = 4

model_old = DefaultPredictor(cfg_old)

# Cấu hình cho model mới
cfg_new = get_cfg()
cfg_new.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg_new.MODEL.WEIGHTS = "models/mask_rcnn_R_101_FPN_3x/time_2/model_final.pth"  # Cập nhật checkpoint model mới
cfg_new.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_new.MODEL.MASK_ON = True
cfg_new.MODEL.ROI_HEADS.NUM_CLASSES = 4
model_new = DefaultPredictor(cfg_new)

# Đường dẫn tới folder chứa ảnh test (không cần annotation COCO)
input_folder = "images"  # Folder chứa ảnh đầu vào
output_folder = "image_detect_test"  # Folder lưu kết quả

# Chạy hàm để lưu kết quả overlay mask từ cả hai model
save_mask_with_original_image_from_folder(model_old, model_new, input_folder, output_folder, num_samples=100)
