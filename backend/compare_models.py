import time
import torch
import detectron2
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import psutil
import GPUtil
from tqdm import tqdm
import cv2
import random
from sklearn.metrics import confusion_matrix

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo

# === Cấu hình đường dẫn mô hình ===
MODEL_OLD = "models/model_final_detect.pth"
MODEL_NEW = "models/mask_rcnn_R_101_FPN_3x/time_2/model_0069999.pth"
IMG_DIR = "annotations/data_eval"
ANN_FILE = "annotations/data_eval/_annotations.json"
DATASET_NAME = "eval_dataset"

# === Kiểm tra nếu checkpoint không tồn tại ===
if not os.path.exists(MODEL_OLD) or not os.path.exists(MODEL_NEW):
    raise FileNotFoundError("❌ Model checkpoint không tồn tại.")

# === Đăng ký dataset ===
register_coco_instances(DATASET_NAME, {}, ANN_FILE, IMG_DIR)
dataset_dicts = DatasetCatalog.get(DATASET_NAME)
metadata = MetadataCatalog.get(DATASET_NAME)
print(f"📌 Dataset '{DATASET_NAME}' đã đăng ký với {len(dataset_dicts)} ảnh.")

# === Load Model ===
def load_model(model_path, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

num_classes = len(metadata.thing_classes)
model_old = load_model(MODEL_OLD, num_classes)
model_new = load_model(MODEL_NEW, num_classes)

# === Đánh giá Precision/Recall ===
def evaluate_model(model, dataset_name, num_samples=None):
    evaluator = COCOEvaluator(dataset_name, ("bbox",), False, output_dir="./output/")
    val_loader = build_detection_test_loader(model.cfg, dataset_name)
    dataset_subset = list(val_loader)[:num_samples] if num_samples else list(val_loader)
    
    print(f"🚀 Running evaluation on {len(dataset_subset)}/{len(dataset_dicts)} images...")
    results = inference_on_dataset(model.model, dataset_subset, evaluator)
    return results

# === Ma trận nhầm lẫn ===
def compute_confusion_matrix(model, dataset_name, num_samples=None):
    dataset_subset = DatasetCatalog.get(dataset_name)[:num_samples] if num_samples else DatasetCatalog.get(dataset_name)
    gt_labels = []
    pred_labels = []

    for d in tqdm(dataset_subset, desc="Processing images for confusion matrix"):
        img = cv2.imread(d["file_name"])  # Đọc ảnh
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi màu
        img = np.asarray(img)  # Chắc chắn là numpy array

        outputs = model(img)
        
        if "instances" in outputs and len(outputs["instances"]) > 0:
            pred_classes = outputs["instances"].pred_classes.cpu().numpy().tolist()
        else:
            pred_classes = []  # Không có dự đoán nào

        gt_classes = [ann["category_id"] for ann in d.get("annotations", [])]

        # Nếu không có ground truth hoặc prediction, đặt `-1`
        if not gt_classes:
            gt_classes = [-1]

        if not pred_classes:
            pred_classes = [-1]

        # Đảm bảo hai list có cùng độ dài bằng cách lặp cho đến khi khớp
        max_len = max(len(gt_classes), len(pred_classes))

        while len(gt_classes) < max_len:
            gt_classes.append(-1)

        while len(pred_classes) < max_len:
            pred_classes.append(-1)

        # Thêm vào confusion matrix data
        gt_labels.extend(gt_classes)
        pred_labels.extend(pred_classes)

    # Thêm -1 vào danh sách nhãn để tránh lỗi "labels size mismatch"
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(num_classes)) + [-1])
    return cm

# === Đo thời gian suy luận chi tiết ===

def measure_inference_time(model, dataset_name, num_samples=None):
    dataset_subset = DatasetCatalog.get(dataset_name)[:num_samples] if num_samples else DatasetCatalog.get(dataset_name)
    times = []
    
    for d in tqdm(dataset_subset, desc="Measuring inference time"):
        img = cv2.imread(d["file_name"])  # Đọc ảnh với OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển BGR → RGB vì Detectron2 cần đúng format
        img = np.asarray(img)  # Chắc chắn ảnh ở dạng numpy array

        start_time = time.time()
        outputs = model(img)  # Chạy inference trên ảnh
        times.append(time.time() - start_time)
    
    return {
        "mean": np.mean(times),
        "min": np.min(times),
        "max": np.max(times),
        "variance": np.var(times)
    }


# === Đo bộ nhớ sử dụng ===
def get_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / (1024 ** 2)  # MB
    gpus = GPUtil.getGPUs()
    gpu_memory = sum([gpu.memoryUsed for gpu in gpus]) if gpus else 0
    return cpu_memory, gpu_memory

def save_mask_with_original_image(model_old, model_new, dataset_name, output_folder="outputs", num_samples=5):
    dataset_subset = DatasetCatalog.get(dataset_name)[:num_samples]
    
    os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục lưu ảnh nếu chưa có

    for i, d in tqdm(enumerate(dataset_subset), total=num_samples, desc="Saving mask images"):
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dự đoán từ mô hình cũ (OLD)
        outputs_old = model_old(img)
        instances_old = outputs_old["instances"].to("cpu")

        # Dự đoán từ mô hình mới (NEW)
        outputs_new = model_new(img)
        instances_new = outputs_new["instances"].to("cpu")

        # Tạo ảnh giữ nguyên nền gốc và chỉ overlay mask màu ngẫu nhiên
        def overlay_random_color_mask(image, instances, alpha=0.8):
            """
            Giữ nguyên ảnh gốc và chỉ overlay mask với màu ngẫu nhiên.
            Alpha = 0.8 (80% độ trong suốt).
            """
            overlay = image.copy()
            if len(instances) > 0:
                masks = instances.pred_masks.numpy()  # Trích xuất mask
                num_masks = masks.shape[0]

                # Tạo danh sách màu ngẫu nhiên (màu sáng để dễ phân biệt)
                colors = [
                    [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
                    for _ in range(num_masks)
                ]

                for idx, mask in enumerate(masks):
                    overlay[mask] = (
                        alpha * np.array(colors[idx]) + (1 - alpha) * overlay[mask]
                    ).astype(np.uint8)  # Áp dụng màu với độ trong suốt 80%
            return overlay

        mask_overlay_old = overlay_random_color_mask(img, instances_old, alpha=0.8)
        mask_overlay_new = overlay_random_color_mask(img, instances_new, alpha=0.8)

        # Lưu ảnh kết quả (Giữ nguyên nền gốc + Mask màu ngẫu nhiên)
        filename = os.path.basename(d["file_name"])
        old_output_path = os.path.join(output_folder, f"old_mask_overlay_{filename}")
        new_output_path = os.path.join(output_folder, f"new_mask_overlay_{filename}")

        cv2.imwrite(old_output_path, cv2.cvtColor(mask_overlay_old, cv2.COLOR_RGB2BGR))
        cv2.imwrite(new_output_path, cv2.cvtColor(mask_overlay_new, cv2.COLOR_RGB2BGR))

        print(f"📌 Đã lưu ảnh mask segmentation trên nền gốc (80% opacity): {old_output_path} & {new_output_path}")


# === Chạy đánh giá ===
n_samples = int(input("Nhập số lượng hình ảnh để phân tích (hoặc nhập -1 để phân tích tất cả): "))
n_samples = None if n_samples == -1 else n_samples

results_old = evaluate_model(model_old, DATASET_NAME, num_samples=n_samples)
results_new = evaluate_model(model_new, DATASET_NAME, num_samples=n_samples)

cm_old = compute_confusion_matrix(model_old, DATASET_NAME, num_samples=n_samples)
cm_new = compute_confusion_matrix(model_new, DATASET_NAME, num_samples=n_samples)

inference_time_old = measure_inference_time(model_old, DATASET_NAME, num_samples=n_samples)
inference_time_new = measure_inference_time(model_new, DATASET_NAME, num_samples=n_samples)

cpu_mem_old, gpu_mem_old = get_memory_usage()
cpu_mem_new, gpu_mem_new = get_memory_usage()

# save_mask_with_original_image(model_old, model_new, DATASET_NAME, output_folder="outputs_compare", num_samples=n_samples)

# === Hiển thị kết quả ===
comparison_data = {
    "Metric": ["mAP (bbox)", "Inference Time (s) - Mean", "Min Time", "Max Time", "Variance", "CPU Mem (MB)", "GPU Mem (MB)"],
    "Old Model": [results_old["bbox"]["AP"], inference_time_old["mean"], inference_time_old["min"], inference_time_old["max"], inference_time_old["variance"], cpu_mem_old, gpu_mem_old],
    "New Model": [results_new["bbox"]["AP"], inference_time_new["mean"], inference_time_new["min"], inference_time_new["max"], inference_time_new["variance"], cpu_mem_new, gpu_mem_new]
}

df_results = pd.DataFrame(comparison_data)
print(df_results)
df_results.to_csv("comparison_results_detailed.csv", index=False)

# === Vẽ confusion matrix ===
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=metadata.thing_classes, yticklabels=metadata.thing_classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(f"{title}.png")
    print(f"📌 Đã lưu {title}.png")
    plt.show()

plot_confusion_matrix(cm_old, "Confusion Matrix - Old Model")
plot_confusion_matrix(cm_new, "Confusion Matrix - New Model")

