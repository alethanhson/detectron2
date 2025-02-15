import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    DatasetEvaluators,
    DatasetEvaluator
)

# --- Định nghĩa Evaluator tùy chỉnh: Counter ---
class Counter(DatasetEvaluator):
    def reset(self):
        self.count = 0

    def process(self, inputs, outputs):
        # Giả sử output chứa key "instances"
        for output in outputs:
            if "instances" in output:
                self.count += len(output["instances"])

    def evaluate(self):
        return {"detected_instances_count": self.count}

# --- Cấu hình đường dẫn ---
MODEL_OLD = "models/model_final_detect.pth"
MODEL_NEW = "models/mask_rcnn_R_101_FPN_3x/time_2/model_0069999.pth"
IMG_DIR = "annotations/data_test/data_coco_summary_test_1"
ANN_FILE = "annotations/data_test/data_coco_summary_test_1/_annotations.json"
DATASET_NAME = "eval_dataset"

# --- Kiểm tra nếu checkpoint không tồn tại ---
if not os.path.exists(MODEL_OLD) or not os.path.exists(MODEL_NEW):
    raise FileNotFoundError("❌ Model checkpoint không tồn tại.")

# --- Đăng ký dataset test theo định dạng COCO ---
register_coco_instances(DATASET_NAME, {}, ANN_FILE, IMG_DIR)
dataset_dicts = DatasetCatalog.get(DATASET_NAME)
metadata = MetadataCatalog.get(DATASET_NAME)
print(f"📌 Dataset '{DATASET_NAME}' đã đăng ký với {len(dataset_dicts)} ảnh.")

# --- Cấu hình model ---
cfg = get_cfg()
# Bạn có thể merge config từ file cấu hình của Detectron2 (ví dụ: từ model zoo)
# cfg.merge_from_file("path/to/config.yaml")
cfg.MODEL.WEIGHTS = MODEL_NEW
cfg.DATASETS.TEST = (DATASET_NAME,)
# Nếu cần, cập nhật thêm các tham số như kích thước ảnh đầu vào
cfg.INPUT.MIN_SIZE_TEST = 256
cfg.INPUT.MAX_SIZE_TEST = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

# --- Xây dựng model và load checkpoint ---
model = build_model(cfg)
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
model.eval()

# --- Xây dựng data loader cho dataset test ---
data_loader = build_detection_test_loader(cfg, DATASET_NAME)

# --- Khởi tạo evaluator: sử dụng COCOEvaluator và custom Counter evaluator ---
output_dir = os.path.join(cfg.OUTPUT_DIR, "inference", DATASET_NAME)
os.makedirs(output_dir, exist_ok=True)
coco_evaluator = COCOEvaluator(DATASET_NAME, cfg, False, output_dir=output_dir)
counter_evaluator = Counter()

# Sử dụng DatasetEvaluators để gộp các evaluator lại với nhau
combined_evaluator = DatasetEvaluators([coco_evaluator, counter_evaluator])

# --- Chạy inference và đánh giá ---
print("🚀 Bắt đầu đánh giá model trên dataset test...")
results = inference_on_dataset(model, data_loader, combined_evaluator)
print("🔍 Kết quả đánh giá:")
for k, v in results.items():
    print(f"  {k}: {v}")
