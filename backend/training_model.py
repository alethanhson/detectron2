import os
import torch
import detectron2
from detectron2.engine import DefaultTrainer, hooks, HookBase
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.solver import WarmupMultiStepLR
from detectron2.data import build_detection_train_loader
from detectron2.solver import build_lr_scheduler
import logging

# ================================
# 🛠️ CONFIGURATION
# ================================
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
MAX_ITER = 70000
EVAL_PERIOD = 2000  # 🔥 Đánh giá mô hình mỗi 2000 iter
BASE_LR = 0.00015
NUM_CLASSES = 4
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
CHECKPOINT_PERIOD = 2000
BEST_CHECKPOINT_METRIC = "bbox/AP"  # Sử dụng AP làm thước đo lưu mô hình tốt nhất
TENSORBOARD_LOGS = "./logs/detectron2"
# TIME_DETECT = "1"

# ================================
# 🔥 REGISTER DATASETS (COCO FORMAT)
# ================================
TRAIN_DATA_SET_NAME = "train_dataset"
TRAIN_DATA_SET_ANN_FILE_PATH = "/workspace/annotations/data_train/data_train_34k_coco/annotations.json"
TRAIN_DATA_SET_IMAGES_DIR_PATH = "/workspace/annotations/data_train/data_train_34k_coco"

VAL_DATA_SET_NAME = "val_dataset"
VAL_DATA_SET_ANN_FILE_PATH = "/workspace/annotations/data_eval/_annotations.json"
VAL_DATA_SET_IMAGES_DIR_PATH = "/workspace/annotations/data_eval"

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

register_coco_instances(
    name=VAL_DATA_SET_NAME,
    metadata={},
    json_file=VAL_DATA_SET_ANN_FILE_PATH,
    image_root=VAL_DATA_SET_IMAGES_DIR_PATH
)

metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)
dataset_dicts = DatasetCatalog.get(TRAIN_DATA_SET_NAME)

# ================================
# 🔥 DETECTRON2 CONFIGURATION
# ================================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
cfg.MODEL.WEIGHTS = os.path.join("models/mask_rcnn_R_101_FPN_3x/time_1", "model_final.pth")

# DATASET
cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME,)
cfg.DATASETS.TEST = (VAL_DATA_SET_NAME,)

# SOLVER (TRAINING CONFIG)
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.WARMUP_ITERS = 2000
cfg.SOLVER.GAMMA = 0.8
cfg.SOLVER.STEPS = (30000, 40000)
cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS

# ROI HEADS (TĂNG PROPOSALS)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1536  # 🔥 Tăng số lượng proposal mỗi ảnh

# INPUT (ADVANCED AUGMENTATION)
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.INPUT.MIN_SIZE_TRAIN = (256, 320, 384, 448, 512)
cfg.INPUT.RANDOM_FLIP = "horizontal"
cfg.INPUT.MIN_SIZE_TEST = 256
cfg.INPUT.MAX_SIZE_TRAIN = 512
cfg.INPUT.MAX_SIZE_TEST = 512
cfg.INPUT.COLOR_AUG_SSD = True  # 🔥 Color Augmentation
cfg.INPUT.ROTATION_RANGE = (-30, 30)  # 🔥 Random Rotation
cfg.INPUT.BRIGHTNESS = 0.2  # 🔥 Adjust brightness
cfg.INPUT.CONTRAST = 0.2  # 🔥 Adjust contrast
cfg.DATALOADER.NUM_WORKERS = 8
# AMP (Mixed Precision Training)
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR" 
cfg.SOLVER.AMP.ENABLED = True
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.8, 0.8]
cfg.INPUT.CROP.ENABLED = True
cfg.MODEL.BACKBONE.FREEZE_AT = 2 
# OUTPUT DIR
OUTPUT_DIR = "./models/mask_rcnn_R_101_FPN_3x/time_2"
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# 🚀 CUSTOM TRAINER VỚI LOGGING & BEST CHECKPOINT
# ================================
class BestCheckpointHook(HookBase):
    """Lưu lại mô hình tốt nhất dựa trên mAP"""
    def __init__(self, cfg, trainer):
        self.cfg = cfg
        self.trainer = trainer
        self.best_metric = -1
        self.best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")

    def after_step(self):
        # 🔥 Kiểm tra để tránh chia cho 0
        eval_period = self.cfg.TEST.EVAL_PERIOD if self.cfg.TEST.EVAL_PERIOD > 0 else 2000
        
        if (self.trainer.iter + 1) % eval_period == 0:
            results = self.trainer.storage.latest().get(BEST_CHECKPOINT_METRIC, None)
            if results is not None and results > self.best_metric:
                self.best_metric = results
                self.trainer.checkpointer.save(self.best_model_path)
                logging.info(f"✅ Saved Best Model at {self.best_model_path} with {BEST_CHECKPOINT_METRIC}: {results:.4f}")

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointHook(self.cfg, self))
        return hooks

# ================================
# 🚀 START TRAINING
# ================================
def main():
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
