import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
os.environ["TORCH_NNPACK"] = "0"  # Vô hiệu hóa NNPACK nếu không hỗ trợ

def setup_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "models/model_final.pth"  # Thay bằng model đã train
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Đổi thành số lớp đúng của mô hình
    cfg.MODEL.DEVICE = "cpu"
    
    return DefaultPredictor(cfg)

predictor = setup_detectron2()
