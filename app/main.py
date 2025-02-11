import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import json
import torch

from .detectron2_config import predictor
from .utils import convert_detectron2_to_labelme

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = np.array(image)

    outputs = predictor(image)
    print(outputs)  # Kiểm tra output từ Detectron2
    instances = outputs["instances"]

    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()

    labelme_json = convert_detectron2_to_labelme(file.filename, image, pred_boxes, pred_classes, pred_masks)

    return labelme_json
