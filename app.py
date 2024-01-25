from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import random
import uvicorn
import numpy as np
import cv2

modelo = YOLO("yolov8n.pt", task="detect")

def draw_results(image,image_results,show_id=False):
    annotator = Annotator(image.copy())
    for box in image_results.boxes:
        b = box.xyxy[0]             # get box coordinates in (top, left, bottom, right) format
        cls = int(box.cls)
        conf = float(box.conf)
        label = f"{modelo.names[cls]} {round(conf*100,2)}"
        if show_id:
            label+= f' id:{int(box.id)}'

        # if cls==2 and conf>=0.35:
        annotator.box_label(b, label, color=random.choices(range(256), k=3))    # boxes, text, color
    image_annotated = annotator.result()
    return image_annotated

def results_to_dict(results):
    data = {}
    for i,info in enumerate(results[0].boxes.data):  
        info = [float(x) for x in info]
        data[i]=info
    return data


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

import json

@app.get("/infere")
async def infere(data: UploadFile = File(...)):
    image_path = "test.jpg"
    image_bytes=data.file.read()
    buffer=np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(buffer,cv2.IMREAD_COLOR)
    results = modelo.predict(image, conf=0.40)
    image_annotated = draw_results(image,results[0])
    cv2.imwrite(image_path,image_annotated)
    data=json.dumps(results_to_dict(results))
    response = FileResponse(image_path)
    response.headers["results"]=data
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
