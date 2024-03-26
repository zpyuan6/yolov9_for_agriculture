from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

def training_original_yolo(dataset_yaml, folder_name):
    model_list = ["yolov9s.pt", "yolov9m.pt", "yolov9e.pt", "yolov9e.pt"]

    model_large = YOLO(model_list[0])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=16, name= f"{folder_name}_s")

    model_large = YOLO(model_list[1])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=16, name= f"{folder_name}_m")

    model_large = YOLO(model_list[2])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=8, name= f"{folder_name}_c")

    model_large = YOLO(model_list[2])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=2, name= f"{folder_name}_e")


# Yolov8 DOCS https://docs.ultralytics.com/modes/train/
if __name__ == "__main__":

    training_original_yolo("uk_pest_dataset_01JAN.yaml","uk_pest_01JAN")
