from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

def training_original_yolo(dataset_yaml, folder_name):
    model_list = ["yolov9c.pt", "yolov9e.pt"]

    model_large = YOLO(model_list[0])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=8, name= f"{folder_name}_c")

    model_large = YOLO(model_list[1])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=2, name= f"{folder_name}_e")

    # model_large = YOLO(model_list[2])
    # model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=8, name= f"{folder_name}_c")

    # model_large = YOLO(model_list[2])
    # model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=2, name= f"{folder_name}_e")


# Yolov8 DOCS https://docs.ultralytics.com/modes/train/
if __name__ == "__main__":

    training_original_yolo("uk_pest_dataset_26MAR.yaml","uk_pest_26MAR_24_classes")
