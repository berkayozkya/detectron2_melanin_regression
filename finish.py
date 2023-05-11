import torch
import torchvision
import cv2
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib.patches import Rectangle
from detectron2.utils.visualizer import ColorMode
import glob
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk

model = joblib.load('model.joblib')

def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["height"] = 700
        record["width"] = 700
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes.index(anno['label']),
                "segmentation": [poly],
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def crop_part(img,box):
    print(img.shape)
    x_top_left = int(box[0])
    y_top_left = int(box[1])
    x_bottom_right = int(box[2])
    y_bottom_right = int(box[3])

    y = y_top_left
    x = x_top_left
    height = y_bottom_right-y_top_left
    width = x_bottom_right - x_top_left
    cropped_part = img[y:y+height, x:x+width]
    return cropped_part
def crop_part2(img, y, height, x, width):
    cropped_part2 = img[y:y+height, x:x+width]
    return cropped_part2
classes = ['sen']

data_path = 'data/'


for d in ["train", "test"]:
    DatasetCatalog.register(
        "my_" + d, 
        lambda d=d: get_data_dicts(data_path+d, classes)
    )
    MetadataCatalog.get("my_" + d).set(thing_classes=classes)

microcontroller_metadata = MetadataCatalog.get("my_train")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")) 
cfg.DATASETS.TRAIN = ("my_train",) 
cfg.DATASETS.TEST = ()
cfg.MODEL.DEVICE = "cpu"
cfg.DATALOADER.NUM_WORKERS = 2 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml") 
cfg.SOLVER.IMS_PER_BATCH = 2 
cfg.SOLVER.BASE_LR = 0.001 
cfg.SOLVER.GAMMA = 0.05 
cfg.SOLVER.STEPS = [500] 
cfg.TEST.EVAL_PERIOD = 200 
cfg.SOLVER.MAX_ITER = 1000 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9 
cfg.DATASETS.TEST = ("my_test", ) 
predictor = DefaultPredictor(cfg) 
test_metadata = MetadataCatalog.get("my_test")
test_dataset_dicts = get_data_dicts(data_path+'test', classes)

for imageName in glob.glob(r'C:\Users\berka\desktop\Proje\son\700x700\0.05\photo6012449616328507274.jpg'):
  im = cv2.imread(imageName)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                metadata=test_metadata, 
                scale=0.8
                 )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

boxes = outputs["instances"].pred_boxes
box_list = []
for i in range(0,len(boxes)):
    box = list(boxes)[i].detach().cpu().numpy()
    box_list.append(box)
box_list = sorted(box_list, key=lambda x:x[0])
# Ürünleri kaydet
for i in range(0, len(box_list)):
    box = list(box_list)[i]
    box_list.append(box)
    crop_img = crop_part(im, box)
    dosya_yolu = r"C:/Users/berka/desktop/Proje\son/imgs/"
    cv2.imwrite(dosya_yolu+"/"+"image_"+str(i)+".jpg",cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))

img0 = cv2.imread(r"C:\Users\berka\desktop\Proje\son\imgs\image_0.jpg")
img1 = cv2.imread(r"C:\Users\berka\desktop\Proje\son\imgs\image_1.jpg")

crop_img0 = crop_part2(img0,
(img0.shape[0]-int(0.8*img0.shape[0])), 
(img0.shape[0]-int(0.8*img0.shape[0]))-(img0.shape[0]-int(0.4*img0.shape[0])), 
(img0.shape[1]-int(0.8*img0.shape[1])),
(img0.shape[1]-int(0.8*img0.shape[1])) - (img0.shape[1]-int(0.4*img0.shape[1])))
crop_img1 = crop_part2(img1,
(img1.shape[0]-int(0.8*img1.shape[0])), 
(img1.shape[0]-int(0.8*img1.shape[0]))-(img1.shape[0]-int(0.4*img1.shape[0])), 
(img1.shape[1]-int(0.8*img1.shape[1])),
(img1.shape[1]-int(0.8*img1.shape[1])) - (img1.shape[1]-int(0.4*img1.shape[1])))


def average_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color= [ int(x) for x in avg_color]
    return avg_color[0]


cons = "bilinmiyor"

img_adi = os.path.basename(imageName)
ters_img_adi = img_adi[::-1]
ind = ters_img_adi.index(".")
sensorters = ters_img_adi[ind+1:]
sensor = sensorters[::-1]




avg_color0 = average_color(crop_img0)
avg_color1 = average_color(crop_img1)
vary0 = np.std(crop_img0.ravel()) ** 2
vary1 = np.std(crop_img1.ravel()) ** 2
min0 = np.min(crop_img0)
min1 = np.min(crop_img1)
max0 = np.max(crop_img0)
max1 = np.max(crop_img1)
std0 = np.std(crop_img0.ravel())
std1 = np.std(crop_img1.ravel())

histogram0, bin_edges = np.histogram(crop_img0.ravel(), bins=256, range=(0, 256))
histogram1, bin_edges = np.histogram(crop_img1.ravel(), bins=256, range=(0, 256))


s0=[str(i) for i in histogram0]
s1=[str(i) for i in histogram1]
ss1 = (" ".join(s1))
ss0 = (" ".join(s0))

Label_list = ['PHOTO','KONSANTRASYON','MIN_control', 'MIN_test', 'MAX_control', 'MAX_test', 'AVG_control', 'AVG_test', 'VAR_control', 'VAR_test',
    'STD_control', 'STD_test','HIST_control','HIST_test']
Data_List = [sensor,cons,min0,min1,max0,max1,avg_color0,avg_color1,vary0,vary1,std0,std1
,ss0,ss1]
new_data = pd.Series(Data_List, index=Label_list)
new_data = pd.Series(new_data).values.reshape(1, -1)
df = pd.DataFrame(new_data)
df.columns = Label_list
df.drop(["PHOTO"], axis = 1, inplace = True)
df.drop(["KONSANTRASYON"], axis = 1, inplace = True)
hist_control_array1 = np.array([np.fromstring(x, dtype=int, sep=" ") for x in df["HIST_control"]])
df = df.drop(columns=["HIST_control"])
for i in range(hist_control_array1.shape[1]):
    df[f"HIST_control_{i}"] = hist_control_array1[:, i]
hist_test_array1 = np.array([np.fromstring(x, dtype=int, sep=" ") for x in df["HIST_test"]])
df = df.drop(columns=["HIST_test"])
for i in range(hist_test_array1.shape[1]):
    df[f"HIST_test_{i}"] = hist_test_array1[:, i]

pred = model.predict(df)

root = tk.Tk()
root.title("TAHMİN")
value = pred

label = tk.Label(root, text="Tahmin edilen konsantrasyon değeri: {}".format(value), fg="black", bg="turquoise", font=("Arial", 14, "bold"))
label.pack()

root.mainloop() 

