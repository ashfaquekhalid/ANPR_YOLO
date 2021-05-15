
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
# %matplotlib inline  
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image
from yolov3.configs import *
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def predict_from_model(image):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    json_file = open('Plate_detect_and_recognize/MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Plate_detect_and_recognize/License_character_recognition_weight.h5")
    labels = LabelEncoder()
    labels.classes_ = np.load('Plate_detect_and_recognize/license_character_classes.npy')
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def plate_number(image_path):
  yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
  yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights
  image,bb = detect_image(yolo, image_path, '', input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  c = np.array(bb[0][:4], dtype=np.int32)
  org_image = cv2.imread(image_path)
  cropped = org_image[c[1]:c[3], c[0]:c[2]]
  cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
  cropped = cv2.resize(cropped, (224,224))
  gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(7,7),0)
  binary = cv2.threshold(blur, 180, 255,
                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
  cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box

  test_roi = cropped.copy()

# Initialize a list which will be used to append charater image
  crop_characters = []

# define standard width and height of character
  digit_w, digit_h = 30, 60

  for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 3<=ratio<5: # Only select contour with defined ratio
        #if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

            # Sperate number and gibe prediction
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)

  #print("Detect {} letters...".format(len(crop_characters)))
  final_string = ''
  for i,character in enumerate(crop_characters):
    title = np.array2string(predict_from_model(character))
    final_string+=title.strip("'[]")

  return final_string