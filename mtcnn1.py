import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import glob
import cv2
import json
import csv
import tensorflow as tf
from mtcnn import MTCNN

print(tf.__version__)

#Parameters and Directories
DATASET_DIR="Driver_Drowsiness_DataSet"
OUTPUT_DIR="Processed_Images"
DEVICE="CPU:0"
os.makedirs(OUTPUT_DIR, exist_ok=True)
#------------------------------------------

#Initialize MTCNN Face Detector
detector=MTCNN()
#------------------------------------------

#Collect all image paths
image_paths=glob.glob(os.path.join(DATASET_DIR, "*", "*.png"))
image_paths.sort()
print("Total images found: "+ str(len(image_paths)))
print("First 5 image names: ")
for path in image_paths[:5]:
    print(path)
#------------------------------------------

#Process images and detect faces
for file in image_paths:
    
    img=cv2.imread(file)

    if img is None:
        print(f"Failed to load {file}")
        continue
    
    #Convert to RGB
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    #Detect Faces
    try:
        faces=detector.detect_faces(img_rgb)
        
    except Exception as e:
        print(f"Error detecting {file}: {e}")

    if faces:
        x, y, width, height = faces[0]['box']
        # safety clamp
        x = max(0, x)
        y = max(0, y)
        cropped_face=img_rgb[y:y+height, x:x+width]
        final_img=cropped_face
    else:
        final_img=img_rgb
    
    # ----------------------------------------
    # Build output subfolder name
    # if file = Driver_Drowsiness_DataSet/Drowsy/A0001.png, class_name=Drowsy
    class_name = os.path.basename(os.path.dirname(file))
    class_out_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_out_dir, exist_ok=True)
    #------------------------------------------

    #------------------------------------------
    # Convert back to BGR before saving
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    #------------------------------------------

    #------------------------------------------
    filename = os.path.basename(file)
    save_path = os.path.join(class_out_dir, filename)
    cv2.imwrite(save_path, final_img_bgr) 
#------------------------------------------
