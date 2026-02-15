import os
import glob
import cv2
import json
import csv
import tensorflow as tf
from mtcnn import MTCNN

print(tf.__version__)

#Parameters
DATASET_DIR="Driver_Drowsiness_DataSet"
OUTPUT_JSON="drowsiness_results.json"
output_csv="drowsiness_summary.csv"
DEVICE="CPU:0"
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
results = []
for file in image_paths[:3]:
    try:
        img=cv2.imread(file)
        if img is None:
            print(f"Failed to load {file}")
            continue
        #Convert to RGB
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #Detect Faces
        '''
        faces=detector.detect_faces(img_rgb)
        
        #Extract Label from the folder
        label=os.path.basename(os.path.dirname(file))

        #Store results
        record={
            "file":file,
            "label":label,
            "num_faces": len(faces),
            "faces": faces
        #}
        print(record)
        
        #Store them in results
        #results.append(record)
        '''
    except Exception as e:
        print(f"Error processing {file}: {e}")
#------------------------------------------
