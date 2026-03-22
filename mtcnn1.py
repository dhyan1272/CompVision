import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import tensorflow as tf
from mtcnn import MTCNN

print(tf.__version__)

#Parameters and Directories
DATASET_FOLDER="Frames"
OUTPUT_DIR="Cropped"
DEVICE="CPU:0"
#------------------------------------------

#Initialize MTCNN Face Detector
detector=MTCNN()
#------------------------------------------


DEBUG_ONLY_TRAINING = True

#Loop over top-level folders (Train/Test/Validation)
for split in os.listdir(DATASET_FOLDER):

    if DEBUG_ONLY_TRAINING and split.lower()!="train":
        continue

    split_path=os.path.join(DATASET_FOLDER, split)
    if not os.path.isdir(split_path):
        continue

    print(f"Processing {split_path}")

    # All users in split path
    users = [u for u in os.listdir(split_path)
             if os.path.isdir(os.path.join(split_path,u))]

    print(users[:5])
   
    #Loop over each folder in a user which will have a particular expression
    for user in users:
        user_path=os.path.join(split_path, user)
        if(not os.path.isdir(user_path)):
            continue

        print(f"Processing User: {user_path}") 

        for clip_folder in os.listdir(user_path):
            clip_path = os.path.join(user_path, clip_folder)
            
            if(not os.path.isdir(clip_path)):
                continue
 
            #Make folder for this clip
            relpath=os.path.relpath(clip_path, DATASET_FOLDER)
            output_dir=os.path.join(OUTPUT_DIR, relpath)
            os.makedirs(output_dir, exist_ok=True)

            for file in os.listdir(clip_path):
                file_path = os.path.join(clip_path, file)
                img=cv2.imread(file_path)
                
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
                    face=img_rgb[y:y+height, x:x+width]
                    face = cv2.resize(face, (224, 224)) 
                    final_img=face
                else:
                    final_img=img_rgb
    

                # Convert back to BGR before saving
                final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    
                #Save the cropped images
                output_path= os.path.join(output_dir, file)
                cv2.imwrite(output_path, final_img_bgr) 
