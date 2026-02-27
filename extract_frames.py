import os
import cv2

#Root Data set folder
dataset_folder = "DAiSEE_/DataSet"
output_root = "Output"

#Loop over top-level folders (Train/Test/Validation)
for split in os.listdir(dataset_folder):
    split_path=os.path.join(dataset_folder, split)
    if(not os.path.isdir(split_path)):
        continue
    print(split_path)
     
    #Loop over users
    for user in os.listdir(split_path):
        user_path=os.path.join(split_path, user)
        if(not os.path.isdir(user_path)):
            continue
        print(user_path) 
 
        #Loop over clips/expressions of a user
        for clip_folder in os.listdir(user_path):
            clip_path = os.path.join(user_path+"/"+clip_folder)
            if(not os.path.isdir(clip_path)):
                continue
            print(clip_path)
 
            videos=[]
            for f in os.listdir(clip_path):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    videos.append(f)
                if not videos:
                    continue
                video_file = videos[0]
                video_path = os.path.join(clip_path + "/" + video_file)
                output_dir = os.path.join(output_root + "/" + clip_path)
                os.makedirs(output_dir, exist_ok=True)
                print(f"Extracting video from:{video_path}")
                print(f"Output will be in:{output_dir}")
                cap=cv2.VideoCapture(video_path)
                frame_count=0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break;
                    # save image as jpeg
                    frame_name = f"{os.path.splitext(video_file)[0]}_{frame_count+1:06d}.jpg"
                    frame_path = os.path.join(output_dir + "/" + frame_name)
                    cv2.imwrite(frame_path, frame)
                    frame_count +=1
                    print(frame_path) 
