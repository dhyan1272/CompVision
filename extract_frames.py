import os
import cv2

#Root Data set folder
DATASET_FOLDER = "DAiSEE_/DataSet"
OUTPUT_ROOT = "Frames"
video_exts = ('.mp4', '.avi', '.mov', '.mkv')

DEBUG_ONLY_TRAINING = True
DEBUG_FIRST_N_USER = True
N = 2
FRAME_STRIDE = 20
DESIRED_FRAMES = 10

#Loop over top-level folders (Train/Test/Validation)
for split in os.listdir(DATASET_FOLDER):

    if DEBUG_ONLY_TRAINING and split.lower()!="train":
        continue

    split_path=os.path.join(DATASET_FOLDER, split)
    if not os.path.isdir(split_path):
        continue

    print(f"Processing {split_path}")

    #Loop over users
    users = [u for u in os.listdir(split_path)
             if os.path.isdir(os.path.join(split_path,u))]

    if DEBUG_FIRST_N_USER:
        users = users[:N]

    for user in users:
        user_path=os.path.join(split_path, user)
        if(not os.path.isdir(user_path)):
            continue
        print(user_path) 

        #Loop over clips/expressions of a user
        for clip_folder in os.listdir(user_path):
            clip_path = os.path.join(user_path, clip_folder)
            if(not os.path.isdir(clip_path)):
                continue
 
            videos=[]
            for f in os.listdir(clip_path):
                if f.lower().endswith(video_exts):
                    videos.append(f)
            if not videos:
                continue
            video_file = videos[0]
            video_path = os.path.join(clip_path, video_file)
            cap=cv2.VideoCapture(video_path)
            frame_count=0
            frames=[]
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # save image as jpeg
                if frame_count % FRAME_STRIDE == 0:
                    frames.append(frame)

                frame_count +=1
            cap.release()
            
            while len(frames) < DESIRED_FRAMES:
                frames.append(frames[-1])  # repeat last frame
            # -----------------------------
            # TRUNCATE to required length if longer
            # -----------------------------
            frames = frames[:DESIRED_FRAMES]

            #Output folder
            rel_path = os.path.relpath(clip_path, DATASET_FOLDER)
            output_dir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                video_base_name = os.path.splitext(video_file)[0]
                frame_name = f"{video_base_name}_{i+1:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame) 
