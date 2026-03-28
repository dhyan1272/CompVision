import os
import cv2
import tensorflow as tf
from mtcnn import MTCNN

# -------------------------
# SETTINGS
# -------------------------
DATASET_FOLDER = "DAiSEE_/DataSet"
OUTPUT_ROOT = "EXTRACT_CROPPED1"

video_exts = ('.mp4', '.avi', '.mov', '.mkv')

DEBUG_ONLY_TRAINING = True
DEBUG_FIRST_N_USER = False
N = 2

FRAME_STRIDE = 20
DESIRED_FRAMES = 10

# Disable GPU if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(tf.__version__)

# Initialize detector
detector = MTCNN()

# -------------------------
# MAIN LOOP
# -------------------------
for split in os.listdir(DATASET_FOLDER):

    if DEBUG_ONLY_TRAINING and split.lower() != "train":
        continue

    split_path = os.path.join(DATASET_FOLDER, split)
    if not os.path.isdir(split_path):
        continue

    print(f"Processing {split_path}")

    users = [u for u in os.listdir(split_path)
             if os.path.isdir(os.path.join(split_path, u))]

    if DEBUG_FIRST_N_USER:
        users = users[:N]

    for user in users:
        user_path = os.path.join(split_path, user)

        if not os.path.isdir(user_path):
            continue

        print(f"Processing User: {user_path}")

        for clip_folder in os.listdir(user_path):
            clip_path = os.path.join(user_path, clip_folder)

            if not os.path.isdir(clip_path):
                continue

            # find video
            videos = [f for f in os.listdir(clip_path)
                      if f.lower().endswith(video_exts)]

            if not videos:
                continue

            video_file = videos[0]
            video_path = os.path.join(clip_path, video_file)

            cap = cv2.VideoCapture(video_path)

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % FRAME_STRIDE == 0:

                    # Convert to RGB
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect face
                    try:
                        faces = detector.detect_faces(img_rgb)
                    except Exception as e:
                        print(f"Error detecting frame: {e}")
                        faces = []

                    if faces:
                        x, y, w, h = faces[0]['box']
                        x, y = max(0, x), max(0, y)

                        face = img_rgb[y:y+h, x:x+w]
                        face = cv2.resize(face, (224, 224))
                        final_img = face
                    else:
                        final_img = cv2.resize(img_rgb, (224, 224))

                    frames.append(final_img)

                frame_count += 1

            cap.release()

            # Pad frames if needed
            if len(frames) > 0:
                while len(frames) < DESIRED_FRAMES:
                    frames.append(frames[-1])

            frames = frames[:DESIRED_FRAMES]

            # Output directory
            rel_path = os.path.relpath(clip_path, DATASET_FOLDER)
            output_dir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            # Save frames
            video_base_name = os.path.splitext(video_file)[0]

            for i, frame in enumerate(frames):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame_name = f"{video_base_name}_{i+1:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_name)

                cv2.imwrite(frame_path, frame_bgr)

