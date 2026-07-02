import os
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf

print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Disable GPU if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
print(tf.__version__)


#Data set Information
DATA_FOLDER = "EXTRACT_CROPPED1/Train"
LABEL_FILE  = "DAiSEE_/Labels/TrainLabels.csv"
BATCH_SIZE  = 4
FRAMES = 10
IMG_SIZE = 224
CHANNELS = 3
#DEPENDS ON BATCH SIZE
# In DAiSEE, 5358 labels and 5481 clips in training, 210552061 has a label but no video
STEPS_PER_EPOCH = 1370
N_EPOCHS = 10

## Loads paths all the expressions of all users
## Two loops: first over each user and then for each expression of a user
def data_generator(folder, label_dict):
    clip_paths=[]

    #Loop over all users
    for user in os.listdir(folder):
        user_path=os.path.join(folder, user)
        
        #Loop over all expressions of an user
        for clip in os.listdir(user_path):
            clip_path=os.path.join(user_path, clip)
            clip_paths.append(clip_path)

    print ("Total CLips:", len(clip_paths), "First five clips: ")
    print ("\n".join(clip_paths[:5]))

    while True: 
        random.shuffle(clip_paths)
        labeled_clips = 0
        unlabeled_clips = 0
 
        for i in range(0, len(clip_paths), BATCH_SIZE):
            batch = clip_paths[i:i+BATCH_SIZE]
            X, y = [], []
 
            for path in batch:
                clip_data=load_clip(path)
                clip_id = os.path.basename(path)

                if clip_id in label_dict:
                    X.append(clip_data)
                    y.append(label_dict[clip_id])
                    labeled_clips = labeled_clips + 1
                else:
                    unlabeled_clips = unlabeled_clips + 1

            #print(f"Sizes: {np.shape(np.array(X))}, {np.shape(np.array(y))}")
            #print(f"Total labeled clips: {labeled_clips}, unlabeled_clips: {unlabeled_clips}")
            yield np.array(X), np.array(y)

        

# Loads all frames/images for each expresion of a user, i.e., one video
def load_clip(folder):
    frames = sorted(os.listdir(folder)) 
    clip = []
 
    for f in frames:
        path = os.path.join(folder, f)
        img =cv2.imread(path)
        if img is None:
            continue
        img= img.astype("float32")/255.0
        clip.append(img)

    if len(clip)==0:
        return None
    assert len(clip) == FRAMES, "Number of frames per clip not equal to #FRAMES"
 
    return np.array(clip)


#Load the y label for each
def load_labels(csv_file):
    df=pd.read_csv(csv_file)
    #Remove spaces etc in colum headers
    df.columns = df.columns.str.strip()
    label_dict={}
 
    for _, row in df.iterrows():
        
        #clip_id = row["ClipID"].replace(".avi", "") 
        #Donot use as it can have mp4 files too
        clip_id = os.path.splitext(row["ClipID"].strip())[0]
        label = [
            row["Boredom"],
            row["Engagement"],
            row["Confusion"],
            row["Frustration"]
        ]

        label_dict[clip_id] = label

    print ("Total labels loaded:", len(label_dict), "First five keys and values")
    print (list(label_dict.keys())[:5])
    print (list(label_dict.values())[:5])

    return label_dict


def build_model():
        
    inputs = tf.keras.Input(shape=(FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS))

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(16, (3,3), activation='relu')
    )(inputs)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D()
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten()
    )(x)

    x = tf.keras.layers.LSTM(32)(x)

    outputs = tf.keras.layers.Dense(4, activation = 'softmax')(x)

    #build model
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(hist):
    # Get values
    loss = hist.history['loss']
    accuracy = hist.history.get("accuracy")

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Loss_accuracy.jpg", dpi=450)
    plt.close()


def main():

    print ("Loading Labels and create label dictionary...")
    label_dict = load_labels(LABEL_FILE)
    
    print ("Loading data and creating generator...")
    gen = data_generator(DATA_FOLDER, label_dict)
 
    print ("Building model...")
    model = build_model()

    print ("Start training...")
    history = model.fit(gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS)

    plot_history(history)

if __name__ == "__main__":
    main()

