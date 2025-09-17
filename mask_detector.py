import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import cv2
import xml.etree.ElementTree as ET


# upload dataset
BASE_DIR = r"FACEMASK\dataset"  
IMG_DIR = os.path.join(BASE_DIR, "images")
ANNOT_DIR = os.path.join(BASE_DIR, "annotations")


# function to read the xml annotation files
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objs = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        objs.append((label, (xmin, ymin, xmax, ymax)))
    return objs


print("Loading dataset...")
data, labels = [], []

# loop through annotation files
for xml_file in os.listdir(ANNOT_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOT_DIR, xml_file)
    image_name = xml_file.replace(".xml", "")
    
    # find image with same name
    image_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        temp_path = os.path.join(IMG_DIR, image_name + ext)
        if os.path.exists(temp_path):
            image_path = temp_path
            break

    if image_path is None:
        print(f"Image not found for {xml_file}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        continue

   
    objects = parse_annotation(xml_path)
    for label, (x1, y1, x2, y2) in objects:
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face = cv2.resize(face, (128, 128))
        data.append(face)
        # 0 = with_mask, 1 = without_mask
        labels.append(0 if label == "with_mask" else 1)  


# convert to numpy arrays + normalize
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print(f"Loaded {len(data)} samples")



X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)



aug = ImageDataGenerator(
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, horizontal_flip=True,
    fill_mode="nearest"
)


# load mobilenetv2 as base model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128,128,3))
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)


# freeze base model layers so only head will train
for layer in baseModel.layers:
    layer.trainable = False


# compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# training
print("Training model...")
history = model.fit(
    aug.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train)//32,
    epochs=10
)


# save the trained model
model.save("mask_detector.h5")
print("Model saved as mask_detector.h5")

# Evaluate model Accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {accuracy*100:.2f}%")
