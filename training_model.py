import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import random 

# Set a fixed seed for reproducibility (Crucial for consistent splits!)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

images_dir = 'grayscaled_images'
labels_file = 'colorful_labels/labels.csv'

IMG_HEIGHT, IMG_WIDTH = 128, 128
SHAPE_CLASSES = ['circle', 'square', 'triangle', 'hexagon', 'pentagon']

def load_data(images_dir, labels_file):
    labels_df = pd.read_csv(labels_file)
    images = []
    labels = []

    for _, row in labels_df.iterrows():
        image_path = os.path.join(images_dir, row['filename'])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0
        images.append(image.reshape(IMG_HEIGHT, IMG_WIDTH, 1))

        shapes = row['shapes'].split(', ')
        labels.append(shapes)

    return np.array(images), labels

def create_model(input_shape, num_classes):
    # Enhanced CNN Architecture (Deeper and wider)
    model = Sequential([
        # 1st Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        # 2nd Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # 3rd Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # 4th Conv Block (Added for enhancement)
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        
        # Dense Layers
        Dense(256, activation='relu'), 
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Main Execution ---

# 1. Load Data
X, labels_list = load_data(images_dir, labels_file)

# 2. Binarize Labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels_list)

# 3. Split data into training and testing sets (CRUCIAL: Use random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# 4. Model Setup (Ensure a fresh start)
input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
num_classes = len(SHAPE_CLASSES)

if os.path.exists('best_model.h5'):
    os.remove('best_model.h5')
if os.path.exists('last_model.h5'):
    os.remove('last_model.h5')

model = create_model(input_shape, num_classes)
print("Starting training from scratch with enhanced model and consistent split.")

# 5. Callbacks
callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max", verbose=1), 
    ModelCheckpoint("last_model.h5", save_best_only=False, monitor="val_accuracy", mode="max", verbose=1), 
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max", verbose=1)
]

# 6. Training
history = model.fit(
    X_train, y_train,
    epochs=18,
    batch_size=500,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete. Best model saved as 'best_model.h5'.")



