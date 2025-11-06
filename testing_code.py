import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from itertools import cycle


IMG_HEIGHT, IMG_WIDTH = 128, 128
SHAPE_CLASSES = ['circle', 'square', 'triangle', 'hexagon', 'pentagon']
images_dir = 'grayscaled_images'
labels_file = 'colorful_labels/labels.csv'
model_path = 'best_model.h5'
SEED = 42 # CRUCIAL: Must match training script

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

    return np.array(images), labels, labels_df

try:
    X, labels_list, labels_df = load_data(images_dir, labels_file)
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels_list)
    
    _, X_test, _, y_test, _, X_test_indices = train_test_split(
        X, y, range(len(X)), test_size=0.2, random_state=SEED
    )

    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    
    y_pred = model.predict(X_test)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the data generation, grayscaling, and training steps were completed successfully.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()


y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy*100:.2f}% (Exact Match Ratio)")


try:
    roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
    print(f"ROC-AUC Score (weighted): {roc_auc:.4f}")
except ValueError:
    print("Could not compute ROC-AUC.")


class_report = classification_report(y_test, y_pred_binary, target_names=SHAPE_CLASSES)
print("\nClassification Report (Micro-Averages are best for multi-label):")
print(class_report)


y_test_single = y_test.argmax(axis=1)
y_pred_single = y_pred.argmax(axis=1)

conf_matrix = confusion_matrix(y_test_single, y_pred_single)
print("\nConfusion Matrix (Based on Single Most Confident Label):")
print(conf_matrix)




predicted_labels_list = []
for i, row in enumerate(y_pred_binary):
    original_index = X_test_indices[i]
    filename = labels_df.iloc[original_index]['filename'] 
    predicted_shapes = mlb.inverse_transform(np.array([row]))[0]
    predicted_labels_list.append([filename, ', '.join(predicted_shapes)])

predicted_labels_df = pd.DataFrame(predicted_labels_list, columns=['filename', 'predicted_shapes'])
predicted_labels_file = 'predicted_labels_test_set.csv'
predicted_labels_df.to_csv(predicted_labels_file, index=False)
print(f"\nPredicted labels saved to {predicted_labels_file}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=SHAPE_CLASSES, yticklabels=SHAPE_CLASSES)
plt.title("Confusion Matrix (Single-Label View)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_plot.png")

class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
plt.figure(figsize=(10, 6))
plt.bar(SHAPE_CLASSES, class_accuracies, color='skyblue')
plt.title("Class-wise Accuracy (Single-Label View)")
plt.xlabel("Shape Class")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("class_accuracy_plot.png")


print("\nSaved confusion_matrix_plot.png and class_accuracy_plot.png.")

plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
fpr, tpr, roc_auc = {}, {}, {}

for i, shape in enumerate(SHAPE_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, color=next(colors),
             label=f"{shape} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Label ROC-AUC Curve')
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_auc_multilabel.png", dpi=300)
plt.close()
print("✅ ROC-AUC plot saved as 'roc_auc_multilabel.png'.")


report = classification_report(y_test, y_pred_binary, target_names=SHAPE_CLASSES, output_dict=True)

metrics = ["precision", "recall", "f1-score"]
report_matrix = np.array([[report[cls][m] for m in metrics] for cls in SHAPE_CLASSES])

plt.figure(figsize=(8, 5))
sns.heatmap(report_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=metrics, yticklabels=SHAPE_CLASSES, cbar=False)
plt.title("Classification Report Heatmap")
plt.tight_layout()
plt.savefig("classification_report_heatmap.png", dpi=300)
plt.close()
print("✅ Classification report heatmap saved as 'classification_report_heatmap.png'.")
