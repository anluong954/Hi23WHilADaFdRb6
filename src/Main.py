import numpy as np
import pandas as pd
import tensorflow as tf
import os

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))

def f1_score_macro_np(y_true, y_pred, num_classes: int | None = None) -> float:
    """
    Macro-F1 for single-label multiclass classification.
    Uses per-class F1 (with 0 when precision+recall==0), then averages across classes.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.size == 0:
        return 0.0

    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)

    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        denom = (2 * tp + fp + fn)
        f1_c = (2 * tp / denom) if denom != 0 else 0.0
        f1s.append(f1_c)

    return float(np.mean(f1s))

# Data Exploration
train_data_dir = 'C:/Users/anluo/OneDrive/Desktop/Projects/Project 4/images/training'
test_data_dir = 'C:/Users/anluo/OneDrive/Desktop/Projects/Project 4/images/testing'
img_height, img_width, img_chns = 180, 180, 3
batch_size = 32

train_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

classes = train_imgs.class_names
num_classes = len(classes)
print(classes)

plt.figure(figsize=(10, 10))
for images, labels in train_imgs.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes[int(labels[i])])
        plt.axis("off")

plt.show()

def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, num_classes: int) -> dict:
    y_true = []
    y_pred = []

    for batch_images, batch_labels in dataset:
        probs = model.predict(batch_images, verbose=0)
        pred_labels = np.argmax(probs, axis=1)

        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(pred_labels.tolist())

    acc = accuracy_score_np(y_true, y_pred)
    f1 = f1_score_macro_np(y_true, y_pred, num_classes=num_classes)

    return {"accuracy": float(acc), "f1_score": float(f1)}

def build_comparison_table():
    try:
        from EFFNET import effnet_model_evaluation, effnet
        from RESNET import resnet_model_evaluation, resnet
        from MOBILE import mobilenet_model_evaluation, mobile
        from VGG16 import vgg_model_evaluation, vgg16
        from CNN_model import custom_model_evaluation, custom
    except ImportError as exc:
        print(f"Could not import model results: {exc}")
        return None

    values = {
        'accuracy': [
            vgg_model_evaluation['accuracy'],
            resnet_model_evaluation['accuracy'],
            mobilenet_model_evaluation['accuracy'],
            effnet_model_evaluation['accuracy'],
            custom_model_evaluation['accuracy'],

        ],
        'f1_score': [
            vgg_model_evaluation['f1_score'],
            resnet_model_evaluation['f1_score'],
            mobilenet_model_evaluation['f1_score'],
            effnet_model_evaluation['f1_score'],
            custom_model_evaluation['f1_score'],
        ],
        'size': [
            vgg16,
            resnet,
            mobile,
            effnet,
            custom
        ]
    }

    df = pd.DataFrame(values, index=['vgg', 'resnet', 'mobilenet', 'efficientnet', 'custom'])
    df.to_pickle('model_comparison.pkl')
    print(df)
    return df

if __name__ == "__main__":
    build_comparison_table()