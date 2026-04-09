import Main
import tensorflow as tf
from Main import train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes, num_classes, evaluate_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2  # MobileNet (version 2)
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

# MobileNet
def mobilenet_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    mobilenet = MobileNetV2(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in mobilenet.layers:
        layer.trainable = False

    x = mobilenet.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    mobilenet_model = Model(inputs=mobilenet.input, outputs=output_layer)
    mobilenet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_mobilenet = mobilenet_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return mobilenet_model, trained_mobilenet


# Training
mobilenet_model, trained_mobilenet = mobilenet_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig4 = plt.gcf()
plt.plot(trained_mobilenet.history["accuracy"], label="accuracy")
plt.plot(trained_mobilenet.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

# Saving
sizes = {}

# Save under a predictable folder next to this script (more portable than hardcoding)
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)

mobile_save_path = os.path.join(save_dir, "mobile.keras")
mobilenet_model.save(mobile_save_path)

sizes["mobile"] = os.path.getsize(mobile_save_path) / 1e6
mobile = sizes["mobile"]
print(f"mobile model size: {sizes['mobile']:.2f} MB")

mobilenet_model_evaluation = evaluate_model(mobilenet_model, test_imgs, num_classes=num_classes)