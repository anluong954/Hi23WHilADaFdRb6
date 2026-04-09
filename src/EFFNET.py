import Main
import tensorflow as tf
from Main import train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes, num_classes, evaluate_model
from tensorflow.keras.applications.efficientnet import EfficientNetB5  # EfficientNet
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

# EfficientNet
def effnet_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    effnet = EfficientNetB5(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in effnet.layers:
        layer.trainable = False

    x = effnet.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    effnet_model = Model(inputs=effnet.input, outputs=output_layer)
    effnet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_effnet = effnet_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return effnet_model, trained_effnet


# Training
effnet_model, trained_effnet = effnet_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig5 = plt.gcf()
plt.plot(trained_effnet.history["accuracy"], label="accuracy")
plt.plot(trained_effnet.history["val_accuracy"], label="val_accuracy")
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

effnet_save_path = os.path.join(save_dir, "effnet.keras")
effnet_model.save(effnet_save_path)

sizes["effnet"] = os.path.getsize(effnet_save_path) / 1e6
effnet = sizes["effnet"]
print(f"Effnet model size: {sizes['effnet']:.2f} MB")

effnet_model_evaluation = evaluate_model(effnet_model, test_imgs, num_classes=num_classes)