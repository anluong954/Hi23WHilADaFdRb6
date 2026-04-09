import Main
import tensorflow as tf
from Main import train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes, num_classes, evaluate_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

# VGG-16
def vgg_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    vgg16 = VGG16(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in vgg16.layers:
        layer.trainable = False

    x = vgg16.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    vgg_model = Model(inputs=vgg16.input, outputs=output_layer)
    vgg_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_vgg16 = vgg_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return vgg_model, trained_vgg16


# Training
vgg_model, trained_vgg16 = vgg_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig2 = plt.gcf()
plt.plot(trained_vgg16.history["accuracy"], label="accuracy")
plt.plot(trained_vgg16.history["val_accuracy"], label="val_accuracy")
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

vgg_save_path = os.path.join(save_dir, "vgg16.keras")
vgg_model.save(vgg_save_path)

sizes["vgg"] = os.path.getsize(vgg_save_path) / 1e6
vgg16 = sizes["vgg"]
print(f"VGG16 model size: {sizes['vgg']:.2f} MB")

vgg_model_evaluation = evaluate_model(vgg_model, test_imgs, num_classes=num_classes)