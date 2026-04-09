import Main
import tensorflow as tf
from Main import train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes, num_classes, evaluate_model
from tensorflow.keras.applications.resnet50 import ResNet50  # ResNet (50 layers)
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

# Resnet
def resnet_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    resnet = ResNet50(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in resnet.layers:
        layer.trainable = False

    x = resnet.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    resnet_model = Model(inputs=resnet.input, outputs=output_layer)
    resnet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_resnet = resnet_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return resnet_model, trained_resnet


# Training
resnet_model, trained_resnet = resnet_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig3 = plt.gcf()
plt.plot(trained_resnet.history["accuracy"], label="accuracy")
plt.plot(trained_resnet.history["val_accuracy"], label="val_accuracy")
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

resnet_save_path = os.path.join(save_dir, "resnet.keras")
resnet_model.save(resnet_save_path)

sizes["resnet"] = os.path.getsize(resnet_save_path) / 1e6
resnet = sizes["resnet"]
print(f"Resnet model size: {sizes['resnet']:.2f} MB")

resnet_model_evaluation = evaluate_model(resnet_model, test_imgs, num_classes=num_classes)