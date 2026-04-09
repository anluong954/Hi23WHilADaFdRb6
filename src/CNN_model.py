import Main
import tensorflow as tf
from Main import train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes, num_classes, evaluate_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os

custom_model = models.Sequential([
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(img_height, img_width, img_chns)
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu'
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(classes), activation='softmax'),
])
custom_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
custom_model.summary()
custom_model_history =custom_model.fit(
    train_imgs,
    epochs=10,
    validation_data=val_imgs
)

fig1 = plt.gcf()
plt.plot(custom_model_history.history['accuracy'])
plt.plot(custom_model_history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Custom Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
custom_model_evaluation = evaluate_model(custom_model, test_imgs, num_classes)
print(custom_model_evaluation)

# Saving
sizes = {}

# Save under a predictable folder next to this script (more portable than hardcoding)
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)

custom_save_path = os.path.join(save_dir, "custom.keras")
custom_model.save(custom_save_path)

sizes["custom"] = os.path.getsize(custom_save_path) / 1e6
custom = sizes["custom"]
print(f"custom model size: {sizes['custom']:.2f} MB")
