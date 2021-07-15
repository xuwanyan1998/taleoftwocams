from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from pyscikit import ImgKit
import tensorflow as tf
from absl import logging


from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random
import tiffile as tiff
import PIL.Image as Image

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)
    else:
        logging.info("No GPU found!")
set_memory_growth()



input_dir = "data2/input/"
target_dir = "data2/target/"


input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".tif")
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".tif") and fname.startswith("M_Align")
    ]
)
# remove not matched pairs
input_img_paths2 = []
target_img_paths2 = []
for filepath in input_img_paths:
    if os.path.exists(filepath.replace("input/C","target/M_Align_")):
        input_img_paths2.append(filepath)
        target_img_paths2.append(filepath.replace("input/C","target/M_Align_"))




for input_path, target_path in zip(input_img_paths2[:10], target_img_paths2[:10]):
    print(input_path, "|", target_path)


print("Number of input_img_paths samples:", len(input_img_paths))
print("Number of target_img_paths samples:", len(target_img_paths))

print("Number of input_img_paths samples:", len(input_img_paths2))
print("Number of target_img_paths samples:", len(target_img_paths2))
# print(input_img_paths2[-10:])
# print(target_img_paths2[-10:])
input_img_paths = input_img_paths2
target_img_paths = target_img_paths2

class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        xi,yi,wi,hi = random.randint(50,100),random.randint(50,100),512,512
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
#             img = load_img(path, target_size=self.img_size)
            img = tiff.imread(path)
#             x[j] = img.crop((xi,yi,xi+wi,yi+hi))
            img = img[xi:xi+wi,yi:yi+hi]
#             img = (img / 127.5) - 1
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
#             img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = tiff.imread(path)
#             img = img.crop((xi,yi,xi+wi,yi+hi))
            img = img[xi:xi+wi,yi:yi+hi]
#             img = (img / 127.5) - 1
    # Crop image
# image_arr = image_arr[700:1400, 1450:2361]
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
#             y[j] -= 1
#         print(x,y)
        return x, y


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, padding="same")(x)
#     outputs = tensorflow.nn.depth_to_space(x, 2)
#     outputs = layers.MaxPooling2D(3, strides=1, padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

img_size = (512, 512)
# img_size = (160, 160)
num_classes = 1
batch_size = 4

# Build model
model = get_model(img_size, num_classes)
model.summary()

import random

# Split our img paths into a training and a validation set
val_samples = 10
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)


# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="adam", loss="mean_absolute_error")

callbacks = [
    keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 300
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)