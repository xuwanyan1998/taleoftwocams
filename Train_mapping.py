from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from pyscikit import ImgKit
import tensorflow as tf
from absl import logging

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

import random

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose


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
# set_memory_growth()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
set_memory_growth()
tf.config.experimental.list_physical_devices('GPU')

input_dir = "data2/input/"
target_dir = "data2/target/"

img_size_w = 512
img_size_h = 512


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

def pack_raw(im):
    # pack Bayer image to 4 channels
#     im = raw.raw_image_visible.astype(np.float32)
#     im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

#     im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


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
        xi,yi,wi,hi = random.randint(50,400),random.randint(50,400),img_size_w,img_size_h
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
#         print(batch_input_img_paths,batch_target_img_paths)
#         x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        x = np.zeros((self.batch_size,) + (256,256) + (4,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path)
            img = img[xi:xi+wi,yi:yi+hi]
#             img = (img / 127.5) - 1
#             x[j] = img
            x[j] = pack_raw(img)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = io.imread(path)
            img = img[xi:xi+wi,yi:yi+hi]
#             img = (img / 127.5) - 1
            y[j] = np.expand_dims(img, 2)

        return x, y


img_size = (img_size_w, img_size_h)
# img_size = (160, 160)
num_classes = 1
batch_size = 16


# Split our img paths into a training and a validation set
val_samples = 50
random.Random(1).shuffle(input_img_paths)
random.Random(1).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)


def get_unet(rows, cols):
    inputs = Input((rows, cols, 4))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    #     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9), tf.image.resize(conv1,size=[512,512])], axis=3)
    #     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    #     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(4, (1, 1), activation='linear')(conv9)

    out = tf.nn.depth_to_space(conv10, 2)

    model = Model(inputs=[inputs], outputs=[out])

    return model

# display(Image.fromarray(y[0]))
# # Build model
model = get_unet(256, 256)
model.summary()

# # display(Image.fromarray(y[0]))
from tensorflow.keras import backend as K
from tensorflow.keras import losses

tf.config.experimental_run_functions_eagerly(True)


@tf.function
def perceptual_loss(y_true, y_pred):
    #     print("Note:Need to remove vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 to C:\\Users\\user_name\\.keras\\models\\")
    vgg_inp = tf.keras.Input([512, 512, 3])
    vgg = tf.keras.applications.VGG19(include_top=False, input_tensor=vgg_inp)
    for l in vgg.layers: l.trainable = False
    vgg_out_layer = vgg.get_layer(index=5).output
    vgg_content = tf.keras.Model(vgg_inp, vgg_out_layer)

    y_true = tf.tile(y_true, [1, 1, 1, 3])
    y_pred = tf.tile(y_pred, [1, 1, 1, 3])
    #     print("***************",y_true)

    y_t = vgg_content(y_true)
    y_p = vgg_content(y_pred)
    loss = tf.keras.losses.mean_squared_error(y_t, y_p)
    return tf.reduce_mean(loss)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true)) / 255 ** 2) / K.log(10.)


# def PSNRLoss(y_true, y_pred):
#     return 10. * K.log(255**2/mean_squared_error(y_pred , y_true))/ K.log(10.)

def customized_loss(y_true, y_pred, lamada=0.01):
    return losses.mean_absolute_error(y_pred, y_true) + lamada * perceptual_loss(y_pred, y_true)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses
# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
# model.compile(optimizer="adam", loss=[losses.mean_absolute_error,perceptual_loss],
#               metrics=['mean_absolute_error',PSNRLoss,perceptual_loss],loss_weights=[1.0,3.0])#'mean_absolute_error',
model.compile(optimizer="adam", loss=customized_loss,
              metrics=['mean_absolute_error',PSNRLoss,perceptual_loss])#'mean_absolute_error',
# model.compile(optimizer="adam", loss=[losses.mean_absolute_error],
#               metrics=['mean_absolute_error',PSNRLoss,perceptual_loss])#
# 监控val_loss，当连续40轮变化小于0.0001时启动early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)

callbacks = [
    keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True),
    es
]

# Train the model, doing validation at the end of each epoch.
epochs = 300
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)