from tensorflow import keras
import tensorflow as tf
from absl import app, flags, logging
import numpy as np
import yaml


def pack_raw(im):
    # pack Bayer image to 4 channels
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def pack_raw_back(im):
    # pack  4 channels to Bayer image
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.zeros((H*2,W*2,1), dtype="float32")
    out[0:H:2, 0:W:2, :] = im[:,:,0]
    out[0:H:2, 1:W:2, :] = im[:,:,0]
    out[1:H:2, 1:W:2, :] = im[:,:,0]
    out[1:H:2, 0:W:2, :] = im[:,:,0]
    return out

def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


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