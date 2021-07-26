from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import os
from pyscikit import ImgKit
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import LossZoo
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard

import random

from ModelZoo import get_unet,get_unet_raw2gray
from TwoCamDatasets import TwoCamDatasets
from Utils import set_memory_growth,load_yaml

flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_integer('batch_size', '0', 'batch size to use')
flags.DEFINE_string('cfg_path', './configs/unetdemo.yaml', 'config file path')


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    cfg = load_yaml(FLAGS.cfg_path)

    if FLAGS.batch_size == 0:
        batch_size = cfg['batch_size']
    else:
        batch_size = FLAGS.batch_size
    # if modify here, TwoCamDS also need to be modified
    img_size_w = 512
    img_size_h = 512
    val_samples = 50
    learning_rate = cfg['base_lr']
    input_dir = cfg['train_dataset']+"/input/"
    target_dir = cfg['train_dataset']+ "/target/"
    epochs = cfg['epochs']

    set_memory_growth()
    tf.config.experimental.list_physical_devices('GPU')

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
        if os.path.exists(filepath.replace("input/C", "target/M_Align_")):
            input_img_paths2.append(filepath)
            target_img_paths2.append(filepath.replace("input/C", "target/M_Align_"))

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

    img_size = (img_size_w, img_size_h)

    # Split our img paths into a training and a validation set
    random.Random(1).shuffle(input_img_paths)
    random.Random(1).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = TwoCamDatasets(
        batch_size, img_size, img_size_w, train_input_img_paths, train_target_img_paths,packraw=False
    )
    val_gen = TwoCamDatasets(batch_size, img_size, img_size_w, val_input_img_paths, val_target_img_paths,packraw=False)
    # # Build model
    # model = get_unet(256, 256)
    model = get_unet_raw2gray(cfg['input_size'], cfg['input_size'])
    model.summary()


    tf.config.experimental_run_functions_eagerly(True)


    # model.compile(optimizer="adam", loss=[losses.mean_absolute_error,perceptual_loss],
    #               metrics=['mean_absolute_error',PSNRLoss,perceptual_loss],loss_weights=[1.0,3.0])#'mean_absolute_error',
    # optimizer = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)  # can use adam sgd
    model.compile(optimizer='adam', loss=LossZoo.customized_loss(0.9),
                  metrics=['mean_absolute_error', LossZoo.PSNRLoss, LossZoo.perceptual_loss])  # 'mean_absolute_error',
    # model.compile(optimizer="adam", loss=[losses.mean_absolute_error],
    #               metrics=['mean_absolute_error',PSNRLoss,perceptual_loss])#
    # 监控val_loss，当连续40轮变化小于0.0001时启动early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)

    tb_callback = TensorBoard(log_dir='logs/'+ cfg['sub_name'],
                                  update_freq=cfg['batch_size'] * 5,
                                  profile_batch=0)
    # tb_callback._total_batches_seen = steps
    # tb_callback._samples_seen = steps * cfg['batch_size']
    callbacks = [
        keras.callbacks.ModelCheckpoint("model_no_packraw_l2.h5", save_best_only=True),
        es,
        tb_callback
    ]

    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

if __name__ == '__main__':
    app.run(main)
