from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras import backend as K




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

# def customized_loss(y_true, y_pred, lamada=0.0):
#     return losses.mean_absolute_error(y_pred, y_true) + lamada * perceptual_loss(y_pred, y_true)
def customized_loss(lamda = 0.0):
    """softmax loss"""
    print('using lamda',lamda)
    def softmax_loss(y_true, y_pred):
        return losses.mean_absolute_error(y_pred, y_true) + lamda * perceptual_loss(y_pred, y_true)
    return softmax_loss
