import keras.backend as K

from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.cast(y_true, "float32")  # Cast to float32
    y_pred = K.cast(y_pred, "float32")
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


