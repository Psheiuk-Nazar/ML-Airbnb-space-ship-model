import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import numpy as np
from keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
)
from model import seg_model, td
from image_and_mask_generator import create_aug_gen, make_image_gen
import main_parameters as mp
from balance_data import BalanceData
from keras import models, layers


def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1 - y_true, 1 - y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return -K.mean((intersection + eps) / (union + eps), axis=0)


weight_path = "{}_weights.best.hdf5".format("seg_model")

checkpoint = ModelCheckpoint(
    weight_path,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=True,
)

reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.33,
    patience=1,
    verbose=1,
    mode="min",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-8,
)

early = EarlyStopping(
    monitor="val_loss", mode="min", verbose=2, patience=20
)  # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]
bd = BalanceData()


def fit():
    seg_model.compile(
        optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=["binary_accuracy"]
    )

    step_count = min(mp.MAX_TRAIN_STEPS, bd.train_df.shape[0] // mp.BATCH_SIZE)
    aug_gen = create_aug_gen(make_image_gen(bd.train_df))
    loss_history = [
        seg_model.fit_generator(
            aug_gen,
            steps_per_epoch=step_count,
            epochs=mp.MAX_TRAIN_EPOCHS,
            validation_data=(td.valid_x, td.valid_y),
            callbacks=callbacks_list,
            workers=1,  # the generator is not very thread safe
        )
    ]
    return loss_history


while True:
    loss_history = fit()
    if np.min([mh.history["val_loss"] for mh in loss_history]) < -0.2:
        break

seg_model.load_weights(weight_path)
seg_model.save("seg_model.h5")

pred_y = seg_model.predict(td.valid_x)
print(pred_y.shape, pred_y.min(axis=0).max(), pred_y.max(axis=0).min(), pred_y.mean())

if mp.IMG_SCALING is not None:
    fullres_model = models.Sequential()
    fullres_model.add(layers.AvgPool2D(mp.IMG_SCALING, input_shape=(None, None, 3)))
    fullres_model.add(seg_model)
    fullres_model.add(layers.UpSampling2D(mp.IMG_SCALING))
else:
    fullres_model = seg_model
fullres_model.save("fullres_model.h5")
