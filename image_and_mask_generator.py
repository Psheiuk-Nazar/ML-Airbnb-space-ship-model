import os

import numpy as np


import exploring_the_data as exd
from matplotlib.pyplot import imread
import main_parameters as mp
from keras.preprocessing.image import ImageDataGenerator

dg_args = dict(
    featurewise_center=False,
    samplewise_center=False,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
    data_format="channels_last",
)


def make_image_gen(in_df, batch_size=mp.BATCH_SIZE):
    all_batches = list(in_df.groupby("ImageId"))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(exd.train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = exd.TrainDataExploring().masks_as_image(
                c_masks["EncodedPixels"].values
            )
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb) / 255.0, np.stack(out_mask)
                out_rgb, out_mask = [], []


image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)


if mp.AUGMENT_BRIGHTNESS:
    dg_args[" brightness_range"] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if mp.AUGMENT_BRIGHTNESS:
    dg_args.pop("brightness_range")
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(
            255 * in_x, batch_size=in_x.shape[0], seed=seed, shuffle=True
        )
        g_y = label_gen.flow(in_y, batch_size=in_x.shape[0], seed=seed, shuffle=True)

        yield next(g_x) / 255.0, next(g_y)
