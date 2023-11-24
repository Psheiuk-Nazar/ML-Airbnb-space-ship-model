import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from skimage.morphology import label

train_image_dir = "train_v2"
test_image_dir = "test_v2"


class TrainDataExploring:
    def __init__(self):
        self.train_image = self.read_and_sort_data()
        self.mask = self.train_ship_segmented_mask()

    @staticmethod
    def read_and_sort_data():
        train_image = os.listdir(train_image_dir)
        train_image.sort()
        return train_image

    def print_total_image(self):
        print(
            f"Total amount of image is: {len(self.train_image)},\nHere how first train_image looks like {self.train_image[:5]}"
        )

    def using_loop_to_show_image(self):
        plt.figure(figsize=(15, 15))
        plt.suptitle("TRAIN IMAGES\n", weight="bold", fontsize=15, color="r")
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(imread(train_image_dir + "/" + self.train_image[i]))
            plt.title(f"{self.train_image[i]}", weight="bold")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def train_ship_segmented_mask():
        masks = pd.read_csv("train_ship_segmentations_v2.csv")
        return masks

    @staticmethod
    def rle_decode(mask_rle, shape=(768, 768)):
        s = mask_rle.split()
        starts, lenghts = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        ends = starts + lenghts - 1
        img = np.zeros(shape[0] * shape[1], dtype=np.int8)
        for lo, hi in zip(starts, ends):
            img[lo : hi + 1] = 1

        return img.reshape(shape).T

    def masks_as_image(self, in_mask_list):
        all_masks = np.zeros((768, 768), dtype=np.int16)
        for mask in in_mask_list:
            if isinstance(mask, str):
                all_masks += self.rle_decode(mask)

        return np.expand_dims(all_masks, -1)

    def show_image(self):
        for num in [3, 4, 5, 6]:
            rle_0 = self.mask.query(f'ImageId=="{self.train_image[num-1]}"')[
                "EncodedPixels"
            ]
            img_0 = self.masks_as_image(rle_0)
            originals = imread(train_image_dir + "/" + self.train_image[num - 1])
            plt.figure(figsize=(15, 8))
            plt.subplot(1, 2, 1)
            plt.title(f"Original - Train Image {originals.shape}")
            plt.imshow(originals)
            plt.subplot(1, 2, 2)
            plt.title(f"Mask generated from the RLE data for each ship ")
            plt.imshow(img_0, cmap="Blues_r")
            plt.tight_layout()
            plt.show()

    @staticmethod
    def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
        """
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        """
        if np.max(img) < min_max_threshold:
            return ""  ## no need to encode if it's all zeros
        if max_mean_threshold and np.mean(img) > max_mean_threshold:
            return ""  ## ignore overfilled mask
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(x) for x in runs)

    def masks_as_color(self, in_mask_list):
        # Take the individual ship masks and create a color mask array for each ships
        all_masks = np.zeros((768, 768), dtype=np.float)
        scale = lambda x: (len(in_mask_list) + x + 1) / (
            len(in_mask_list) * 2
        )  ## scale the heatmap image to shift
        for i, mask in enumerate(in_mask_list):
            if isinstance(mask, str):
                all_masks[:, :] += scale(i) * self.rle_decode(mask)
        return all_masks

    def multi_rle_encode(self, img, **kwargs):
        """
        Encode connected regions as separated masks
        """
        labels = label(img)
        if img.ndim > 2:
            return [
                self.rle_encode(np.sum(labels == k, axis=2), **kwargs)
                for k in np.unique(labels[labels > 0])
            ]
        else:
            return [
                self.rle_encode(labels == k, **kwargs)
                for k in np.unique(labels[labels > 0])
            ]
