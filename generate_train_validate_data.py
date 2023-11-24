import numpy as np

import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
from skimage.util import montage

from image_and_mask_generator import make_image_gen, create_aug_gen
from balance_data import BalanceData
from preparing_train_and_validation_data import TrainPreparing
import main_parameters as mp


balance_train = BalanceData().balanced_train_df
valid_data = TrainPreparing().valid_df


class TrainData:
    def __init__(self):
        self.train_gen = make_image_gen(balance_train)
        self.train_x, self.train_y = next(self.train_gen)
        self.valid_x, self.valid_y = next(
            make_image_gen(valid_data, mp.VALID_IMG_COUNT)
        )
        self.t_x, self.t_y = next(self.cur_gen_method())

    def cur_gen_method(self):
        cur_gen = create_aug_gen(self.train_gen, seed=42)
        return cur_gen

    def visualize_img(self):
        montage_rgb = lambda x: np.stack(
            [montage(x[:, :, :, i]) for i in range(x.shape[3])], -1
        )
        batch_rgb = montage_rgb(self.train_x)
        batch_seg = montage(self.train_y[:, :, :, 0])
        batch_overlap = mark_boundaries(batch_rgb, batch_seg.astype(int))
        titles = ["Image", "Segmentations", "Bounding Boxes on ship in Image"]
        colors = ["g", "m", "b"]
        display = [batch_rgb, batch_seg, batch_overlap]
        plt.figure(figsize=(25, 10))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(display[i])
            plt.title(titles[i], fontsize=18, color=colors[i])
            plt.axis("off")
        plt.suptitle("Batch Visualizations", fontsize=20, color="r", weight="bold")
        plt.tight_layout()
        plt.show()
