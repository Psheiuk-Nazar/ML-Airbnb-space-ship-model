import pandas as pd
from pandas import DataFrame


def read_masks() -> DataFrame:
    masks = pd.read_csv(".\\train_ship_segmentations_v2.csv")
    not_empty = pd.notna(masks.EncodedPixels)
    print(not_empty.sum(), "masks in", masks[not_empty].ImageId.nunique(), "images")
    print(
        (~not_empty).sum(), "empty images in", masks.ImageId.nunique(), "total images"
    )
    return masks
