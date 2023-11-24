from numpy import np
import os
from exploring_the_data import test_image_dir
from exploring_the_data import TrainDataExploring as tde
from skimage.morphology import binary_opening, disk
from skimage.io import imread
from fit_model_prepering import fullres_model
import pandas as pd
from tqdm import tqdm_notebook


test_paths = np.array(os.listdir(test_image_dir))
print(len(test_paths), "test images found")


def raw_prediction(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = fullres_model.predict(c_img)[0]
    return cur_seg, c_img[0]


def smooth(cur_seg):
    return binary_opening(cur_seg > 0.99, np.expand_dims(disk(2), -1))


def predict(img, path=test_image_dir):
    cur_seg, c_img = raw_prediction(img, path=path)
    return smooth(cur_seg), c_img


def pred_encode(img, **kwargs):
    cur_seg, _ = predict(img)
    cur_rles = tde.multi_rle_encode(cur_seg, **kwargs)
    return [[img, rle] for rle in cur_rles if rle is not None]


out_pred_rows = []
for c_img_name in tqdm_notebook(
    test_paths[:30000]
):  ## only a subset as it takes too long to run
    out_pred_rows += pred_encode(c_img_name, min_max_threshold=1.0)

sub = pd.DataFrame(out_pred_rows)
sub.columns = ["ImageId", "EncodedPixels"]
sub = sub[sub.EncodedPixels.notnull()]

sub1 = pd.read_csv("../input/sample_submission_v2.csv")
sub1 = pd.DataFrame(
    np.setdiff1d(sub1["ImageId"].unique(), sub["ImageId"].unique(), assume_unique=True),
    columns=["ImageId"],
)
sub1["EncodedPixels"] = None
print(len(sub1), len(sub))

sub = pd.concat([sub, sub1])
print(len(sub))
sub.to_csv("submission.csv", index=False)
sub.head()
