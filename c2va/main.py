"""
Multispectral images Change Detection analysis using Compressed Change Vector Analysis (C2VA)

Version 0.1

Author: Luca Martinatti

Original release: 12/11/2020

Dataset: Onera Dataset (https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple
from enum import Enum

TRAIN_FILE = "train.txt"


class SplitSelectionType(Enum):
    MAGNITUDE = "magnitude"
    DIRECTION = "direction"


class PerformanceMetrics:
    """Performance metrics class

    Parameters:
        x = 1-D or 2-D array like
            Binary ([0,1] or [0,255]) array or matrix where 0 indicate absence of changing
            and 1 (or 255) indicate the presence of changing.

        y = 1-D or 2-D array like
            Binary ([0,1] or [0,255]) array or matrix where 0 indicate absence of changing
            and 1 (or 255) indicate the presence of changing.

    """

    x: np.array
    y: np.array

    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y
        self._calcualte_metrics()

    def _calcualte_metrics(self):
        if self.x.shape == self.y.shape:
            error = self.x - self.y
            # False Positive
            FP = np.count_nonzero(error > 0)
            # False Negative
            FN = np.count_nonzero(error < 0)
            true = np.count_nonzero(error == 0)  # TP + TN
            pos = np.count_nonzero(self.x)  # TP + FP
            neg = np.count_nonzero(self.x == 0)  # TN + FN
            # True Negative
            TN = neg - FN
            # True Positive
            TP = pos - FP

            # False Alarm probability
            self.prob_false_allarm = (FP / pos) * 100
            # Missed Alarm probability
            self.prob_missed_allarm = (FN / neg) * 100

            # Recall
            self.reccall = TP / (TP + FN)
            # Precision
            self.precision = TP / (TP + FP)
            # F1-score
            self.f1 = (2 * self.reccall * self.precision) / (
                self.reccall + self.precision
            )
            error = abs(self.x - self.y)
            summa = np.count_nonzero(error)
            if self.x.size == 0:
                raise ValueError("Prediction size equal to 0, invalid operation!")
            self.accuracy = ((self.x.size - summa) / self.x.size) * 100

        else:
            raise ValueError("Error sizes, accuracy function has different size input")


# Onera Dataset Multispectral images loader
def load_onera_dataset(
    base_path: str, image_with_groundtruth: bool = False, places_list: bool = False
) -> Tuple[list, list, list]:
    if image_with_groundtruth == True:
        f = open(base_path + TRAIN_FILE, "r")
        content = f.read()
        nameList = content.split(",")

    for i in range(2):
        ls = []
        l_entry = []
        for entry in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, entry)):
                if image_with_groundtruth == True:
                    if any(entry in s for s in nameList):
                        if i == 0:
                            im = "/imgs_1_rect/"
                        elif i == 1:
                            im = "/imgs_2_rect/"
                        path = base_path + entry + im
                        fil_list = os.listdir(path)
                        x = np.array(
                            [np.array(Image.open(path + fname)) for fname in fil_list]
                        )
                        l_entry.append(entry)
                        ls.append(x)
                elif image_with_groundtruth == False:
                    if i == 0:
                        im = "/imgs_1_rect/"
                    elif i == 1:
                        im = "/imgs_2_rect/"
                    path = base_path + entry + im
                    fil_list = os.listdir(path)
                    x = np.array(
                        [np.array(Image.open(path + fname)) for fname in fil_list]
                    )
                    l_entry.append(entry)
                    ls.append(x)
        if i == 0:
            ls1 = ls
        elif i == 1:
            ls2 = ls
    if places_list == True:
        return ls1, ls2, l_entry
    else:
        return ls1, ls2, None


# Onera Dataset Ground Truth loader
def load_ground_truth(base_path: str, places_list: bool = False) -> Tuple[list, list]:
    ls = []
    l_entry = []
    for entry in os.listdir(base_path):
        if os.path.isdir(entry):
            if os.path.isdir(os.path.join(base_path, entry)):
                path = base_path + entry + "/cm/"
                x = np.array(Image.open(path + entry + "-cm.tif"))
                l_entry.append(entry)
            ls.append(
                x - 1
            )  # original is ls.append(x)   the -1 is to convert ground truth from [2,1] range into [1,0] range
    if places_list == True:
        return ls, l_entry
    else:
        return ls, None


# Compressed Change Vector Analysis function
def c2va(i_before: np.array, i_after: np.array) -> Tuple[np.array, np.array]:
    # Magnitude
    sum_x = 0
    dif = np.zeros((i_before.shape[0], i_before.shape[1], i_before.shape[2]))
    for i in range(i_before.shape[0]):
        # dif[i] = cv2.absdiff(i_before[i], i_after[i])
        dif[i] = i_after[i] - i_before[i]
        sum_x = dif[i] ** 2

    d_mag = np.sqrt(sum_x)

    # Direction (phase)
    x_ref = np.ones(i_before.shape[0]) * (
        np.divide(np.sqrt(i_before.shape[0]), i_before.shape[0])
    )
    num = np.zeros(d_mag.shape)
    tmp = np.zeros(d_mag.shape)
    for b in range(dif.shape[0]):
        num += dif[b] * x_ref[b]
        tmp += dif[b] ** 2
    den = np.sqrt(tmp * (sum(x_ref**2)))

    out = (
        np.ones((i_before.shape[1], i_before.shape[2])) * 0.0000001
    )  # Zero value are redefined as small number 0.0000001 (see below)
    d_dir = np.arccos(
        np.divide(num, den, out=out, where=den != 0)
    )  # To avoid nan problem due to an dominator equal to zero
    return d_mag, d_dir


# Split magnitude and phase images in smaller ones
def image_splitting(mag: np.array, dir: np.array, w_size: list) -> list:
    """
    image_splitting(mag, dir, image=None)

        PARAMETERS:
                mag = 2-D array like magnitude image (image raw, image column)

                dir = 2-D array like direction image (image raw, image column)

                w_size = integer 2-D tuple with the dimension of the splitting window

        RETURNS:
                structure:  [out_Mag, out_Dir, out_Image]

                ls = list of 2-D array which contain:
                        |
                        |----N x M split (not all are equal) which contain:
                            |
                            |---- I x J pixel of one of the input  (not all are equal)
    """
    a, b = mag.shape[:]
    ls = []

    split_list = [mag, dir]

    r_size_dimensioning = True
    c_size_dimensioning = True
    while r_size_dimensioning:
        if (a % w_size[0]) == 0:
            rFinal = a % w_size[0]
            r_size_dimensioning = False
        elif (a % w_size[0]) <= (0.35 * w_size[0]):
            w_size[0] += 1
        elif (a % w_size[0]) >= (0.75 * w_size[0]):
            rFinal = a % w_size[0]
            r_size_dimensioning = False
        else:
            w_size[0] -= 1
    while c_size_dimensioning:
        if (b % w_size[1]) == 0:
            cFinal = b % w_size[1]
            c_size_dimensioning = False
        elif (b % w_size[1]) <= (0.35 * w_size[1]):
            w_size[1] += 1
        elif (b % w_size[1]) >= (0.75 * w_size[1]):
            cFinal = b % w_size[1]
            c_size_dimensioning = False
        else:
            w_size[1] -= 1

    for sl in split_list:
        window = []
        for r in range(0, a - w_size[0], w_size[0]):
            tmp = []
            for c in range(0, b - w_size[1], w_size[1]):
                tmp.append(sl[r : r + w_size[0], c : c + w_size[1]])
                if (c + w_size[1] + cFinal) == b:
                    tmp.append(
                        sl[r : r + w_size[0], c + w_size[1] : c + w_size[1] + cFinal]
                    )
            if tmp != []:
                window.append(tmp)
            tmp = []
            if (r + w_size[0] + rFinal) == a:
                for c in range(0, b - w_size[1], w_size[1]):
                    tmp.append(
                        sl[r + w_size[0] : r + w_size[0] + rFinal, c : c + w_size[1]]
                    )
                    if (c + w_size[1] + cFinal) == b:
                        tmp.append(
                            sl[
                                r + w_size[0] : r + w_size[0] + rFinal,
                                c + w_size[1] : c + w_size[1] + cFinal,
                            ]
                        )
            if tmp != []:
                window.append(tmp)
        win = np.array(window)
        ls.append(win)
    return ls


# Select the
def split_selection(
    split: list,
    original_dimensions: list,
    window_size: list,
    selection_value: SplitSelectionType = SplitSelectionType.MAGNITUDE,
    num_of_split: int = 6,
) -> Tuple[np.array, np.array]:
    """
    split_selection(split, selection_value=0, num_of_split = 6)

        PARAMETERS:
                split = list of splitted images produced by "image_splitting" function

                original_dimensions = int list, original image dimensions

                window_size = tuple of int, split window dimensions

                selection_value = SplitSelectionType, optional
                                  Decide which element of split use to make the selection.
                                  Default value is 'magnitude' indicate that use magnitude for the selection
                                  Other possible value is 'direction' that indicate that use direction for the selection

                num_of_split = int, optional
                               Indicate the number of split selected (default value = 6)

        RETURNS:
                structure:  [out_Mag, out_Dir]

                ls = tuple of 'num_of_split' selected split.
    """
    if selection_value.value == SplitSelectionType.DIRECTION.value:
        sp = split[1]
    else:
        sp = split[0]
    standDev = np.zeros((sp.shape[0], sp.shape[1]))
    for r in range(sp.shape[0]):
        for c in range(sp.shape[1]):
            standDev[r, c] = sp[r, c].std()

    mag = np.zeros((original_dimensions[0], original_dimensions[1]))
    pha = np.zeros((original_dimensions[0], original_dimensions[1]))
    for i in range(int(num_of_split)):
        ind = np.unravel_index(np.argmax(standDev, axis=None), standDev.shape)
        mag[
            ind[0] * window_size[0] : ind[0] * window_size[0] + split[0][ind].shape[0],
            ind[1] * window_size[1] : ind[1] * window_size[1] + split[0][ind].shape[1],
        ] = split[0][ind]
        pha[
            ind[0] * window_size[0] : ind[0] * window_size[0] + split[0][ind].shape[0],
            ind[1] * window_size[1] : ind[1] * window_size[1] + split[0][ind].shape[1],
        ] = split[1][ind]
        standDev[ind] = 0.0
    return mag, pha


# Change Detection image producer
def CDmap(image: np.array, threshold: float = None) -> np.array:
    # Normalize image between 0 and max
    im = image - image.min()
    if threshold != None:
        T = threshold
    else:
        rang = im.max()
        if abs(rang) < 1.0:
            T = round((rang / 4), ndigits=1)
        else:
            # Shift threshold from 1/2 to be more sensitive
            T = round(rang / 4)
    t, it = cv2.threshold(im, T, 1, type=cv2.THRESH_BINARY)
    w = np.count_nonzero(it)
    b = np.count_nonzero(it == 0)
    if w > b:
        ti = abs(it - 1)
        return np.array(ti, dtype=int)
    else:
        return np.array(it, dtype=int)


# Remove specific bands from the dataset
def remove_bands(before: np.array, after: np.array, bands_list: list):
    bl = (np.array(bands_list)) - 1
    if bands_list != []:
        i_before = []
        i_after = []
        for im in range(len(before)):
            tmpA = np.zeros(
                (
                    before[im].shape[0] - len(bands_list),
                    before[im].shape[1],
                    before[im].shape[2],
                )
            )
            tmpB = np.zeros(
                (
                    before[im].shape[0] - len(bands_list),
                    before[im].shape[1],
                    before[im].shape[2],
                )
            )
            z = 0
            for b in range(before[im].shape[0]):
                if b in bl:
                    pass
                else:
                    tmpA[z] = after[im][b]
                    tmpB[z] = before[im][b]
                    z += 1
            i_after.append(tmpA)
            i_before.append(tmpB)
        return i_before, i_after
    else:
        return before, after


def comparison_plot(mag: np.array, dir: np.array, g_true: np.array, titles: list):
    fig, ax = plt.subplots(1, 3)
    fig.suptitle("Change Detection maps", fontsize=20)
    cdM = [g_true, CDmap(mag), CDmap(dir)]
    ax[0].imshow(cdM[0], cmap="gray", interpolation="nearest")
    ax[0].set_title(titles[0])
    ax[1].imshow(cdM[1], cmap="gray", interpolation="nearest")
    conf_mat = PerformanceMetrics(cdM[1], g_true)
    ax[1].set_title(
        titles[1] + " acc= %.2f" % conf_mat.accuracy + "%" + " f1= %.2f" % conf_mat.f1
    )
    ax[2].imshow(cdM[2], cmap="gray", interpolation="nearest")
    conf_mat_2 = PerformanceMetrics(cdM[2], g_true)
    ax[1].set_title(
        titles[1]
        + " acc= %.2f" % conf_mat_2.accuracy
        + "%"
        + " f1= %.2f" % conf_mat_2.f1
    )
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def graphs(mag: np.array, dir: np.array, g_true: np.array, split_mag: np.array):
    # Plot the Magnitude histogram graph
    his = plt.figure(figsize=(8, 8))
    n, bins, patches = plt.hist(
        mag.ravel(), bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel(r"X$\rho$")
    plt.ylabel(r"h(X$\rho)$")
    plt.title("Magnitude", fontsize=20)
    plt.text(23, 45, r"$\mu=15, b=3$")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    # Plot the magnitude and phase images
    fig, ax = plt.subplots(1, 2)
    fig.suptitle("Magnitude and Phase images", fontsize=20)
    ax[0].imshow(mag)
    ax[0].set_title(r"Magnitude ($\rho) map$")
    ax[1].imshow(dir)
    ax[1].set_title(r"Phase ($\theta) map$")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    # Plot the comparison between the true and the producted Change Detection maps
    comparison_plot(mag, dir, g_true, titles=["Ground truth", "Magnitude", "Direction"])

    # Plot the comparison between the true and the producted Change Detection maps (also the splitted ones)
    comparison_plot(
        mag,
        g_true,
        split_mag,
        titles=["Ground truth", "Magnitude", "Splitted Magnitude"],
    )


def main():
    ### LOAD DATA AND PRE-PROCESSING ###

    # Onera Dataset folders paths
    base_path = "data/images/Onera Satellite Change Detection dataset - Images/"
    ground_truth_path = (
        "data/train_labels/Onera Satellite Change Detection dataset - Train Labels/"
    )

    # Load dataset and split it in before/after change images
    img_before, img_after, _ = load_onera_dataset(
        base_path=base_path, image_with_groundtruth=True
    )

    # Choose the list of images to analyze (Alphabetically ordered)
    imageList = [0]

    # Remove unuseful bands for change detection
    unuseful_bands = [10]
    i_before, i_after = remove_bands(img_before, img_after, unuseful_bands)

    # Load ground truth
    groundTr, _ = load_ground_truth(ground_truth_path)

    ### IMAGE ANALYSIS ###

    window_size = [100, 100]
    for im in imageList:
        # Apply C2VA Amalysis
        d_mag, d_dir = c2va(i_before[im], i_after[im])

        # Split image
        splittedImage = image_splitting(d_mag, d_dir, window_size)

        # Image produced adaptive split selection
        s_mag, s_dir = split_selection(
            splittedImage,
            original_dimensions=d_mag.shape[:],
            window_size=window_size,
            num_of_split=splittedImage[0].shape[0] * splittedImage[0].shape[1] * 0.1,
        )

        ### PLOT GRAPHS ###
        graphs(
            mag=d_mag,
            dir=d_dir,
            g_true=groundTr[im],
            split_mag=s_mag,
        )


if __name__ == "__main__":
    main()
