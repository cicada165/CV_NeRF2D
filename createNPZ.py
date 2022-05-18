from __future__ import print_function
from __future__ import absolute_import
from distutils.dir_util import copy_tree

import os
import sys
import glob
import json
import re
import shutil
from shutil import copytree, ignore_patterns
import numpy as np

CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 72000

class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

print("Converting images to numpy arrays...")

for f in os.listdir("./image"):
    if f.find(".png") != -1:
        img = Utils.get_preprocessed_img("{}/{}".format("./image", f), IMAGE_SIZE)
        file_name = f[:f.find(".png")]

        np.savez_compressed("{}/{}".format("./", file_name), features=img)
        retrieve = np.load("{}/{}.npz".format("./", file_name))["features"]

        assert np.array_equal(img, retrieve)

        # shutil.copyfile("{}/{}.gui".format("./image", file_name), "{}/{}.gui".format("./", file_name))

print("Numpy arrays saved in {}".format("./"))

data_all = [np.load(fname) for fname in ["image000.npz",
"image001.npz","image002.npz","image003.npz","image004.npz","image005.npz","image006.npz","image007.npz","image008.npz","image009.npz",
"image010.npz","image011.npz","image012.npz","image013.npz","image014.npz","image015.npz","image016.npz","image017.npz","image018.npz","image019.npz",]]
merged_data = {}
for data in data_all:
    [merged_data.update({k: v}) for k, v in data.items()]
np.savez('data.npz', **merged_data)