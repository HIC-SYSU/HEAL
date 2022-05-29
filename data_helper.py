import os
import pprint
import csv
from PIL import Image
import cv2 as cv
from skimage.transform import resize
import numpy as np
import pickle
import scipy.io as scio


resize_width=256;resize_height=256
def image_resize(image_data):
    resize_image_data = resize(image_data, (resize_width, resize_height), order=1, preserve_range=True)
    return resize_image_data

def from_image_to_data(image_name):
    im = Image.open(image_name)
    data = im.getdata()
    data = np.reshape(data, (512, 512, 1))
    data = data / 255.0
    data=image_resize(data)
    return data


def from_image_to_normal_data(image_name):
    image = cv.imread(image_name)
    #print(image)
    result = np.zeros(image.shape, dtype=np.float32)
    cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # print(result)
    # print(result.shape)
    re_data=image_resize(result)
    # print(re_data)
    # print(re_data.shape)
    return re_data


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min(batch_size * (batch_num + 1), data_size)
            yield shuffled_data[start_index:end_index]
