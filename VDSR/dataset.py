import os
import argparse
import cv2
import h5py
import numpy as np
from util import *

def prepare_data(path):

    count = 0
    names = os.listdir(path)
    names = sorted(names)
    nums = names.__len__()

    data = np.zeros((size_input, size_input, 1, 432968), dtype=np.double)
    label = np.zeros((size_label, size_label, 1, 432968), dtype=np.double)

    for i in range(nums):
        for flip in range(2):
            for degree in range(4):
                for scale in scales:

                    name = path + names[i]
                    hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
                    #shape = hr_img.shape
                    hr_img = cv2.cvtColor(hr_img, cv2.IMREAD_COLOR)
                    hr_img = cv2.flip(hr_img, flip)

                    if degree == 1:
                        hr_img = cv2.rotate(hr_img, cv2.ROTATE_90_CLOCKWISE)

                    elif degree == 2:
                        hr_img = cv2.rotate(hr_img, cv2.ROTATE_180)

                    else:
                        hr_img = cv2.rotate(hr_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    if hr_img.shape[2] == 3:
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
                        hr_img = im2double(hr_img[:, :, 0])
                        shape = hr_img.shape
                        # print(hr_img.shape)
                        height = shape[0]
                        width = shape[1]

                        im_label = hr_img[0:height - (height % scale), 0:width - (width % scale)]
                        # print(im_label.shape)
                        [hei, wid] = im_label.shape
                        #print(hei," ",wid)
                        im_input = cv2.resize(im_label, (wid//scale, hei//scale), cv2.INTER_CUBIC)
                        # print(im_input.shape)
                        im_input = cv2.resize(im_input, (wid, hei), cv2.INTER_CUBIC)
                        # print(im_label.shape)
                        # print(im_input.shape)
                        for x in range(0, hei-size_input, stride):
                            for y in range(0, wid-size_input, stride):

                                subim_input = im_input[x:x+size_input, y:y+size_input]
                                subim_label = im_label[x:x+size_label, y:y+size_label]
                                # print(subim_input.shape,"   ",subim_label.shape,"  --",im_input.shape,"  ",scale)
                                #if (subim_label.shape == (41,41)).all() and (subim_input== (41,41)).all():
                                data[:,:,0,count] = subim_input
                                #print(subim_label.shape)
                                label[:,:,0,count] = subim_label
                                # data.append(np.expand_dims(subim_input,axis=2))
                                # label.append(np.expand_dims(subim_label,axis=2))
                                count += 1

    order = np.random.permutation(count)
    print(count)
    # data = np.reshape(np.array(data),(41,41,1,count))
    # label = np.reshape(np.array(label), (41, 41, 1, count))
    # label = np.array(label)
    data = np.take(data, order, axis=3)
    label = np.take(label, order, axis=3)
    # data = np.expand_dims(data, axis=2)
    # label = np.expand_dims(label, axis=2)

    return data, label

def write_hdf5(data, label, file):

    x = data.astype(np.float32)
    y = label.astype(np.float32)

    with h5py.File(file, 'w') as h:
        h.create_dataset('data', data=x)
        h.create_dataset('label', data=y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="path to the dataset directory")
    args = parser.parse_args()

    DATA_PATH = args.dir
    size_input = 41
    size_label = 41
    stride = 41
    scales = [2, 3, 4]
    downsizes = [1, 0.7, 0.5]

    print("preparing data")
    data, label = prepare_data(DATA_PATH)
    print("storing data")
    write_hdf5(data, label, "train.h5")
