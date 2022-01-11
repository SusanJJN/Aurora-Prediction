import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from .config import *


def image_initialize(image, img_size, img_chns):
    picture = Image.open(image)
    picture = picture.resize((img_size, img_size), Image.ANTIALIAS)
    if img_chns != 1:
        picture = picture.convert('L')
    data = np.array(picture.getdata()).reshape(img_size, img_size, img_chns)
    data = data.astype(np.float32) / 255

    return data


def get_npz(img_size, data_path, path, npz_save_path, img_chns):
    num = 0
    img_list = np.array([])
    file_list = os.listdir(os.path.join(data_path, path))
    file_list.sort(key=lambda x: int(x[:-4]))
    for i in range(len(file_list)):
        image_array = image_initialize(os.path.join(os.path.join(data_path, path), file_list[i]), img_size, img_chns)
        img_list = np.append(img_list, image_array)
        num += 1
    img_list = img_list.reshape(num, img_size * img_size * img_chns)
    np.savez(npz_save_path + path + '.npz', images=img_list)


# 用于多帧的训练
def get_3chn_seq(npz_path, npz_file, basic_frames, interval_frames, img_size, img_chns):
    raw_seq = np.load(os.path.join(npz_path, npz_file))['images']  # load array
    raw_len = raw_seq.shape[0]
    raw_seq = raw_seq.reshape(raw_len, img_size, img_size, img_chns)

    basic_len = raw_len - basic_frames - interval_frames + 1
    next_len = basic_len
    basic_seq = np.zeros((basic_len, basic_frames, img_size, img_size, img_chns))
    next_seq = np.zeros((next_len, basic_frames, img_size, img_size, img_chns))

    for i in range(basic_frames):
        basic_seq[:, i, :, :] = raw_seq[i:i + basic_len]
        next_seq[:, i, :, :] = raw_seq[i + interval_frames:i + basic_len + interval_frames]

    return basic_seq, next_seq


