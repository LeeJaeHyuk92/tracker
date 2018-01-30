# from PY-GOTURN github page: https://github.com/nrupatunga/PY-GOTURN
# thanks for nrupatunga
# Date: Friday 02 June 2017 07:00:47 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: loading VOT dataset

import os
import numpy as np
import matplotlib.pyplot as plt

# data path

# return img, bbox coordinate for train


def image_reader(dir, filename):
    """
    :param dir: directory path
    :param filename: 00000000
    :return image:
    :return coor_gt:
    :return etc: occlusion, etc..
    """

    file_path = os.path.join(dir, filename+'.jpg')
    gt_path = os.path.join(dir, 'groundtruth.txt')

    image = plt.imread(file_path)
    f = open(gt_path, 'r')
    lines = f.readlines()
    bboxes = [x.strip() for x in lines]
    coor_gt = bboxes[int(filename)+1]

    return image, coor_gt, etc

def vot_to_rect(boxes):
    """
    convert vot gt information to rectangle
    :param boxes: vot gt
    :return: rects
    """

