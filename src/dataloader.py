import os
import matplotlib.pyplot as plt
from loader.loader_vot import loader_vot



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

    return image, coor_gt

def vot_to_rect(boxes):
    """
    convert vot gt information to rectangle
    :param boxes: vot gt
    :return: rects
    """


if __name__ == '__main__':
    DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'
    objLoaderVot = loader_vot(DATA_PATH)
    videos = objLoaderVot.get_videos()
    print('test done')