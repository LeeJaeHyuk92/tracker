from training_schedules import POLICY_TEST
from src.dataloader import load_batch
from progressbar import ProgressBar, Percentage, Bar
from scipy.misc import imread, imresize
from loader.loader_vot import loader_vot
import os
import tensorflow as tf
from model import net
import numpy as np
from src.utils import bcolors
import cv2
import time

# DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

ckpt = './logs/180212_long_vot2015_ROI/model.ckpt-392900'
# symlink = 'datapath'
data_dir = './data/vot2015_full'
# os.symlink(symlink, data_dir)
result_dir = './result'

# Verify arguments are valid
if not os.path.exists(ckpt + '.meta'):
    raise ValueError('ckpt does not exist')
if not os.path.isdir(data_dir):
    raise ValueError('data_dir must exist and be a directory')

sub_vot_dirs = [dir_name for dir_name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir_name))]
for vot_sub_dir in sub_vot_dirs:
    vot_dir_path = os.path.join(result_dir, data_dir, vot_sub_dir)
    if not os.path.exists(vot_dir_path):
        os.makedirs(vot_dir_path)


Tracker = net()
Tracker.build_model_test(POLICY=POLICY_TEST)

# Load each data sample (pimg, cimg, pbox, cbox) and write it to the TFRecord
objLoaderVot = loader_vot(data_dir)
videos = objLoaderVot.get_videos()
video_keys = videos.keys()

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
saver = tf.train.Saver()
# TODO, FIX, graph g
with tf.Session(config=run_config).as_default() as sess:
    saver.restore(sess, ckpt)
    print(bcolors.WARNING + 'Model is restored' + bcolors.ENDC)

for idx in range(len(videos)):
    video_frames = videos[video_keys[idx]][0]         # ex) bag/*.jpg list
    annot_frames = videos[video_keys[idx]][1]         # ex) bag/groundtruth, rectangle box info
    num_frames = min(len(video_frames), len(annot_frames))

    # num_frame+1.jpg does not exist
    for i in range(0, 4):# num_frames-1):
        pimg_path = video_frames[i]
        cimg_path = video_frames[i+1]
        pbox = annot_frames[i]
        cbox = annot_frames[i+1]

        pimg = imread(pimg_path)
        cimg = imread(cimg_path)

        # pbox
        h, w, _ = pimg.shape
        cellx = 1. * w / POLICY_TEST['side']
        celly = 1. * h / POLICY_TEST['side']
        centerx = .5 * (pbox.x1 + pbox.x2)  # xmin, xmax
        centery = .5 * (pbox.y1 + pbox.y2)  # ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        pbox_xy = np.array([np.floor(cx), np.floor(cy)], dtype=np.int32)
        pbox_xy = np.reshape(pbox_xy, [1, 2])

        pimg_trans = pimg[..., [2, 1, 0]]
        cimg_trans = cimg[..., [2, 1, 0]]
        pimg_resize = imresize(pimg_trans, [POLICY_TEST['height'], POLICY_TEST['width'], 3], POLICY_TEST['interpolation'])
        cimg_resize = imresize(cimg_trans, [POLICY_TEST['height'], POLICY_TEST['width'], 3], POLICY_TEST['interpolation'])
        pimg_resize = (pimg_resize / 255.0).astype(np.float32)
        cimg_resize = (cimg_resize / 255.0).astype(np.float32)

        pimg_resize = np.expand_dims(pimg_resize, 0)
        cimg_resize = np.expand_dims(cimg_resize, 0)
        pbox_xy = np.expand_dims(pbox_xy, 0)

        feed_dict = {
            Tracker.pimg_resize: pimg_resize,
            Tracker.cimg_resize: cimg_resize,
            Tracker.pbox_xy: pbox_xy
        }

        start = time.time()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            adjusted_net_out = sess.run(Tracker.test_images_(POLICY=POLICY_TEST), feed_dict=feed_dict)
        end = time.time()

    top_obj_indexs = np.where(adjusted_net_out[..., 4] == np.max(adjusted_net_out[..., 4]))
    objectness_s = adjusted_net_out[top_obj_indexs][..., 4]

    for idx, objectness in np.ndenumerate(objectness_s):
        predict = adjusted_net_out[top_obj_indexs]
        pred_cx = (float(top_obj_indexs[1][idx]) + predict[idx][0]) / POLICY_TEST['width'] * w
        pred_cy = (float(top_obj_indexs[0][idx]) + predict[idx][1]) / POLICY_TEST['height'] * h
        pred_w = predict[idx][2] * w
        pred_h = predict[idx][3] * h
        pred_obj = predict[idx][4]

        pred_xl = int(pred_cx - pred_w / 2)
        pred_yl = int(pred_cy - pred_h / 2)
        pred_xr = int(pred_cx + pred_w / 2)
        pred_yr = int(pred_cy + pred_h / 2)

        if objectness > POLICY_TEST['thresh']:
            pred_cimg = cv2.rectangle(cimg, (pred_xl, pred_yl), (pred_xr, pred_yr), (0, 255, 0), 3)
            cv2.putText(pred_cimg, str(objectness), (pred_xl, pred_yl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imwrite('./result/' + cimg_path, pred_cimg)
            cv2.imwrite('./result/' + cimg_path.split('/')[-2] + "_" + cimg_path.split('/')[-1], pred_cimg)
            print(bcolors.WARNING + cimg_path + bcolors.ENDC)
            print(bcolors.WARNING + "Inference time {:3f}".format(end-start) + "    obj: " + str(objectness) + bcolors.ENDC)

        else:
            pred_cimg = cv2.rectangle(cimg, (pred_xl, pred_yl), (pred_xr, pred_yr), (0, 0, 255), 3)
            cv2.putText(pred_cimg, str(objectness), (pred_xl, pred_yl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.imwrite('./result/' + cimg_path, pred_cimg)
            cv2.imwrite('./result/' + cimg_path.split('/')[-2] + "_" + cimg_path.split('/')[-1], pred_cimg)
            print(bcolors.FAIL + cimg_path + bcolors.ENDC)
            print(bcolors.FAIL + "FAIL "+ "Inference time {:3f}".format(end-start) + "    obj: " + str(objectness) + bcolors.ENDC)