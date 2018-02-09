from training_schedules import POLICY_TEST
from src.dataloader import load_batch
from progressbar import ProgressBar, Percentage, Bar
from scipy.misc import imread, imresize
from loader.loader_vot import loader_vot
import os
import argparse
from model import net
# DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

ckpt = './logs/180209_short_vot2015/model.ckpt-15311'
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

# Load each data sample (pimg, cimg, pbox, cbox) and write it to the TFRecord
objLoaderVot = loader_vot(data_dir)
videos = objLoaderVot.get_videos()
video_keys = videos.keys()

once = True
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


        if once:
            out = Tracker.test_images(ckpt=ckpt,
                                      pimg_path=pimg_path,
                                      cimg_path=cimg_path,
                                      POLICY=POLICY_TEST,
                                      pbox=pbox,
                                      reuse=False)
            once = False

        else:
            out = Tracker.test_images(ckpt=ckpt,
                                      pimg_path=pimg_path,
                                      cimg_path=cimg_path,
                                      POLICY=POLICY_TEST,
                                      pbox=pbox,
                                      reuse=True)
