from training_schedules import POLICY
from src.dataloader import load_batch
from progressbar import ProgressBar, Percentage, Bar
from scipy.misc import imread, imresize
from loader.loader_vot import loader_vot
import os
import argparse
from model import net
# DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

ckpt = './logs/180208_reduce_mean/model.ckpt-122414'
data_dir = './data/vot2015'

# Verify arguments are valid
if not os.path.exists(ckpt + '.meta'):
    raise ValueError('ckpt does not exist')
if not os.path.isdir(data_dir):
    raise ValueError('data_dir must exist and be a directory')

Tracker = net()

# Load each data sample (pimg, cimg, pbox, cbox) and write it to the TFRecord
objLoaderVot = loader_vot(data_dir)
videos = objLoaderVot.get_videos()
video_keys = videos.keys()

for idx in range(len(videos)):
    video_frames = videos[video_keys[idx]][0]         # ex) bag/*.jpg list
    annot_frames = videos[video_keys[idx]][1]         # ex) bag/groundtruth, rectangle box info
    num_frames = min(len(video_frames), len(annot_frames))

    # num_frame+1.jpg does not exist
    for i in range(0, num_frames-1):
        pimg_path = video_frames[i]
        cimg_path = video_frames[i+1]
        pbox = annot_frames[i]
        cbox = annot_frames[i+1]

        out = Tracker.test_images(ckpt=ckpt,
                                  pimg_path=pimg_path,
                                  cimg_path=cimg_path,
                                  POLICY=POLICY,
                                  pbox=pbox)
