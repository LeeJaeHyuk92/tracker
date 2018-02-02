import tensorflow as tf
from training_schedules import POLICY
from loader.loader_vot import loader_vot
from model import net


DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'


Tracker = net(net_input_size=412)


# data upload & training

# data loader
objLoaderVot = loader_vot(DATA_PATH)
videos = objLoaderVot.get_videos()
videos_keys = videos.keys()


net.train(log_dir=,
          training_schedule=POLICY,
          pimg=,
          cimg=,
          gt=,
          ckpt=None,
          debug=True)