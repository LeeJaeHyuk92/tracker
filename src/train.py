from training_schedules import POLICY
from src.dataloader import load_batch
from model import net
# DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

Tracker = net(net_input_size=412)

# data loader
# TODO, dimension check
pimg_resize, cimg_resize, pbox_xy, \
confs, coord, areas, upleft, botright = load_batch(POLICY, 'train')

net.train(log_dir='.logs/Tracker',
          training_schedule=POLICY,
          pimg=pimg_resize,
          cimg=cimg_resize,
          pbox_xy=pbox_xy,
          confs=confs,
          coord=coord,
          areas=areas,
          upleft=upleft,
          botright=botright,
          ckpt=None,
          debug=True)