from training_schedules import POLICY
from src.dataloader import load_batch
from src.utils import schedule_verbose
from model import net
# DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

# check POLICY
# check load_batch, train, sample, test
# check log_dir(slim get latest ckpt)

Tracker = net()
schedule_verbose(POLICY)

# data loader
# TODO, dimension check
pimg_resize, cimg_resize, pbox_xy, \
confs, coord, areas, upleft, botright = load_batch(POLICY, 'sample')

Tracker.train(log_dir='./logs/Tracker',
              training_schedule=POLICY,
              pimg_resize=pimg_resize,
              cimg_resize=cimg_resize,
              pbox_xy=pbox_xy,
              confs=confs,
              coord=coord,
              areas=areas,
              upleft=upleft,
              botright=botright,
              ckpt=None,
              debug=True)