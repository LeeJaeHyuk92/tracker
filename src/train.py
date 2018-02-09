from training_schedules import POLICY
from src.dataloader import load_batch
from src.utils import schedule_verbose, ckpt_reader
from model import net
# DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

# check POLICY
# check load_batch, train, sample, test
# check log_dir(slim get latest ckpt)

# ckpt = './logs/Tracker/model.ckpt-1043223'
ckpt = None
log_dir = './logs/Tracker/'
data_type = 'train'

Tracker = net()
schedule_verbose(POLICY, data_type)

# data loader
# TODO, dimension check
pimg_resize, cimg_resize, pbox_xy, \
confs, coord, areas, upleft, botright = load_batch(POLICY, data_type)

# check ckpt variable
if ckpt:
    ckpt_reader(ckpt, value=False)

Tracker.train(log_dir=log_dir,
              training_schedule=POLICY,
              pimg_resize=pimg_resize,
              cimg_resize=cimg_resize,
              pbox_xy=pbox_xy,
              confs=confs,
              coord=coord,
              areas=areas,
              upleft=upleft,
              botright=botright,
              ckpt=ckpt,
              debug=True)