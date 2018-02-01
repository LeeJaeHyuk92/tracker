import tensorflow as tf
from .training_schedules import POLICY
from .model import net


Tracker = net(net_input_size=412)



optimizer = tf.train.RMSPropOptimizer
# data upload & training


net.train(log_dir=,
          training_schedule=POLICY,
          pimg=,
          cimg=,
          gt=,
          ckpt=None,
          debug=True)