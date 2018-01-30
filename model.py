import tensorflow as tf
slim = tf.contrib.slim



# hyper param


# bulid model
# return: net out, feature(ROI)
class net:
    """ network """

    def __init__(self, mode, debug=False):
        self.mode = mode
        self.debug = debug
        self.global_step = slim.get_or_create_global_step()




    def model(self, pimg, cimg, training_schedule, trainable=True):
        _, height, width, channel = pimg.shape.as_list()

        # build network
        with tf.variable_scope('Tracker'):


    def loss(self, feature):


    def train(self, log_dir, training_schedule, pimg, cimg, gt, ckpt=None):


    def test(self, ckpt, pimg, cimg, etc...):