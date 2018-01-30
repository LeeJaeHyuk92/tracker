import tensorflow as tf
from .ops import conv_bn

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




    def model_conv(self, image, training_schedule, trainable=True):
        """
        build network before correlation
        conv6 width, height = origin/32
        """
        _, height, width, channel = image.shape.as_list()

        with tf.variable_scope('Tracker'):
            conv_1 = conv_bn(image, filters= 64, kernel=3, scope='conv1')
            pool_1 = slim.max_pool2d(conv_1, [2,2], 2, padding='SAME', scope='pool1')

            conv_2 = conv_bn(pool_1, filters= 128, kernel=3, scope='conv2')
            pool_2 = slim.max_pool2d(conv_2, [2, 2], 2, padding='SAME', scope='pool2')

            conv_3 = conv_bn(pool_2, filters= 256, kernel=3, scope='conv3')
            pool_3 = slim.max_pool2d(conv_2, [2, 2], 2, padding='SAME', scope='pool3')

            conv_4 = conv_bn(pool_3, filters= 512, kernel=3, scope='conv4')
            pool_4 = slim.max_pool2d(conv_2, [2, 2], 2, padding='SAME', scope='pool4')

            conv_5 = conv_bn(pool_4, filters= 256, kernel=3, scope='conv5')
            pool_5 = slim.max_pool2d(conv_2, [2, 2], 2, padding='SAME', scope='pool5')

            conv_6 = conv_bn(pool_5, filters= 64, kernel=3, scope='conv6')

        return conv_6

    def model_ROI(self, bbox, conv_6):
        """
        Extract ROI feature from pimg conv_6 by pimg bbox
        :return : ROI feature
        :return : ROI coordinate
        """

        bbox = bbox / 32

        # TODO, refine bbox for extracting ROI
        bbox = bbox blabla
        ROI_feature = tf.extract_image_patches(conv_6)

        return ROI_feature, ROI coordinate

    def model_pred(self, cimg_conv6, ROI_feature, ROI_coordinate):
        """
        Predict final feature map
        """

        # TODO, average pooling from ROI_feature


        # TODO, correlation with high level feature
        # tf implementation https://github.com/jgorgenucsd/corr_tf
        # @tf.RegisterGradient("Correlation")
        corr = correlation(cimg_conv6, ROI_feature, ...)

        # TODO, FC or 1D conv if you want

        # TODO, get highest object score
        # softmax,

        # TODO, calculate box with ROI_coordinate





    def loss(self, feature):


    def train(self, log_dir, training_schedule, pimg, cimg, gt, ckpt=None):


    def test(self, ckpt, pimg, cimg, etc...):