import tensorflow as tf
from .ops import conv_bn
import numpy as np

slim = tf.contrib.slim


def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))


# hyper param


# bulid model
# return: net out, feature(ROI)
class net:
    """ network """

    def __init__(self, net_input_size=412):
        self.global_step = slim.get_or_create_global_step()
        self.net_hw = net_input_size
        self.net_out_hw = net_input_size/32




    def model_conv(self, image, trainable=True):
        """
        build network before correlation
        conv6 width, height = origin/32
        """
        _, height, width, channel = image.shape.as_list()

        conv_1 = conv_bn(image, filters= 64, kernel=3, scope='conv1', trainable=trainable)
        pool_1 = slim.max_pool2d(conv_1, [2,2], 2, padding='SAME', scope='pool1')

        conv_2 = conv_bn(pool_1, filters= 128, kernel=3, scope='conv2', trainable=trainable)
        pool_2 = slim.max_pool2d(conv_2, [2, 2], 2, padding='SAME', scope='pool2')

        conv_3 = conv_bn(pool_2, filters= 256, kernel=3, scope='conv3', trainable=trainable)
        pool_3 = slim.max_pool2d(conv_3, [2, 2], 2, padding='SAME', scope='pool3')

        conv_4 = conv_bn(pool_3, filters= 512, kernel=3, scope='conv4', trainable=trainable)
        pool_4 = slim.max_pool2d(conv_4, [2, 2], 2, padding='SAME', scope='pool4')

        conv_5 = conv_bn(pool_4, filters= 64, kernel=3, scope='conv5', trainable=trainable)
        pool_5 = slim.max_pool2d(conv_5, [2, 2], 2, padding='SAME', scope='pool5')

        conv_6 = conv_bn(pool_5, filters= 64, kernel=3, scope='conv6', trainable=trainable)

        return conv_5, conv_6


    def model_pred(self, cimg_conv5, cimg_conv6, pimg_conv6, ROI_coordinate, trainable=True):
        """
        Predict final feature map
        :param ROI_coordinate: int, topleft, botright array index value
        :return net_out: [_, model_size/32, model_size/32, 5]
                       : object_score, tx, ty, tw, th ... YOLOv2
        """

        xl, yl, xr, yr = ROI_coordinate

        ROI_feature = pimg_conv6[:, xl:xr, yl:yr ,:]
        _, h, w, c = ROI_feature.shape.as_list()
        pool_avg = slim.avg_pool2d(ROI_feature, [h, w], scope='avg_pool')

        # TODO, correlation with high level feature
        # tf implementation https://github.com/jgorgenucsd/corr_tf
        # @tf.RegisterGradient("Correlation")
        # corr = correlation(cimg_conv6, pool_avg, ...)
        # pool_avg(pimg conv_6) should not be trainable
        correlation_conv5 = tf.nn.conv2d(cimg_conv5, pool_avg, strides=[1, 1, 1, 1], padding='SAME', name='correlation5')
        correlation_conv5 = tf.extract_image_patches(correlation_conv5,
                                                     ksize=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1],
                                                     rates=[1, 1, 1, 1],
                                                     padding='SAME')
        correlation_conv6 = tf.nn.conv2d(cimg_conv6, pool_avg, strides=[1, 1, 1, 1], padding='SAME', name='correlation6')
        correlation = tf.concat([correlation_conv5, correlation_conv6], axis=3, name='correlation')

        # TODO, FC or 1D conv if you want
        net_out = conv_bn(correlation, filters= 5, kernel=1, scope='conv_final', trainable=trainable)

        # TODO, get highest object score
        # softmax,

        # TODO, calculate box with ROI_coordinate

        return net_out


    def loss(self, net_out, anchor_selected, bbox_gt, objectscore_gt, scope='loss'):
        """
        training pair of images(distance can be more than 1)
        training sequence images
        :param net_out: network output feature (size/32)
        :param anchor_selected: anchor selected by prev object size(float64 0~1)
        :param bbox_gt: bbox coordinate in original size (float64 0~1)
        :param objectscore_gt: object score in net_out (float64 0~1)
        :return: total loss
        """

        # TODO, define parameter's type and code below

        # loss object score

        # loss bbox param by anchor in max obj score

        # otherwise ?

    def train(self, log_dir, training_schedule, pimg, cimg, gt, ckpt, debug=True):

        if debug:
            tf.summary.image("pimg", pimg, max_outputs=1)
            tf.summary.image("cimg", cimg, max_outputs=1)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = training_schedule['optimizer']['rmsprop'](
            learning_rate=self.learning_rate,
            decay=training_schedule['decay'],
            momentum=training_schedule['momentum'])

        # TODO, Is it OK just No trainable?
        _, pimg_conv6 = self.model_conv(pimg, trainable=False)
        cimg_conv5, cimg_conv6 = self.model_conv(cimg, trainable=True)

        # TODO, ROI_coordinate
        net_out = self.model_pred(self, cimg_conv5, cimg_conv6, pimg_conv6, ROI_coordinate, trainable=True)

        # loss
        total_loss = self.loss(net_out, gt)
        tf.summary.scalar('loss', total_loss)

        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        # checkpoint
        saver = tf.train.Saver()
        if ckpt:
            with tf.Session() as sess:
                saver.restore(sess, ckpt)
                print("Model restored.")
        else:
            print("Model will be trained from scratch")

        # TODO,  tensorboard; if you want to analyze tracking result
        # if debug:


        slim.learning.train(train_op, log_dir,
                            global_step=self.global_step,
                            save_summaries_secs=60,
                            number_of_steps=training_schedule['max_iter'],
                            save_interval_secs=600)

    def test(self, ckpt, pimg, cimg, etc...):

    def loss(self, net_out, gt):
        """
        from YOLOv2, link: https://github.com/thtrieu/darkflow
        """
        # meta
        m = self.meta
        sconf = float(m['object_scale'])
        snoob = float(m['noobject_scale'])
        scoor = float(m['coord_scale'])
        H, W, _ = m['out_size']
        B = m['num']
        HW = H * W  # number of grid cells
        anchors = m['anchors']

        print('{} loss hyper-parameters:'.format(m['model']))
        print('\tH       = {}'.format(H))
        print('\tW       = {}'.format(W))
        print('\tbox     = {}'.format(m['num']))
        print('\tclasses = {}'.format(m['classes']))
        print('\tscales  = {}'.format([sconf, snoob, scoor]))

        size1 = [None, HW, B, C]
        size2 = [None, HW, B]

        # return the below placeholders
        _probs = tf.placeholder(tf.float32, size1)
        _confs = tf.placeholder(tf.float32, size2)
        _coord = tf.placeholder(tf.float32, size2 + [4])
        # weights term for L2 loss
        _proid = tf.placeholder(tf.float32, size1)
        # material calculating IOU
        _areas = tf.placeholder(tf.float32, size2)
        _upleft = tf.placeholder(tf.float32, size2 + [2])
        _botright = tf.placeholder(tf.float32, size2 + [2])

        self.placeholders = {
            'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
            'areas': _areas, 'upleft': _upleft, 'botright': _botright
        }

        # Extract the coordinate prediction from net.out
        net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1)])
        coords = net_out_reshape[:, :, :, :, :4]
        coords = tf.reshape(coords, [-1, H * W, B, 4])
        adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
        adjusted_coords_wh = tf.sqrt(
            tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

        adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c], 3)

        wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]
        floor = centers - (wh * .5)
        ceil = centers + (wh * .5)

        # calculate the intersection areas
        intersect_upleft = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil, _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)

        # take care of the weight terms
        conid = snoob * (1. - confs) + sconf * confs
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
        cooid = scoor * weight_coo


        # self.fetch += [confs, conid, cooid]
        true = tf.concat([_coord, tf.expand_dims(confs, 3)], 3)
        wght = tf.concat([cooid, tf.expand_dims(conid, 3)], 3)

        loss = tf.pow(adjusted_net_out - true, 2)
        loss = tf.multiply(loss, wght)
        loss = tf.reshape(loss, [-1, H * W * B * (4 + 1)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        # tf.summary.scalar('{} loss'.format(m['model']), loss)

        return loss

