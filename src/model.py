import tensorflow as tf
from .ops import conv_bn, conv_linear
import numpy as np
from scipy.misc import imread, imresize
from src.utils import bcolors, calculate_box_tf
import time
import cv2
from tensorflow.python import debug as tf_debug

slim = tf.contrib.slim


def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))


def offset_map(net_out):
    offset = np.array([np.arange(13)] * 13 * self.num_box)		# [13*5 ,13]
    offset = np.reshape(offset, (self.num_box, 13, 13))			# [5, 13, 13]
    offset = np.transpose(offset, (1, 2, 0))					# [13, 13, 5]
    offset = tf.constant(offset, dtype=tf.float32)
    offset = tf.reshape(offset, [1, 13, 13, self.num_box])
    offset = tf.tile(offset, [tf.shape(net_out)[0], 1, 1, 1])	# [None, 13, 13, 5]
    offset = tf.div(1. ,offset)
    return offset


# hyper param


# bulid model
# return: net out, feature(ROI)
class net:
    """ network """

    def __init__(self):
        # self.global_step = slim.get_or_create_global_step()
        self.global_step = tf.train.get_or_create_global_step()

    def build_model_test(self, POLICY):

        self.pimg_resize = tf.placeholder(dtype=tf.float32, shape=[None, POLICY['height'], POLICY['width'], 3])
        self.cimg_resize = tf.placeholder(dtype=tf.float32, shape=[None, POLICY['height'], POLICY['width'], 3])
        self.pbox_xy = tf.placeholder(dtype=tf.int32, shape=[None, 1, 2])
        _, pimg_conv6 = self.model_conv(self.pimg_resize, trainable=False, reuse=False)
        cimg_conv5, cimg_conv6 = self.model_conv(self.cimg_resize, trainable=False, reuse=True)
        self.net_out = self.model_pred(cimg_conv5, cimg_conv6, pimg_conv6, self.pbox_xy, None, POLICY, roi_pool=None, trainable=False, reuse=False)

    def model_conv(self, image, trainable=True, reuse=False):
        """
        build network before correlation
        conv6 width, height = origin/32
        """

        _, height, width, channel = image.shape.as_list()

        with slim.arg_scope([slim.conv2d],
                            reuse=reuse):
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

    def model_pred(self, cimg_conv5, cimg_conv6, pimg_conv6, ROI_coordinate, pROI, POLICY, roi_pool=True, trainable=True, reuse=False, test=False):
        """
        Predict final feature map
        :param ROI_coordinate: pbox_xy
        :return net_out: [_, model_size/32, model_size/32, 5]
                       : tx, ty, tw, th, object_score ... YOLOv2
        """

        if roi_pool:
            from roi_pooling.roi_pooling_ops import roi_pooling
            pROI = tf.expand_dims(pROI, 0)
            pROI = tf.expand_dims(pROI, 0)
            rois = [[0, pROI[0, 0, 0], pROI[0, 0, 1], pROI[0, 0, 2], pROI[0, 0, 3]]]

            # roi_pooling -> batch, wl, hl, wr, hr list
            ROI_feature = roi_pooling(pimg_conv6, rois, pool_height=1, pool_width=1)
            ROI_feature = tf.transpose(ROI_feature, perm=[0, 3, 2, 1])


        else:
            # crop[:, xl:xr, yl:yr, :]
            # xl, yl, xr, yr = ROI_coordinate,
            # using first' frames coordinate
            xl = ROI_coordinate[0, 0, 0]
            yl = ROI_coordinate[0, 0, 1]

            # TODO, ROI 1x1 region
            xr = xl+1
            yr = yl+1

            ROI_feature = pimg_conv6[:, yl:yr, xl:xr, :]

        # TODO, if ROI is not 1x1, modify this region
        #_, h, w, c = ROI_feature.shape.as_list()
        #pool_avg = slim.avg_pool2d(ROI_feature, [h, w], scope='avg_pool')

        # TODO, correlation with high level feature
        # tf implementation https://github.com/jgorgenucsd/corr_tf
        # @tf.RegisterGradient("Correlation")
        # corr = correlation(cimg_conv6, pool_avg, ...)
        # pool_avg(pimg conv_6) should not be trainable

        # correlation_conv5 = tf.nn.conv2d(cimg_conv5, ROI_feature, strides=[1, 1, 1, 1], padding='SAME', name='correlation5')
        # correlation_conv5 = tf.extract_image_patches(correlation_conv5,
        #                                              ksizes=[1, 2, 2, 1],
        #                                              strides=[1, 2, 2, 1],
        #                                              rates=[1, 1, 1, 1],
        #                                              padding='SAME')
        # correlation_conv6 = tf.nn.conv2d(cimg_conv6, ROI_feature, strides=[1, 1, 1, 1], padding='SAME', name='correlation6')
        # correlation = tf.concat([correlation_conv5, correlation_conv6], axis=3, name='correlation')

        correlation_conv5 = tf.multiply(cimg_conv5, ROI_feature)
        correlation_conv5 = tf.reduce_mean(correlation_conv5, axis=3, keep_dims=True, name='correlation5')
        correlation_conv5 = tf.extract_image_patches(correlation_conv5,
                                                     ksizes=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1],
                                                     rates=[1, 1, 1, 1],
                                                     padding='SAME')
        correlation_conv6 = tf.multiply(cimg_conv6, ROI_feature)
        correlation_conv6 = tf.reduce_mean(correlation_conv6, axis=3, keep_dims=True, name='correlation6')
        self.correlation = tf.concat([correlation_conv5, correlation_conv6], axis=3, name='correlation')
        tf.summary.image("correlation_0", self.correlation[:, :, :, 0:1], max_outputs=2)
        tf.summary.image("correlation_1", self.correlation[:, :, :, 1:2], max_outputs=2)

        # TODO, FC or 1D conv if you want
        # for test
        with slim.arg_scope([slim.conv2d],
                            reuse=reuse):
            net_out = conv_linear(self.correlation, filters=5, kernel=1, scope='conv_final', trainable=trainable)
        tf.summary.image("objectness", self.correlation[:, :, :, 4:], max_outputs=2)

        # TODO, get highest object score
        # softmax,

        # TODO, calculate box with ROI_coordinate

        return net_out


    def model_pred_train(self, cimg_conv5, cimg_conv6, pimg_conv6, ROI_coordinate, POLICY, trainable=True, test=False):
        """
        Predict final feature map
        :param ROI_coordinate: pbox_xy
        :return net_out: [_, model_size/32, model_size/32, 5]
                       : tx, ty, tw, th, object_score ... YOLOv2
        """

        # crop[:, xl:xr, yl:yr, :]
        # xl, yl, xr, yr = ROI_coordinate,
        # using first' frames coordinate


        # TODO, FIX, spaghetti code -_-  for 4 batch size
        xl_0 = ROI_coordinate[0, 0, 0]
        yl_0 = ROI_coordinate[0, 0, 1]
        xl_1 = ROI_coordinate[1, 0, 0]
        yl_1 = ROI_coordinate[1, 0, 1]
        xl_2 = ROI_coordinate[2, 0, 0]
        yl_2 = ROI_coordinate[2, 0, 1]
        xl_3 = ROI_coordinate[3, 0, 0]
        yl_3 = ROI_coordinate[3, 0, 1]

        # TODO, FIX,spaghetti code -_- ROI 1x1 region
        ROI_feature = tf.concat([pimg_conv6[0:1, yl_0:yl_0 + 1, xl_0:xl_0 + 1, :],
                                 pimg_conv6[1:2, yl_1:yl_1 + 1, xl_1:xl_1 + 1, :],
                                 pimg_conv6[2:3, yl_2:yl_2 + 1, xl_2:xl_2 + 1, :],
                                 pimg_conv6[3:4, yl_3:yl_3 + 1, xl_3:xl_3 + 1, :]],
                                axis=0)

        # TODO, if ROI is not 1x1, modify this region
        #_, h, w, c = ROI_feature.shape.as_list()
        #pool_avg = slim.avg_pool2d(ROI_feature, [h, w], scope='avg_pool')

        # TODO, correlation with high level feature
        # tf implementation https://github.com/jgorgenucsd/corr_tf
        # @tf.RegisterGradient("Correlation")
        # corr = correlation(cimg_conv6, pool_avg, ...)
        # pool_avg(pimg conv_6) should not be trainable

        # correlation_conv5 = tf.nn.conv2d(cimg_conv5, ROI_feature, strides=[1, 1, 1, 1], padding='SAME', name='correlation5')
        # correlation_conv5 = tf.extract_image_patches(correlation_conv5,
        #                                              ksizes=[1, 2, 2, 1],
        #                                              strides=[1, 2, 2, 1],
        #                                              rates=[1, 1, 1, 1],
        #                                              padding='SAME')
        # correlation_conv6 = tf.nn.conv2d(cimg_conv6, ROI_feature, strides=[1, 1, 1, 1], padding='SAME', name='correlation6')
        # correlation = tf.concat([correlation_conv5, correlation_conv6], axis=3, name='correlation')

        correlation_conv5 = tf.multiply(cimg_conv5, ROI_feature)
        correlation_conv5 = tf.reduce_mean(correlation_conv5, axis=3, keep_dims=True, name='correlation5')
        correlation_conv5 = tf.extract_image_patches(correlation_conv5,
                                                     ksizes=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1],
                                                     rates=[1, 1, 1, 1],
                                                     padding='SAME')
        correlation_conv6 = tf.multiply(cimg_conv6, ROI_feature)
        correlation_conv6 = tf.reduce_mean(correlation_conv6, axis=3, keep_dims=True, name='correlation6')
        correlation = tf.concat([correlation_conv5, correlation_conv6], axis=3, name='correlation')
        tf.summary.image("correlation_0", correlation[:, :, :, 0:1], max_outputs=2)
        tf.summary.image("correlation_1", correlation[:, :, :, 1:2], max_outputs=2)

        # TODO, FC or 1D conv if you want
        net_out = conv_linear(correlation, filters=5, kernel=1, scope='conv_final', trainable=trainable)
        tf.summary.image("objectness", net_out[:, :, :, 4:], max_outputs=2)

        # TODO, get highest object score
        # softmax,

        # TODO, calculate box with ROI_coordinate

        return net_out

    def model_pred_train_ROI(self, cimg_conv5, cimg_conv6, pimg_conv6, pROI, POLICY, trainable=True, test=False):
        """
        Predict final feature map
        :param pROI: pROI
        :return net_out: [_, model_size/32, model_size/32, 5]
                       : tx, ty, tw, th, object_score ... YOLOv2
        """

        from roi_pooling.roi_pooling_ops import roi_pooling
        # crop[:, xl:xr, yl:yr, :]
        # xl, yl, xr, yr = ROI_coordinate,
        # using first' frames coordinate

        # W, H  for 4 batch size
        # batch, Wl, Hl, WR, HR
        # rois = []
        # rois.append(np.append([0], pROI[0][0]))
        # rois.append(np.append([1], pROI[1][1]))
        # rois.append(np.append([2], pROI[2][2]))
        # rois.append(np.append([3], pROI[3][3]))

        print(pROI.shape)
        rois = [[0, pROI[0, 0, 0], pROI[0, 0, 1], pROI[0, 0, 2], pROI[0, 0, 3]],
                [1, pROI[1, 0, 0], pROI[1, 0, 1], pROI[1, 0, 2], pROI[1, 0, 3]],
                [2, pROI[2, 0, 0], pROI[2, 0, 1], pROI[2, 0, 2], pROI[2, 0, 3]],
                [3, pROI[3, 0, 0], pROI[3, 0, 1], pROI[3, 0, 2], pROI[3, 0, 3]]]

        # roi_pooling -> batch, wl, hl, wr, hr list
        ROI_feature = roi_pooling(pimg_conv6, rois, pool_height=1, pool_width=1)
        ROI_feature = tf.transpose(ROI_feature, perm=[0, 3, 2, 1])

        correlation_conv5 = tf.multiply(cimg_conv5, ROI_feature)
        correlation_conv5 = tf.reduce_mean(correlation_conv5, axis=3, keep_dims=True, name='correlation5')
        correlation_conv5 = tf.extract_image_patches(correlation_conv5,
                                                     ksizes=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1],
                                                     rates=[1, 1, 1, 1],
                                                     padding='SAME')
        correlation_conv6 = tf.multiply(cimg_conv6, ROI_feature)
        correlation_conv6 = tf.reduce_mean(correlation_conv6, axis=3, keep_dims=True, name='correlation6')
        correlation = tf.concat([correlation_conv5, correlation_conv6], axis=3, name='correlation')
        tf.summary.image("correlation_0", correlation[:, :, :, 0:1], max_outputs=2)
        tf.summary.image("correlation_1", correlation[:, :, :, 1:2], max_outputs=2)

        # TODO, FC or 1D conv if you want
        net_out = conv_linear(correlation, filters=5, kernel=1, scope='conv_final', trainable=trainable)
        tf.summary.image("objectness", net_out[:, :, :, 4:], max_outputs=2)

        # TODO, get highest object score
        # softmax,

        # TODO, calculate box with ROI_coordinate

        return net_out

    def model_pred_train_ROI_ori(self, cimg_conv5, cimg_conv6, pimg_conv6, pROI, POLICY, trainable=True, test=False):
        """
        Predict final feature map
        :param pROI: pROI
        :return net_out: [_, model_size/32, model_size/32, 5]
                       : tx, ty, tw, th, object_score ... YOLOv2
        """

        from roi_pooling.roi_pooling_ops import roi_pooling

        # W, H  for 4 batch size
        # batch, Wl, Hl, WR, HR
        rois = [[0, pROI[0, 0, 0], pROI[0, 0, 1], pROI[0, 0, 2], pROI[0, 0, 3]],
                [1, pROI[1, 0, 0], pROI[1, 0, 1], pROI[1, 0, 2], pROI[1, 0, 3]],
                [2, pROI[2, 0, 0], pROI[2, 0, 1], pROI[2, 0, 2], pROI[2, 0, 3]],
                [3, pROI[3, 0, 0], pROI[3, 0, 1], pROI[3, 0, 2], pROI[3, 0, 3]]]

        # roi_pooling -> batch, wl, hl, wr, hr list
        ROI_feature = roi_pooling(pimg_conv6, rois, pool_height=3, pool_width=3)
        cimg_conv5 = tf.extract_image_patches(cimg_conv5,
                                              ksizes=[1, 2, 2, 1],
                                              strides=[1, 2, 2, 1],
                                              rates=[1, 1, 1, 1],
                                              padding='SAME')
        cimg_conv5_0, cimg_conv5_1, cimg_conv5_2, cimg_conv5_3= tf.split(cimg_conv5, 4, axis=3)

        cimg_concat = tf.stack([cimg_conv5_0,
                                cimg_conv5_1,
                                cimg_conv5_2,
                                cimg_conv5_3,
                                cimg_conv6], axis=1)           # [batch, 5, H, W, depth]


        # for 4 batch size
        ROI_feature = tf.transpose(ROI_feature, perm=[0, 3, 2, 1])      # batch, height, width, in_channels
        ROI_feature = tf.expand_dims(ROI_feature, axis=4)               # 1 out_channel
        correlation0 = tf.nn.conv2d(cimg_concat[0, :, :, :, :], ROI_feature[0, :, :, :, :], [1, 1, 1, 1], padding='SAME')
        correlation1 = tf.nn.conv2d(cimg_concat[1, :, :, :, :], ROI_feature[1, :, :, :, :], [1, 1, 1, 1], padding='SAME')
        correlation2 = tf.nn.conv2d(cimg_concat[2, :, :, :, :], ROI_feature[2, :, :, :, :], [1, 1, 1, 1], padding='SAME')
        correlation3 = tf.nn.conv2d(cimg_concat[3, :, :, :, :], ROI_feature[3, :, :, :, :], [1, 1, 1, 1], padding='SAME')

        correlation = tf.stack([correlation0, correlation1, correlation2, correlation3], axis=0)    # [batch, 5, H, W, 1]
        correlation = tf.transpose(correlation, perm=[0, 2, 3, 1, 4])
        correlation = correlation[..., 0]

        tf.summary.image("correlation_0", correlation[:, :, :, 0:1], max_outputs=2)
        tf.summary.image("correlation_1", correlation[:, :, :, 1:2], max_outputs=2)
        # tf.summary.text('pROI', pROI)
        # TODO, FC or 1D conv if you want
        net_out = conv_linear(correlation, filters=5, kernel=1, scope='conv_final', trainable=trainable)
        tf.summary.image("objectness", net_out[:, :, :, 4:], max_outputs=2)

        # TODO, get highest object score
        # softmax,

        # TODO, calculate box with ROI_coordinate

        return net_out

    def train(self, log_dir, training_schedule, pimg_resize, cimg_resize,
              pbox_xy, pROI, pROI_anchor, confs, coord, areas, upleft, botright, ckpt, debug=True):

        if debug:
            tf.summary.image("pimg", pimg_resize, max_outputs=2)
            tf.summary.image("cimg", cimg_resize, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        # optimizer = training_schedule['optimizer']['rmsprop'](
        #     learning_rate=self.learning_rate,
        #     decay=training_schedule['decay'],
        #     epsilon=1e-8,
        #     momentum=training_schedule['momentum'])

        optimizer = training_schedule['optimizer']['adam'](
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])


        # TODO, Is it OK just No trainable?
        _, pimg_conv6 = self.model_conv(pimg_resize, trainable=True, reuse=False)
        cimg_conv5, cimg_conv6 = self.model_conv(cimg_resize, trainable=True, reuse=True)

        # TODO, ROI_coordinate
        # net_out = self.model_pred_train(cimg_conv5, cimg_conv6, pimg_conv6, pbox_xy, POLICY=training_schedule, trainable=True)
        net_out = self.model_pred_train_ROI(cimg_conv5, cimg_conv6, pimg_conv6, pROI, POLICY=training_schedule, trainable=True)


        # TODO, summary box
        # calculate_box_tf(cimg_resize, net_out, training_schedule)

        # loss
        total_loss = self.loss(net_out, pROI_anchor, confs, coord, areas, upleft, botright, training_schedule=training_schedule)
        tf.summary.scalar('loss', total_loss)

        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        # checkpoint
        saver = tf.train.Saver()
        if ckpt:
            with tf.Session() as sess:
                saver.restore(sess, ckpt)
                print(bcolors.WARNING + "Model restored." + bcolors.ENDC)
        else:
            print(bcolors.WARNING + "Model will be trained from scratch" + bcolors.ENDC)

        # TODO,  tensorboard; if you want to analyze tracking result
        # if debug:

        slim.learning.train(train_op, log_dir,
                            global_step=self.global_step,
                            save_summaries_secs=60,
                            number_of_steps=training_schedule['max_iter'],
                            save_interval_secs=600)
                            # session_wrapper=tf_debug.LocalCLIDebugWrapperSession)

    def test_sequence(self, ckpt, pimg_path, cimg_path, POLICY, pbox, out_path, video_play=True, save_image=False):

        H, W = POLICY['side'], POLICY['side']
        B = POLICY['num']
        HW = H * W  # number of grid cells
        anchors = POLICY['anchors']

        objLoaderVot = loader_vot(FLAGS.data_dir)
        videos = objLoaderVot.get_videos()
        video_keys = videos.keys()

        for idx in range(len(videos)):
            video_frames = videos[video_keys[idx]][0]  # ex) bag/*.jpg list
            annot_frames = videos[video_keys[idx]][1]  # ex) bag/groundtruth, rectangle box info
            num_frames = min(len(video_frames), len(annot_frames))

            # num_frame+1.jpg does not exist
            for i in range(0, num_frames - 1):
                pimg_path = video_frames[i]
                cimg_path = video_frames[i + 1]
                pbox = annot_frames[i]
                cbox = annot_frames[i + 1]

                # pbox
                h, w, _ = pbox.shape
                cellx = 1. * w / POLICY['side']
                celly = 1. * h / POLICY['side']
                centerx = .5 * (pbox.x1 + pbox.x2)  # xmin, xmax
                centery = .5 * (pbox.y1 + pbox.y2)  # ymin, ymax
                cx = centerx / cellx
                cy = centery / celly
                pbox_xy = np.array([np.floor(cx), np.floor(cy)], dtype=np.int32)

                pimg = imread(pimg_path)
                cimg = imread(cimg_path)
                pimg = pimg[..., [2, 1, 0]]
                cimg = cimg[..., [2, 1, 0]]
                pimg_resize = imresize(pimg, [POLICY['height'], POLICY['width'], 3], POLICY['interpolation'])
                cimg_resize = imresize(cimg, [POLICY['height'], POLICY['width'], 3], POLICY['interpolation'])

                pimg_resize = pimg_resize / 255.0
                cimg_resize = cimg_resize / 255.0

                # for network input & extension for batch loader
                pimg_resize = tf.expand_dims(pimg_resize, 0)
                cimg_resize = tf.expand_dims(cimg_resize, 0)

                _, pimg_conv6 = self.model_conv(pimg_resize, trainable=True, reuse=False)
                cimg_conv5, cimg_conv6 = self.model_conv(cimg_resize, trainable=True, reuse=True)
                net_out = self.model_pred(cimg_conv5, cimg_conv6, pimg_conv6, pbox_xy, trainable=True)

                self.net_out

                # calculate box
                net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1)])
                coords = net_out_reshape[:, :, :, :, :4]
                coords = tf.reshape(coords, [-1, H * W, B, 4])
                adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
                adjusted_coords_wh = tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape(
                    [W, H], [1, 1, 1, 2])

                adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
                adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

                adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c], 3)

                if i==0:
                    saver = tf.train.Saver()
                    with tf.Session as sess:
                        saver.restore(sess, ckpt)
                        adjusted_net_out = sess.run(adjusted_net_out)        # [batch, HW, B, 5]
                else:
                    with tf.Session as sess:
                        adjusted_net_out = sess.run(adjusted_net_out)


    def test_images(self, ckpt, pimg_path, cimg_path, POLICY, pbox, first_frame=True, reuse=False): #, out_path, video_play=True, save_image=False):
        '''
        image [h, w, c] no batch
        :param reuse: model_conv reuse for test
        '''

        H, W = POLICY['side'], POLICY['side']
        B = POLICY['num']
        HW = H * W  # number of grid cells

        anchors = POLICY['anchors']
        # pROI_w = (pbox.x2 - pbox.x1) / W
        # pROI_h = (pbox.y2 - pbox.y1) / H
        # pROI_anchor = np.array([pROI_w, pROI_h], dtype=np.float32)



        pimg = imread(pimg_path)
        cimg = imread(cimg_path)

        # pbox
        h, w, _ = pimg.shape
        cellx = 1. * w / POLICY['side']
        celly = 1. * h / POLICY['side']
        centerx = .5 * (pbox.x1 + pbox.x2)  # xmin, xmax
        centery = .5 * (pbox.y1 + pbox.y2)  # ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        pbox_xy = np.array([np.floor(cx), np.floor(cy)], dtype=np.int32)
        pbox_xy = np.reshape(pbox_xy, [1, 2])
        pbox_xy = tf.expand_dims(pbox_xy, 0)

        pROI = np.array([pbox.x1 / w * W, pbox.y1 / h * H, pbox.x2 / w * W, pbox.y2 / h * H], dtype=np.float32)
        pROI = np.floor(pROI).astype(np.int32)

        pimg_trans = pimg[..., [2, 1, 0]]
        cimg_trans = cimg[..., [2, 1, 0]]
        pimg_resize = imresize(pimg_trans, [POLICY['height'], POLICY['width'], 3], POLICY['interpolation'])
        cimg_resize = imresize(cimg_trans, [POLICY['height'], POLICY['width'], 3], POLICY['interpolation'])

        pimg_resize = tf.to_float(pimg_resize / 255.0)
        cimg_resize = tf.to_float(cimg_resize / 255.0)

        pimg_resize = tf.expand_dims(pimg_resize, 0)
        cimg_resize = tf.expand_dims(cimg_resize, 0)

        _, pimg_conv6 = self.model_conv(pimg_resize, trainable=False, reuse=reuse)
        cimg_conv5, cimg_conv6 = self.model_conv(cimg_resize, trainable=False, reuse=True)
        net_out = self.model_pred(cimg_conv5, cimg_conv6, pimg_conv6, pbox_xy, pROI, POLICY, roi_pool=True, trainable=False, reuse=reuse)

        # calculate box
        net_out_reshape = tf.reshape(net_out, [H, W, B, (4 + 1)])
        coords = net_out_reshape[:, :, :, :4]
        adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
        adjusted_coords_wh = tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])

        adjusted_c = expit_tensor(net_out_reshape[:, :, :, 4:])

        adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c], 3)

        # TODO, FIX, graph g
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            start = time.time()
            adjusted_net_out = sess.run(adjusted_net_out)        # [H, W, B, 5]
            correlation = sess.run(self.correlation)
            end = time.time()

        mask = np.zeros(adjusted_net_out[..., 4].shape)
        for idx in [-1, 0, 1]:
            if (int(np.floor(cx)) + idx > H-1) or (int(np.floor(cx)) + idx < 0):
                continue
            for idy in [-1, 0, 1]:
                if (int(np.floor(cy)) + idy > H-1) or (int(np.floor(cy)) + idy < 0):
                    continue
                mask[int(np.floor(cy)) + idy, int(np.floor(cx)) + idx, :] = 1.
        adjusted_net_out[..., 4] = adjusted_net_out[..., 4] * mask
        # for mask
        # top_obj_indexs = np.where(adjusted_net_out[..., 4] == np.max(adjusted_net_out[..., 4]) * mask)
        # top_obj_indexs = np.where(adjusted_net_out[..., 4] == np.max(adjusted_net_out[..., 4]))
        top_obj_indexs = np.where(adjusted_net_out[..., 4] > POLICY['thresh'])

        objectness_s = adjusted_net_out[top_obj_indexs][..., 4]

        for idx, objectness in np.ndenumerate(objectness_s):
            predict = adjusted_net_out[top_obj_indexs]
            pred_cx = (float(top_obj_indexs[1][idx]) + predict[idx][0]) / W * w
            pred_cy = (float(top_obj_indexs[0][idx]) + predict[idx][1]) / H * h
            pred_w = predict[idx][2] * w
            pred_h = predict[idx][3] * h
            pred_obj = predict[idx][4]

            pred_xl = int(pred_cx - pred_w / 2)
            pred_yl = int(pred_cy - pred_h / 2)
            pred_xr = int(pred_cx + pred_w / 2)
            pred_yr = int(pred_cy + pred_h / 2)

            if objectness > POLICY['thresh']:
                pred_cimg = cv2.rectangle(cimg, (pred_xl, pred_yl), (pred_xr, pred_yr), (0, 255, 0), 3)
                cv2.putText(pred_cimg, str(objectness), (pred_xl, pred_yl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.imwrite('./result/' + cimg_path, pred_cimg)
                cv2.imwrite('./result/' + cimg_path.split('/')[-2] + "_" + cimg_path.split('/')[-1], pred_cimg)
                print(bcolors.WARNING + cimg_path + bcolors.ENDC)
                print(bcolors.WARNING + "Inference time {:3f}".format(end-start) + "    obj: " + str(objectness) + bcolors.ENDC)

            else:
                pred_cimg = cv2.rectangle(cimg, (pred_xl, pred_yl), (pred_xr, pred_yr), (0, 0, 255), 3)
                cv2.putText(pred_cimg, str(objectness), (pred_xl, pred_yl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # cv2.imwrite('./result/' + cimg_path, pred_cimg)
                cv2.imwrite('./result/' + cimg_path.split('/')[-2] + "_" + cimg_path.split('/')[-1], pred_cimg)
                print(bcolors.FAIL + cimg_path + bcolors.ENDC)
                print(bcolors.FAIL + "FAIL "+ "Inference time {:3f}".format(end-start) + "    obj: " + str(objectness) + bcolors.ENDC)

        return adjusted_net_out


    def test_images_(self, POLICY): #, out_path, video_play=True, save_image=False):
        '''
        image [h, w, c] no batch
        :param reuse: model_conv reuse for test
        '''

        H, W = POLICY['side'], POLICY['side']
        B = POLICY['num']
        HW = H * W  # number of grid cells
        anchors = POLICY['anchors']

        # calculate box
        net_out_reshape = tf.reshape(self.net_out, [H, W, B, (4 + 1)])
        coords = net_out_reshape[:, :, :, :4]
        adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
        adjusted_coords_wh = tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])
        adjusted_c = expit_tensor(net_out_reshape[:, :, :, 4:])
        adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c], 3)

        return adjusted_net_out

    def loss(self, net_out, pROI_anchor, _confs, _coord, _areas, _upleft, _botright, training_schedule):
        """
        from YOLOv2, link: https://github.com/thtrieu/darkflow
        """
        # meta
        m = training_schedule
        sconf = float(m['object_scale'])
        snoob = float(m['noobject_scale'])
        scoor = float(m['coord_scale'])
        H, W = m['side'], m['side']
        B = m['num']
        HW = H * W  # number of grid cells
        anchors = m['anchors']


        # Extract the coordinate prediction from net.out
        net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1)])
        coords = net_out_reshape[:, :, :, :, :4]
        coords = tf.reshape(coords, [-1, H * W, B, 4])
        adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
        adjusted_coords_wh = tf.sqrt(
                 tf.exp(coords[:, :, :, 2:4]) * tf.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]) + 1e-8)

        # for prev anchor
        # adjusted_coords_wh_0 = tf.sqrt(
        #     tf.exp(coords[0:1, :, :, 2:4]) * tf.reshape(pROI_anchor[0, :], [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]) + 1e-8)
        # adjusted_coords_wh_1 = tf.sqrt(
        #     tf.exp(coords[1:2, :, :, 2:4]) * tf.reshape(pROI_anchor[1, :], [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]) + 1e-8)
        # adjusted_coords_wh_2 = tf.sqrt(
        #     tf.exp(coords[2:3, :, :, 2:4]) * tf.reshape(pROI_anchor[2, :], [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]) + 1e-8)
        # adjusted_coords_wh_3 = tf.sqrt(
        #     tf.exp(coords[3:4, :, :, 2:4]) * tf.reshape(pROI_anchor[3, :], [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]) + 1e-8)
        # adjusted_coords_wh = tf.concat([adjusted_coords_wh_0,
        #                                 adjusted_coords_wh_1,
        #                                 adjusted_coords_wh_2,
        #                                 adjusted_coords_wh_3], axis=0)

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

        tf.summary.scalar('loss_cx', tf.reduce_sum(loss, [0, 1, 2])[0])
        tf.summary.scalar('loss_cy', tf.reduce_sum(loss, [0, 1, 2])[1])
        tf.summary.scalar('loss_w_root', tf.reduce_sum(loss, [0, 1, 2])[2])
        tf.summary.scalar('loss_h_root', tf.reduce_sum(loss, [0, 1, 2])[3])
        tf.summary.scalar('loss_obj', tf.reduce_sum(loss, [0, 1, 2])[4])

        loss = tf.reshape(loss, [-1, H * W * B * (4 + 1)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        # tf.summary.scalar('{} loss'.format(m['model']), loss)

        return loss



