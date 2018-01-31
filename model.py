import tensorflow as tf
from .ops import conv_bn

slim = tf.contrib.slim




# hyper param


# bulid model
# return: net out, feature(ROI)
class net:
    """ network """

    def __init__(self, mode, model_size=412, debug=False):
        self.mode = mode
        self.debug = debug
        self.global_step = slim.get_or_create_global_step()
        self.net_hw = model_size
        self.net_out_hw = model_size/32




    def model_conv(self, image, training_schedule, trainable=True):
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

        return conv_6


    def model_pred(self, cimg_conv5, cimg_conv6, pimg_conv6, ROI_coordinate, trainable=Ture):
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

        with tf.variable_scope(scope):
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                        [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:],
                                       [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (
                                           0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]),
                                           tf.square(predict_boxes[:, :, :, :, 3])])
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                        name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                         name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                           name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                        name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)





    def train(self, log_dir, training_schedule, pimg, cimg, gt, ckpt=None):


    def test(self, ckpt, pimg, cimg, etc...):


