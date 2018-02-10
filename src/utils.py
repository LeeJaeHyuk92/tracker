import tensorflow as tf


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))


def schedule_verbose(POLICY, data_type):
    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)
    print(bcolors.FAIL + POLICY['says'] + bcolors.ENDC)
    for key in POLICY:
        if key in ['BATCH_SIZE',
                   'step_values',
                   'learning_rates',
                   'momentum',
                   'decay',
                   'max_iter',
                   'object_scale',
                   'noobject_scale',
                   'class_scale',
                   'coord_scale',
                   'thresh ',
                   'num',
                   'anchors']:
            print(bcolors.BOLD + str(key) + bcolors.ENDC)
            print(str(POLICY[str(key)]).rjust(80))
        elif key in 'SIZES':
            print(bcolors.BOLD + data_type + ' SIZE ' + bcolors.ENDC)
            print(bcolors.FAIL + str(POLICY[str(key)][data_type]).rjust(80) + bcolors.ENDC)
        elif key in 'PATHS':
            print(bcolors.BOLD + data_type + ' PATH ' + bcolors.ENDC)
            print(bcolors.FAIL + str(POLICY[str(key)][data_type]).rjust(80) + bcolors.ENDC)
    print("\n" * 2)

    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)
    print(bcolors.WARNING + 'network' + bcolors.ENDC)
    for key in POLICY:
        if key in ['height',
                   'width',
                   'channels',
                   'side',
                   'interpolation']:
            print(bcolors.BOLD + str(key) + bcolors.ENDC)
            print(str(POLICY[str(key)]).rjust(80))
    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)
    print("\n" * 2)


def ckpt_reader(ckpt, value=False):
    from tensorflow.python import pywrap_tensorflow
    reader=pywrap_tensorflow.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)
    print(bcolors.WARNING + 'checkpoint' + bcolors.ENDC)
    for key in sorted(var_to_shape_map):
          print("tensor_name: ", key)

          if value:
            print(reader.get_tensor(key)) # if you look tensor value

    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)


def calculate_box_tf(cimg_resize_batch, net_out, POLICY):
    H, W = POLICY['side'], POLICY['side']
    B = POLICY['num']
    HW = H * W  # number of grid cells
    anchors = POLICY['anchors']

    # calculate box
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1)])
    coords = net_out_reshape[:, :, :, :, :4]
    adjusted_coords_xy = expit_tensor(coords[:, :, :, :, 0:2])
    adjusted_coords_wh = tf.exp(coords[:, :, :, :, 2:4]) * tf.reshape(anchors, [1, 1, 1, B, 2]) / tf.reshape([float(W), float(H)], [1, 1, 1, 1, 2])

    xymin = adjusted_coords_xy - adjusted_coords_wh
    xymax = adjusted_coords_xy + adjusted_coords_wh
    xyminmax = tf.concat([xymin, xymax], axis=4)
    # yxminmax = tf.transpose(xyminmax, [0, 2, 1, 4, 3])

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4:])
    adjusted_net_out = tf.concat([xyminmax, adjusted_c], 4)
    top_obj_index = tf.where(adjusted_net_out[:, :, :, :, 4] > tf.reshape([POLICY['thresh']], [POLICY['BATCH_SIZE'], H, W, B]))
    # TODO, FIX IT, tf.where
    predict = adjusted_net_out[top_obj_index]

    yx_predict = tf.concat([predict[..., 1], predict[..., 0], predict[..., 3], predict[..., 2]], axis=2)

    cimg_resize_batch = cimg_resize_batch * 255
    tf.image.draw_bounding_boxes(cimg_resize_batch, yx_predict)