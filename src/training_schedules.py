import tensorflow as tf

# https://github.com/sampepose/flownet2-tf

POLICY = {
    'step_values': [400000, 600000, 800000, 1000000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'decay': 0.0005,
    'max_iter': 1200000,

    'BATCH_SIZE': 2,
    'height': 416,
    'width': 416,
    'side': 13,
    'channels': 3,
    'optimizer': dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,}),

    'object_scale': 5,
    'noobject_scale': 1,
    'class_scale': 1,
    'coord_scale': 1,
    'thresh ':  .6,
    'num': 1,
    'anchors': [6.63, 11.38],

    'ITEMS_TO_DESCRIPTIONS': {
        'pimg': 'A 3-channel previous image.',
        'cimg': 'A 3-channel current image.',
        'pbox': 'pimg box x1, y1, x2, y2',
        'cbox': 'cimg box x1, y1, x2, y2',
    },
    'SIZES': {
        'train': 196,
        'validate': 196,
        'sample': 196,
    },
    'PATHS': {
        'train': './data/tfrecords/train_1_adj.tfrecords',
        'validate': './data/tfrecords/fc_val.tfrecords',
        'sample': './data/tfrecords/train_1_adj_sample.tfrecords',
    },
    'interpolation': 'bicubic'
}
