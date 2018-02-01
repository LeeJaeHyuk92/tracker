import tensorflow as tf

POLICY = {
    'step_values': [400000, 600000, 800000, 1000000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'decay': 0.0005,
    'max_iter': 1200000,

    'batch': 64,
    'height' : 416,
    'width' : 416,
    'channels' : 3,
    'optimizer' : dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,}),
}