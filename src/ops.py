import tensorflow as tf
slim = tf.contrib.slim
from .training_schedules import POLICY
# tensorflow wrapper

# ex) conv + batch + relu

# https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)



def conv_bn(input,
            filters,
            kernel=3,
            scope='conv',
            training_schedule=POLICY,
            trainable=True):
    """
    control filter size, kernel size, tr_schedule
    """

    weights_regularizer = slim.l2_regularizer(training_schedule['decay'])
    bn_params = {
        # Decay for the moving averages
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d],
                        trainable=trainable,
                        weights_initializer=slim.variance_scaling_initializer(),
                        weights_regularizer=weights_regularizer,
                        activation_fn=LeakyReLU,
                        padding="SAME",
                        stride=1,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=bn_params,
                        scope=scope):

        out = slim.conv2d(input, filters, kernel)

        return out


def conv_linear(input,
                filters,
                kernel=3,
                scope='conv',
                training_schedule=POLICY,
                trainable=True):
    """
    control filter size, kernel size, tr_schedule
    """
    weights_regularizer = slim.l2_regularizer(training_schedule['decay'])
    with slim.arg_scope([slim.conv2d],
                        trainable=trainable,
                        weights_initializer=slim.variance_scaling_initializer(),
                        weights_regularizer=weights_regularizer,
                        biases_initializer=tf.zeros_initializer(),
                        activation_fn=None,
                        padding="SAME",
                        stride=1,
                        scope=scope):

        out = slim.conv2d(input, filters, kernel)

        return out
