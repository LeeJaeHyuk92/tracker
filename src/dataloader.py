# -*- coding: utf-8 -*-
import tensorflow as tf
slim = tf.contrib.slim


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/tfexample_decoder.py
class Image(slim.tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                 image_key=None,
                 format_key=None,
                 shape=None,
                 channels=3,
                 dtype=tf.uint8,
                 repeated=False):
        """Initializes the image.
        Args:
          image_key: the name of the TF-Example feature in which the encoded image
            is stored.
          shape: the output shape of the image as 1-D `Tensor`
            [height, width, channels]. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should
            be supplied only if all the stored images have the same shape.
          channels: the number of channels in the image.
          dtype: images will be decoded at this bit depth. Different formats
            support different bit depths.
              See tf.image.decode_image,
                  tf.decode_raw,
          repeated: if False, decodes a single image. If True, decodes a
            variable number of image strings from a 1D tensor of strings.
        """
        if not image_key:
            image_key = 'image/encoded'

        super(Image, self).__init__([image_key])
        self._image_key = image_key
        self._shape = shape
        self._channels = channels
        self._dtype = dtype
        self._repeated = repeated

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_buffer = keys_to_tensors[self._image_key]

        if self._repeated:
            return functional_ops.map_fn(lambda x: self._decode(x),
                                         image_buffer, dtype=self._dtype)
        else:
            return self._decode(image_buffer)

    def _decode(self, image_buffer):
        """Decodes the image buffer.
        Args:
          image_buffer: The tensor representing the encoded image tensor.
        Returns:
          A tensor that represents decoded image of self._shape, or
          (?, ?, self._channels) if self._shape is not specified.
        """
        def decode_raw():
            """Decodes a raw image."""
            return tf.decode_raw(image_buffer, out_type=self._dtype)

        image = decode_raw()
        # image.set_shape([None, None, self._channels])
        if self._shape is not None:
            image = tf.reshape(image, self._shape)

        return image


def __get_dataset(dataset_config, split_name):
    """
    dataset_config: A dataset_config defined in datasets.py
    split_name: 'train'/'validate'
    """
    with tf.name_scope('__get_dataset'):
        if split_name not in dataset_config['SIZES']:
            raise ValueError('split name %s not recognized' % split_name)

        IMAGE_HEIGHT, IMAGE_WIDTH, SIDE = dataset_config['height'], dataset_config['width'], dataset_config['side']
        reader = tf.TFRecordReader
        keys_to_features = {
            'pimg_resize': tf.FixedLenFeature((), tf.string),
            'cimg_resize': tf.FixedLenFeature((), tf.string),
            'pbox_xy': tf.FixedLenFeature((), tf.string),
            'pROI': tf.FixedLenFeature((), tf.string),
            'confs': tf.FixedLenFeature((), tf.string),
            'coord': tf.FixedLenFeature((), tf.string),
            'areas': tf.FixedLenFeature((), tf.string),
            'upleft': tf.FixedLenFeature((), tf.string),
            'botright': tf.FixedLenFeature((), tf.string),
        }
        items_to_handlers = {
            'pimg_resize': Image(
                image_key='pimg_resize',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3),
            'cimg_resize': Image(
                image_key='cimg_resize',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3),
            'pbox_xy': Image(
                image_key='pbox_xy',
                dtype=tf.int32,
                shape=[2],
                channels=1),
            'confs': Image(
                image_key='confs',
                dtype=tf.float64,
                shape=[SIDE*SIDE, 1],
                channels=1),
            'coord': Image(
                image_key='coord',
                dtype=tf.float64,
                shape=[SIDE*SIDE, 1, 4],
                channels=1),
            'areas': Image(
                image_key='areas',
                dtype=tf.float64,
                shape=[SIDE*SIDE, 1],
                channels=1),
            'upleft': Image(
                image_key='upleft',
                dtype=tf.float64,
                shape=[SIDE*SIDE, 1, 2],
                channels=1),
            'botright': Image(
                image_key='botright',
                dtype=tf.float64,
                shape=[SIDE*SIDE, 1, 2],
                channels=1),
            'pROI': Image(
                image_key='pROI',
                dtype=tf.int32,
                shape=[4],
                channels=1),


            # 'pbox_xy': slim.tfexample_decoder.Tensor('pbox_xy'),
            # 'confs': slim.tfexample_decoder.Tensor('confs'),
            # 'coord': slim.tfexample_decoder.Tensor('coord'),
            # 'areas': slim.tfexample_decoder.Tensor('areas'),
            # 'upleft': slim.tfexample_decoder.Tensor('upleft'),
            # 'botright': slim.tfexample_decoder.Tensor('botright'),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
        return slim.dataset.Dataset(
            data_sources=dataset_config['PATHS'][split_name],
            reader=reader,
            decoder=decoder,
            num_samples=dataset_config['SIZES'][split_name],
            items_to_descriptions=dataset_config['ITEMS_TO_DESCRIPTIONS'])


def load_batch(dataset_config, split_name):
    num_threads = 8
    reader_kwargs = {'options': tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)}

    with tf.name_scope('load_batch'):
        dataset = __get_dataset(dataset_config, split_name)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_threads,
            common_queue_capacity=2048,
            common_queue_min=1024,
            reader_kwargs=reader_kwargs)
        pimg_resize, cimg_resize, pbox_xy, pROI,\
        confs, coord, areas, upleft, botright = data_provider.get(['pimg_resize',
                                                                   'cimg_resize',
                                                                   'pbox_xy',
                                                                   'pROI',
                                                                   'confs',
                                                                   'coord',
                                                                   'areas',
                                                                   'upleft',
                                                                   'botright'])
        pimg_resize, cimg_resize, \
        confs, coord, areas, upleft, botright = map(tf.to_float, [pimg_resize,
                                                                  cimg_resize,
                                                                  confs,
                                                                  coord,
                                                                  areas,
                                                                  upleft,
                                                                  botright])
        pbox_xy, pROI = map(tf.to_int32, [pbox_xy, pROI])

        pimg_resize, cimg_resize, pbox_xy, pROI,\
        confs, coord, areas, upleft, botright = map(lambda x: tf.expand_dims(x, 0), [pimg_resize,
                                                                                     cimg_resize,
                                                                                     pbox_xy,
                                                                                     pROI,
                                                                                     confs,
                                                                                     coord,
                                                                                     areas,
                                                                                     upleft,
                                                                                     botright])



        # with tf.device('/cpu:0'):

        return tf.train.batch([pimg_resize, cimg_resize, pbox_xy, pROI,
                               confs, coord, areas, upleft, botright],
                              enqueue_many=True,
                              batch_size=dataset_config['BATCH_SIZE'],
                              capacity=dataset_config['BATCH_SIZE'] * 4,
                              num_threads=num_threads,
                              allow_smaller_final_batch=False)
