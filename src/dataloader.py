import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from progressbar import ProgressBar, Percentage, Bar
from scipy.misc import imread
from loader.loader_vot import loader_vot


# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_dataset(FLAGS, name):
    # TODO, occlusion label for training

    # Open a TFRRecordWriter
    filename = os.path.join(FLAGS.out, name + '.tfrecords')
    writeOpts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(filename, options=writeOpts)

    # Load each data sample (pimg, cimg, pbox, cbox) and write it to the TFRecord
    objLoaderVot = loader_vot(FLAGS.data_dir)
    videos = objLoaderVot.get_videos()
    video_keys = videos.keys()

    # for progressbar
    count = 0
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(indices)).start()
    for idx in range(len(videos)):
        video_frames = videos[video_keys[idx]][0]         # ex) bag/*.jpg list
        annot_frames = videos[video_keys[idx]][1]         # ex) bag/groundtruth, rectangle box info
        num_frames = min(len(video_frames), len(annot_frames))

        for i in range(0, num_frames):
            pimg_path = video_frames[i]
            cimg_path = video_frames[i+1]
            pbox = annot_frames[i]
            cbox = annot_frames[i+1]

            pimg = imread(pimg_path)
            cimg = imread(cimg_path)
            pimg = pimg[..., [2, 1, 0]] / 255.0
            cimg = cimg[..., [2, 1, 0]] / 255.0
            pimg_raw = pimg.tostring()
            cimg_raw = cimg.tostring()

            pbox_coor = [int(pbox.x1), int(pbox.y1), int(pbox.x2), int(pbox.y2)]
            cbox_coor = [int(cbox.x1), int(cbox.y1), int(cbox.x2), int(cbox.y2)]
            pbox_coor_raw = pbox_coor.tostring()
            cbox_coor_raw = cbox_coor.tostirng()

            example = tf.train.Example(features=tf.train.Features(feature={
                'pimg': _bytes_feature(pimg_raw),
                'cimg': _bytes_feature(cimg_raw),
                'pbox': _bytes_feature(pbox_coor_raw),
                'cbox': _bytes_feature(cbox_coor_raw)}))
            writer.write(example.SerializeToString())

            pbar.update(count + 1)
            count += 1
    writer.close()



def main()
    # DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

    convert_dataset(FLAGS, 'train_1_adj')
    # convert_dataset(train_idxs, 'train_1_dis')
    # convert_dataset(train_idxs, 'train_2_seq')  ... ex) train_2_seq_bag.tfrecords... ...



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='votxxxx directory'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Directory for output .tfrecords files'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('data_dir must exist and be a directory')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out must exist and be a directory')
    main()