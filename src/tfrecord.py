import os
import argparse
import tensorflow as tf
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
from training_schedules import POLICY
from scipy.misc import imread, imresize
from loader.loader_vot import loader_vot
import cv2

# from .misc import show



# https://github.com/thtrieu/darkflow
def _batch(w, h, pbox, cbox, training_schedule):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    :param box: box.x1, box.y1, box.x2, box.y2
    """
    meta = training_schedule
    S, B = meta['side'], meta['num']

    # Calculate regression target
    cellx = 1. * w / S
    celly = 1. * h / S
    obj = [0, 0, 0, 0, 0]

    for box in [pbox, cbox]:
        if box.x1 < 0:
            box.x1 = 0.
            print('adjust box coordinate')
        if box.y1 < 0:
            box.y1 = 0.
            print('adjust box coordinate')
        if box.x2 > w:
            box.x2 = float(w)
            print('adjust box coordinate')
        if box.y2 > h:
            box.y2 = float(h)
            print('adjust box coordinate')
    # pbox
    centerx = .5 * (pbox.x1 + pbox.x2)  # xmin, xmax
    centery = .5 * (pbox.y1 + pbox.y2)  # ymin, ymax
    cx = centerx / cellx
    cy = centery / celly
    pbox_xy = np.array([np.floor(cx), np.floor(cy)], dtype=np.int32)

    # cbox
    centerx = .5 * (cbox.x1 + cbox.x2)  # xmin, xmax
    centery = .5 * (cbox.y1 + cbox.y2)  # ymin, ymax
    cx = centerx / cellx
    cy = centery / celly
    if cx >= S or cy >= S:
        raise('center point error')
        return None, None
    obj[3] = float(cbox.x2 - cbox.x1) / w
    obj[4] = float(cbox.y2 - cbox.y1) / h
    obj[3] = np.sqrt(obj[3])
    obj[4] = np.sqrt(obj[4])
    obj[1] = cx - np.floor(cx)  # centerx
    obj[2] = cy - np.floor(cy)  # centery
    obj += [int(np.floor(cy) * S + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    confs = np.zeros([S * S, B])
    coord = np.zeros([S * S, B, 4])
    prear = np.zeros([S * S, 4])

    coord[obj[5], :, :] = [obj[1:5]] * B
    prear[obj[5], 0] = obj[1] - obj[3] ** 2 * .5 * S  # xleft
    prear[obj[5], 1] = obj[2] - obj[4] ** 2 * .5 * S  # yup
    prear[obj[5], 2] = obj[1] + obj[3] ** 2 * .5 * S  # xright
    prear[obj[5], 3] = obj[2] + obj[4] ** 2 * .5 * S  # ybot
    confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at loss layer
    loss_feed_val = {
        'pbox_xy': pbox_xy,
        'confs': confs, 'coord': coord,
        'areas': areas, 'upleft': upleft,
        'botright': botright
    }

    return loss_feed_val

# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# https://github.com/sampepose/flownet2-tf
def convert_dataset(FLAGS, name):
    # TODO, occlusion label for training,
    # TODO, pbox, cbox, area, upleft, botright (0~1)

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
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(videos)).start()
    for idx in range(len(videos)):
        video_frames = videos[video_keys[idx]][0]         # ex) bag/*.jpg list
        annot_frames = videos[video_keys[idx]][1]         # ex) bag/groundtruth, rectangle box info
        num_frames = min(len(video_frames), len(annot_frames))

        # num_frame+1.jpg does not exist
        for i in range(0, num_frames-1):
            pimg_path = video_frames[i]
            cimg_path = video_frames[i+1]
            pbox = annot_frames[i]
            cbox = annot_frames[i+1]

            pimg = imread(pimg_path)
            cimg = imread(cimg_path)
            pimg = pimg[..., [2, 1, 0]]
            cimg = cimg[..., [2, 1, 0]]
            pimg_resize = imresize(pimg, [POLICY['height'], POLICY['width'], 3], POLICY['interpolation'])
            cimg_resize = imresize(cimg, [POLICY['height'], POLICY['width'], 3], POLICY['interpolation'])

            pimg_resize = pimg_resize / 255.0
            cimg_resize = cimg_resize / 255.0

            pimg_resize_raw = pimg_resize.tostring()
            cimg_resize_raw = cimg_resize.tostring()

            h, w, _ = cimg.shape
            loss_feed_val = _batch(w, h, pbox, cbox, POLICY)

            pbox_xy = loss_feed_val['pbox_xy']
            confs = loss_feed_val['confs']
            coord = loss_feed_val['coord']
            areas = loss_feed_val['areas']
            upleft = loss_feed_val['upleft']
            botright = loss_feed_val['botright']

            pbox_xy_raw = pbox_xy.tostring()
            confs_raw = confs.tostring()
            coord_raw = coord.tostring()
            areas_raw = areas.tostring()
            upleft_raw = upleft.tostring()
            botright_raw = botright.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'pimg_resize': _bytes_feature(pimg_resize_raw),
                'cimg_resize': _bytes_feature(cimg_resize_raw),
                'pbox_xy': _bytes_feature(pbox_xy_raw),
                'confs': _bytes_feature(confs_raw),
                'coord': _bytes_feature(coord_raw),
                'areas': _bytes_feature(areas_raw),
                'upleft': _bytes_feature(upleft_raw),
                'botright': _bytes_feature(botright_raw)}))
            writer.write(example.SerializeToString())

            pbar.update(count + 1)
        count += 1
    writer.close()

def dataset_visualizatioin_gt(FLAGS):
    # TODO, occlusion label for training,
    # TODO, pbox, cbox, area, upleft, botright (0~1)

    sub_vot_dirs = [dir_name for dir_name in os.listdir(FLAGS.data_dir) if os.path.isdir(os.path.join(FLAGS.data_dir, dir_name))]
    for vot_sub_dir in sub_vot_dirs:
        vot_dir_path = os.path.join('./data/tfrecords', FLAGS.data_dir, vot_sub_dir)
        if not os.path.exists(vot_dir_path):
            os.makedirs(vot_dir_path)


    # Load each data sample (pimg, cimg, pbox, cbox) and write it to the TFRecord
    objLoaderVot = loader_vot(FLAGS.data_dir)
    videos = objLoaderVot.get_videos()
    video_keys = videos.keys()

    # for progressbar
    count = 1
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(videos)).start()
    for idx in range(len(videos)):
        video_frames = videos[video_keys[idx]][0]  # ex) bag/*.jpg list
        annot_frames = videos[video_keys[idx]][1]  # ex) bag/groundtruth, rectangle box info
        num_frames = min(len(video_frames), len(annot_frames))

        for i in range(0, num_frames):
            pimg_path = video_frames[i]
            if pimg_path=='./data/vot2015_full/birds2/00000384.jpg':
                pass
            pbox = annot_frames[i]

            pimg = imread(pimg_path)
            # pimg = pimg[..., [2, 1, 0]]
            # pimg_resize = imresize(pimg, [POLICY['height'], POLICY['width'], 3], POLICY['interpolation'])
            pimg_gt = cv2.rectangle(pimg, (int(pbox.x1), int(pbox.y1)), (int(pbox.x2), int(pbox.y2)), (0, 0, 255), 3)
            # cv2.imwrite('./data/tfrecords/'+ pimg_path, pimg_gt)

            # try:
            #     cv2.imwrite('./data/tfrecords/data/vot2015_full_all/' + '{:05d}'.format(count)+ '_' + pimg_path.split('/')[-2] + '_' +pimg_path.split('/')[-1],
            #                 pimg_gt)
            # except:
            #     raise('write false')

            count += 1






def main():
    # DATA_PATH = '/home/jaehyuk/code/own/tracker/data/vot2015'

    convert_dataset(FLAGS, FLAGS.name)
    # dataset_visualizatioin_gt(FLAGS)

    # convert_dataset(train_idxs, 'train_1_adj')
    # convert_dataset(train_idxs, 'train_1_dis')
    # convert_dataset(train_idxs, 'train_2_seq')  ... ex) train_2_seq_bag.tfrecords... ...



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='votxxxx/ directory'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Directory for output .tfrecords files'
    )
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Name of .tfrecords files'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('data_dir must exist and be a directory')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out must exist and be a directory')
    main()