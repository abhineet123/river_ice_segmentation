# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""

import sys

sys.path.append('..')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os.path
import time
import cv2
import numpy as np

import tensorflow as tf

import paramparse

from new_deeplab import common
from new_deeplab import model

from new_deeplab.datasets import data_generator
from new_deeplab.utils import save_annotation, linux_path
from new_deeplab.datasets.build_utils import undo_resize_ar, read_class_info

from new_deeplab_vis_params import NewDeeplabVisParams

# from old_deeplab.utils import save_annotation

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
    """Converts the predicted label for evaluation.

    There are cases where the training labels are not equal to the evaluation
    labels. This function is used to perform the conversion so that we could
    evaluate the results on the evaluation server.

    Args:
      prediction: Semantic segmentation prediction.
      train_id_to_eval_id: A list mapping from train id to evaluation id.

    Returns:
      Semantic segmentation prediction whose labels have been changed.
    """
    converted_prediction = prediction.copy()
    for train_id, eval_id in enumerate(train_id_to_eval_id):
        converted_prediction[prediction == train_id] = eval_id

    return converted_prediction


def _process_batch(params, sess, images,
                   original_images_resized,
                   semantic_predictions, image_names,
                   image_heights, image_widths,
                   # image_id_offset,
                   save_dir,
                   raw_save_dir, stacked_save_dir,
                   class_to_color,
                   # train_id_to_eval_id=None
                   ):
    """Evaluates one single batch qualitatively.

    Args:
      sess: TensorFlow session.
      original_images: One batch of original images.
      semantic_predictions: One batch of semantic segmentation predictions.
      image_names: Image names.
      image_heights: Image heights.
      image_widths: Image widths.
      image_id_offset: Image id offset for indexing images.
      save_dir: The directory where the predictions will be saved.
      raw_save_dir: The directory where the raw predictions will be saved.
      train_id_to_eval_id: A list mapping from train id to eval id.

    :param NewDeeplabVisParams params:

    """
    (original_images,
     original_images_resized,
     semantic_predictions,
     image_names,
     image_heights,
     image_widths) = sess.run([images,
                               original_images_resized,
                               semantic_predictions,
                               image_names,
                               image_heights,
                               image_widths])

    num_image = semantic_predictions.shape[0]
    for i in range(num_image):
        image_height = np.squeeze(image_heights[i])
        image_width = np.squeeze(image_widths[i])
        original_image = np.squeeze(original_images[i])
        # original_image_resized = np.squeeze(original_images_resized[i])
        semantic_prediction = np.squeeze(semantic_predictions[i])
        crop_semantic_prediction = semantic_prediction[:image_height, :image_width]



        # Save image.
        # save_annotation.save_annotation(
        #     original_image, save_dir, _IMAGE_FORMAT % (image_id_offset + i),
        #     add_colormap=False)

        # res_h, res_w = original_image_resized.shape[:2]
        # print('\n\noriginal_image_resized: {}'.format((res_w, res_h)))

        image_filename = (os.path.splitext(os.path.basename(image_names[i]))[0])
        image_dir_path = os.path.dirname(image_names[i])
        image_dir = os.path.basename(image_dir_path)
        image_dir_root = os.path.basename(os.path.dirname(image_dir_path))

        image_dir = tf.compat.as_str_any(image_dir)
        image_dir_root = tf.compat.as_str_any(image_dir_root)
        image_filename = tf.compat.as_str_any(image_filename)

        if image_dir == 'images':
            seq_name = image_dir_root
        else:
            seq_name = image_dir

        out_stacked_save_dir = os.path.join(stacked_save_dir, seq_name)
        out_raw_save_dir = os.path.join(raw_save_dir, seq_name)

        os.makedirs(out_stacked_save_dir, exist_ok=1)
        os.makedirs(out_raw_save_dir, exist_ok=1)

        # print('image_dir_path, image_filename: {}, {}'.format(image_dir_path, image_filename))
        stacked_path = os.path.join(out_stacked_save_dir, image_filename + '.jpg')
        raw_path = os.path.join(out_raw_save_dir, image_filename + '.png')

        unresized_crop_semantic_prediction = undo_resize_ar(crop_semantic_prediction, image_width, image_height)
        cv2.imwrite(raw_path, unresized_crop_semantic_prediction)

        mask_img = np.zeros_like(original_image)
        for _class_id, _col in class_to_color.items():
            mask_img[crop_semantic_prediction == _class_id] = _col

        _max = np.max(crop_semantic_prediction)
        _min = np.min(crop_semantic_prediction)

        # mask_img = (crop_semantic_prediction * (255.0 / np.max(crop_semantic_prediction))).astype(np.uint8)
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

        unresized_original_image = undo_resize_ar(original_image, image_width, image_height)
        unresized_mask_img = undo_resize_ar(mask_img, image_width, image_height)
        stacked_img = np.concatenate((unresized_original_image, unresized_mask_img), axis=1)

        # cv2.imshow('stacked_img', stacked_img)
        # cv2.waitKey(0)

        cv2.imwrite(stacked_path, stacked_img)

        """more mind-bogglingly annoying gunky foulness - if ever there was horribly designed meaninglessly 
        complicated piece of crappy garbage, then this is it"""

        # Save prediction.
        # save_annotation.save_annotation(
        #     crop_semantic_prediction, save_dir,
        #     _PREDICTION_FORMAT % (image_id_offset + i),
        #     add_colormap=True,
        #     colormap_type=FLAGS.colormap_type)

        # save_annotation.save_annotation(
        #     crop_semantic_prediction, out_raw_save_dir, image_filename,
        #     add_colormap=False)

        # Save prediction.
        # if params.also_save_vis_predictions:
        #     save_annotation.save_annotation(
        #         crop_semantic_prediction, save_dir, image_filename,
        #         # _PREDICTION_FORMAT % (image_id_offset + i), add_colormap=True,
        #         colormap_type=params.colormap_type)

        # if FLAGS.also_save_raw_predictions:
        #     image_filename = os.path.basename(image_names[i])
        #
        #     if train_id_to_eval_id is not None:
        #         crop_semantic_prediction = _convert_train_id_to_eval_id(
        #             crop_semantic_prediction,
        #             train_id_to_eval_id)
        #     save_annotation.save_annotation(
        #         crop_semantic_prediction, raw_save_dir, image_filename,
        #         add_colormap=False)


def run(params):
    """

    :param NewDeeplabVisParams params:
    """

    paramparse.to_flags(tf.app.flags.FLAGS, params, verbose=0, allow_missing=1)

    tf.logging.set_verbosity(tf.logging.INFO)

    # paramparse.from_flags(FLAGS, to_clipboard=1)

    print('reading input images from {}'.format(params.dataset_dir))

    # Get dataset-dependent information.
    dataset = data_generator.Dataset(
        dataset_name=params.dataset,
        is_test=1,
        split_name=params.vis_split,
        dataset_dir=params.dataset_dir,
        batch_size=params.vis_batch_size,
        crop_size=params.vis_crop_size,
        min_resize_value=params.min_resize_value,
        max_resize_value=params.max_resize_value,
        resize_factor=params.resize_factor,
        model_variant=params.model_variant,
        is_training=False,
        should_shuffle=False,
        should_repeat=False)

    train_id_to_eval_id = None
    if dataset.dataset_name == data_generator.get_cityscapes_dataset_name():
        print('Cityscapes requires converting train_id to eval_id.')
        train_id_to_eval_id = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

    # Prepare for visualization.
    os.makedirs(params.vis_logdir, exist_ok=1)

    save_dir = os.path.join(params.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
    # os.makedirs(save_dir, exist_ok=1)

    raw_save_dir = os.path.join(
        params.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
    stacked_save_dir = os.path.join(
        params.vis_logdir, 'stacked')
    os.makedirs(raw_save_dir, exist_ok=1)
    os.makedirs(stacked_save_dir, exist_ok=1)

    print('Visualizing on set: {} with split: {}'.format(params.dataset, params.vis_split))

    classes, composite_classes = read_class_info(params.class_info_path)
    class_to_color = {i: k[1] for i, k in enumerate(classes)}

    with tf.Graph().as_default():
        samples = dataset.get_one_shot_iterator().get_next()

        model_options = common.ModelOptions(
            # params=params,
            outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
            crop_size=[int(sz) for sz in params.vis_crop_size],
            atrous_rates=params.atrous_rates,
            output_stride=params.output_stride)

        if tuple(params.eval_scales) == (1.0,):
            print('Performing single-scale test.')
            predictions = model.predict_labels(
                samples[common.IMAGE],
                model_options=model_options,
                image_pyramid=params.image_pyramid)
        else:
            print('Performing multi-scale test.')
            if params.quantize_delay_step >= 0:
                raise ValueError(
                    'Quantize mode is not supported with multi-scale test.')
            predictions = model.predict_labels_multi_scale(
                samples[common.IMAGE],
                model_options=model_options,
                eval_scales=params.eval_scales,
                add_flipped_images=params.add_flipped_images)
        predictions = predictions[common.OUTPUT_TYPE]

        # if params.min_resize_value and params.max_resize_value:
        #     """
        #     foul buggy garbage since v1.12: https://github.com/tensorflow/tensorflow/issues/38694
        #     as is so typical of the gunky piece of crap that is tensorflow
        #     """
        #     # Only support batch_size = 1, since we assume the dimensions of original
        #     # image after tf.squeeze is [height, width, 3].
        #     assert params.vis_batch_size == 1
        #
        #     # Reverse the resizing and padding operations performed in preprocessing.
        #     # First, we slice the valid regions (i.e., remove padded region) and then
        #     # we resize the predictions back.
        #     original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
        #     original_image_shape = tf.shape(original_image)
        #     predictions = tf.slice(
        #         predictions,
        #         [0, 0, 0],
        #         [1, original_image_shape[0], original_image_shape[1]])
        #     resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
        #                                  tf.squeeze(samples[common.WIDTH])])
        #     predictions = tf.squeeze(
        #         tf.image.resize_images(tf.expand_dims(predictions, 3),
        #                                resized_shape,
        #                                method=tf.image.ResizeMethod.BILINEAR,
        #                                align_corners=True), 3)

        tf.train.get_or_create_global_step()
        if params.quantize_delay_step >= 0:
            tf.contrib.quantize.create_eval_graph()

        num_iteration = 0
        max_num_iteration = params.max_number_of_iterations

        checkpoints_iterator = tf.contrib.training.checkpoints_iterator(
            params.checkpoint_dir, min_interval_secs=params.eval_interval_secs)

        print('save_dir: {}'.format(save_dir))
        print('raw_save_dir: {}'.format(raw_save_dir))
        print('stacked_save_dir: {}'.format(stacked_save_dir))

        for checkpoint_path in checkpoints_iterator:
            num_iteration += 1
            print(
                'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))
            print('Visualizing with model {}'.format(checkpoint_path))

            scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=scaffold,
                master=params.master,
                checkpoint_filename_with_path=checkpoint_path)
            with tf.train.MonitoredSession(
                    session_creator=session_creator, hooks=None) as sess:
                batch = 0
                # image_id_offset = 0

                while not sess.should_stop():
                    print('Visualizing batch {:d}'.format(batch + 1))
                    _process_batch(
                        params=params,
                        sess=sess,
                        images=samples[common.IMAGE],
                        original_images_resized=samples[common.ORIGINAL_IMAGE],
                        semantic_predictions=predictions,
                        image_names=samples[common.IMAGE_NAME],
                        image_heights=samples[common.HEIGHT],
                        image_widths=samples[common.WIDTH],
                        # image_id_offset=image_id_offset,
                        save_dir=save_dir,
                        raw_save_dir=raw_save_dir,
                        stacked_save_dir=stacked_save_dir,
                        class_to_color=class_to_color,
                        # train_id_to_eval_id=train_id_to_eval_id
                    )
                    # image_id_offset += params.vis_batch_size
                    batch += 1

            print(
                'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))
            if num_iteration >= max_num_iteration > 0:
                break


if __name__ == '__main__':
    params = NewDeeplabVisParams()
    paramparse.process(params)

    params.process()

    run(params)
