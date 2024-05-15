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
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""
try:
    from tensorflow.python.util import deprecation
except BaseException:
    pass
else:
    deprecation._PRINT_DEPRECATION_WARNINGS = False

import six
import sys

sys.path.append('..')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.ops import math_ops

import paramparse

from new_deeplab import common
from new_deeplab import model
from new_deeplab.datasets import data_generator
from new_deeplab.utils import train_utils, linux_path
from new_deeplab.datasets.build_utils import read_class_info

from new_deeplab_train_params import NewDeeplabTrainParams

# paramparse.from_flags(FLAGS, to_clipboard=1)


print()


def _build_deeplab(params, iterator, outputs_to_num_classes, ignore_label, class_to_color):
    """Builds a clone of DeepLab.

    Args:
      iterator: An iterator of type tf.data.Iterator for images and labels.
      outputs_to_num_classes: A map from output type to the number of classes. For
        example, for the task of semantic segmentation with 21 semantic classes,
        we would have outputs_to_num_classes['semantic'] = 21.
      ignore_label: Ignore label.

    :param NewDeeplabTrainParams params:
    """
    samples = iterator.get_next()
    samples[common.IMAGE].set_shape([params.train_batch_size, params.train_crop_size[0], params.train_crop_size[1], 3])

    # Add name to input and label nodes so we can add to summary.
    samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
    samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

    model_options = common.ModelOptions(
        # params=params,
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=[int(sz) for sz in params.train_crop_size],
        atrous_rates=params.atrous_rates,
        output_stride=params.output_stride)

    outputs_to_scales_to_logits = model.multi_scale_logits(
        samples[common.IMAGE],
        model_options=model_options,
        image_pyramid=params.image_pyramid,
        weight_decay=params.weight_decay,
        is_training=True,
        fine_tune_batch_norm=params.fine_tune_batch_norm,
        nas_training_hyper_parameters={
            'drop_path_keep_prob': params.drop_path_keep_prob,
            'total_training_steps': params.train_steps,
        })

    # Add name to graph node so we can add to summary.
    output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
    output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
        output_type_dict[model.MERGED_LOGITS_SCOPE], name=common.OUTPUT_TYPE)

    for output, num_classes in six.iteritems(outputs_to_num_classes):
        train_utils.add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples[common.LABEL],
            num_classes,
            ignore_label,
            loss_weight=1.0,
            upsample_logits=params.upsample_logits,
            hard_example_mining_step=params.hard_example_mining_step,
            top_k_percent_pixels=params.top_k_percent_pixels,
            scope=output)

        # Log the summary
        _log_summaries(samples[common.IMAGE], samples[common.LABEL], num_classes,
                       output_type_dict[model.MERGED_LOGITS_SCOPE],
                       save_summaries_images=params.save_summaries_images,
                       class_to_color=class_to_color,
                       )


def _tower_loss(params, iterator, num_of_classes, ignore_label, scope, reuse_variable, class_to_color):
    """Calculates the total loss on a single tower running the deeplab model.

    Args:
      iterator: An iterator of type tf.data.Iterator for images and labels.
      num_of_classes: Number of classes for the dataset.
      ignore_label: Ignore label for the dataset.
      scope: Unique prefix string identifying the deeplab tower.
      reuse_variable: If the variable should be reused.

    Returns:
       The total loss for a batch of data.
    """
    with tf.variable_scope(
            tf.get_variable_scope(), reuse=True if reuse_variable else None):
        _build_deeplab(params, iterator, {common.OUTPUT_TYPE: num_of_classes}, ignore_label, class_to_color)

    losses = tf.losses.get_losses(scope=scope)
    for loss in losses:
        tf.summary.scalar('Losses/%s' % loss.op.name, loss)

    regularization_loss = tf.losses.get_regularization_loss(scope=scope)
    tf.summary.scalar('Losses/%s' % regularization_loss.op.name,
                      regularization_loss)

    total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
    return total_loss


def _average_gradients(tower_grads):
    """Calculates average of gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list is
        over individual gradients. The inner list is over the gradient calculation
        for each tower.

    Returns:
       List of pairs of (gradient, variable) where the gradient has been summed
         across all towers.

    :param NewDeeplabTrainParams params:
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads, variables = zip(*grad_and_vars)
        grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

        # All vars are of the same value, using the first tower here.
        average_grads.append((grad, variables[0]))

    return average_grads


def _log_summaries(input_image, label, num_of_classes, output, save_summaries_images, class_to_color):
    """Logs the summaries for the model.

    Args:
      input_image: Input image of the model. Its shape is [batch_size, height,
        width, channel].
      label: Label of the image. Its shape is [batch_size, height, width].
      num_of_classes: The number of classes of the dataset.
      output: Output of the model. Its shape is [batch_size, height, width].

    :param bool save_summaries_images:
    """
    # Add summaries for model variables.
    for model_var in tf.model_variables():
        tf.summary.histogram(model_var.op.name, model_var)

    # Add summaries for images, labels, semantic predictions.
    if save_summaries_images:
        batch_size, input_h, input_w, n_channels = input_image.shape

        # tf.summary.image('samples/%s' % common.IMAGE, input_image)

        summary_image = tf.cast(input_image, tf.uint8)

        # Scale up summary image pixel values for better visualization.
        # pixel_scaling = max(1, 255 // num_of_classes)
        summary_label = tf.cast(label, tf.uint8)
        # tf.summary.image('samples/%s' % common.LABEL, summary_label)

        print('num_of_classes: {}'.format(num_of_classes))
        # print('pixel_scaling: {}'.format(pixel_scaling))
        print('batch_size: {}'.format(batch_size))
        print('input_h: {}'.format(input_h))
        print('input_w: {}'.format(input_w))
        print('n_channels: {}'.format(n_channels))

        predictions = tf.expand_dims(tf.argmax(output, 3), -1)
        summary_predictions = tf.cast(predictions, tf.uint8)
        # tf.summary.image('samples/%s' % common.OUTPUT_TYPE, summary_predictions)

        summary_label_resized = tf.image.resize(summary_label, (input_h, input_w))
        summary_predictions_resized = tf.image.resize(summary_predictions, (input_h, input_w))

        summary_label_resized = tf.cast(summary_label_resized, tf.uint8)
        summary_predictions_resized = tf.cast(summary_predictions_resized, tf.uint8)

        summary_label_orig_min = tf.reduce_min(summary_label)
        summary_label_orig_max = tf.reduce_max(summary_label)
        summary_label_orig_unique, _ = tf.unique(tf.reshape(summary_label, shape=(-1,)))
        n_summary_label_orig_unique = tf.size(summary_label_orig_unique)

        summary_label_min = tf.reduce_min(summary_label_resized)
        summary_label_max = tf.reduce_max(summary_label_resized)
        summary_label_unique, _ = tf.unique(tf.reshape(summary_label_resized, shape=(-1,)))
        n_summary_label_unique = tf.size(summary_label_unique)

        summary_pred_min = tf.reduce_min(summary_predictions_resized)
        summary_pred_max = tf.reduce_max(summary_predictions_resized)
        summary_pred_unique, _ = tf.unique(tf.reshape(summary_predictions_resized, shape=(-1,)))
        n_summary_pred_unique = tf.size(summary_pred_unique)

        # print('summary_label: {}, {}'.format(summary_label_min, summary_label_max))
        # print('summary_pred: {}, {}'.format(summary_pred_min, summary_pred_max))

        tf.summary.scalar('summary_label_min', summary_label_min)
        tf.summary.scalar('summary_label_max', summary_label_max)
        tf.summary.scalar('n_summary_label_unique', n_summary_label_unique)

        tf.summary.scalar('summary_pred_min', summary_pred_min)
        tf.summary.scalar('summary_pred_max', summary_pred_max)
        tf.summary.scalar('n_summary_pred_unique', n_summary_pred_unique)

        tf.summary.scalar('summary_label_orig_min', summary_label_orig_min)
        tf.summary.scalar('summary_label_orig_max', summary_label_orig_max)
        tf.summary.scalar('n_summary_label_orig_unique', n_summary_label_orig_unique)

        # if n_channels == 3:
        # summary_label_resized_rgb = tf.image.grayscale_to_rgb(summary_label_resized)
        # summary_predictions_resized_rgb = tf.image.grayscale_to_rgb(summary_predictions_resized)

        summary_predictions_resized_list = []
        summary_label_resized_list = []

        for _class_id, _col in class_to_color.items():
            label_mask1 = tf.equal(summary_label_resized, _class_id)
            label_mask2 = tf.cast(label_mask1, tf.uint8)
            label_mask = tf.squeeze(label_mask2)
            # slice_y_greater_than_one = tf.boolean_mask(summary_label_resized_rgb, mask)
            label_red = tf.multiply(label_mask, _col[2])
            label_green = tf.multiply(label_mask, _col[1])
            label_blue = tf.multiply(label_mask, _col[0])

            label_mask_min = tf.reduce_min(label_mask)
            label_mask_max = tf.reduce_max(label_mask)
            label_mask_unique, _ = tf.unique(tf.reshape(label_mask, shape=(-1,)))
            n_label_mask_unique = tf.size(label_mask_unique)

            tf.summary.scalar('class {} label_mask_min'.format(_class_id), label_mask_min)
            tf.summary.scalar('class {} label_mask_max'.format(_class_id), label_mask_max)
            tf.summary.scalar('class {} n_label_mask_unique'.format(_class_id), n_label_mask_unique)

            label_mask3 = tf.multiply(label_mask2, 255)
            tf.summary.image('class {} label_mask'.format(_class_id), label_mask3)

            label_red2 = tf.expand_dims(label_red, axis=3)
            label_green2 = tf.expand_dims(label_green, axis=3)
            label_blue2 = tf.expand_dims(label_blue, axis=3)

            tf.summary.image('class {} label_red'.format(_class_id), label_red2)
            tf.summary.image('class {} label_green'.format(_class_id), label_green2)
            tf.summary.image('class {} label_blue'.format(_class_id), label_blue2)

            _summary_label_resized = tf.stack([label_red, label_green, label_blue], axis=3)

            pred_mask = tf.squeeze(tf.cast(tf.equal(summary_predictions_resized, _class_id), tf.uint8))
            pred_red = tf.multiply(pred_mask, _col[2])
            pred_green = tf.multiply(pred_mask, _col[1])
            pred_blue = tf.multiply(pred_mask, _col[0])

            # tf.summary.image('class {} pred_mask'.format(_class_id), pred_mask)
            # tf.summary.image('class {} pred_red'.format(_class_id), pred_red)
            # tf.summary.image('class {} pred_green'.format(_class_id), pred_green)
            # tf.summary.image('class {} pred_blue'.format(_class_id), pred_blue)

            _summary_predictions_resized = tf.stack([pred_red, pred_green, pred_blue], axis=3)

            summary_predictions_resized_list.append(_summary_predictions_resized)
            summary_label_resized_list.append(_summary_label_resized)

        # else:
        #     summary_label_resized_rgb = summary_label_resized
        #     summary_predictions_resized_rgb = summary_predictions_resized

        summary_label_resized_rgb = tf.add_n(summary_label_resized_list)
        summary_predictions_resized_rgb = tf.add_n(summary_predictions_resized_list)

        concat_tensor = tf.concat((summary_image, summary_label_resized_rgb, summary_predictions_resized_rgb), axis=2)
        tf.summary.image('img_gt_pred/%s' % common.OUTPUT_TYPE, concat_tensor)


def _train_deeplab_model(params, iterator, num_of_classes, ignore_label, class_to_color):
    """

    Trains the deeplab model.

    Args:
      iterator: An iterator of type tf.data.Iterator for images and labels.
      num_of_classes: Number of classes for the dataset.
      ignore_label: Ignore label for the dataset.

    Returns:
      train_tensor: A tensor to update the model variables.
      summary_op: An operation to log the summaries.

    :param NewDeeplabTrainParams params:
    """

    """
    """
    global_step = tf.train.get_or_create_global_step()

    learning_rate = train_utils.get_model_learning_rate(
        params.learning_policy, params.base_learning_rate,
        params.learning_rate_decay_step, params.learning_rate_decay_factor,
        params.train_steps, params.learning_power,
        params.slow_start_step, params.slow_start_learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(learning_rate, params.momentum)

    tower_losses = []
    tower_grads = []
    for i in range(params.num_clones):
        with tf.device('/gpu:%d' % i):
            # First tower has default name scope.
            name_scope = ('clone_%d' % i) if i else ''
            with tf.name_scope(name_scope) as scope:
                loss = _tower_loss(
                    params=params,
                    iterator=iterator,
                    num_of_classes=num_of_classes,
                    ignore_label=ignore_label,
                    scope=scope,
                    reuse_variable=(i != 0),
                    class_to_color=class_to_color)

                tower_losses.append(loss)

    if params.quantize_delay_step >= 0:
        if params.num_clones > 1:
            raise ValueError('Quantization doesn\'t support multi-clone yet.')
        tf.contrib.quantize.create_training_graph(
            quant_delay=params.quantize_delay_step)

    for i in range(params.num_clones):
        with tf.device('/gpu:%d' % i):
            name_scope = ('clone_%d' % i) if i else ''
            with tf.name_scope(name_scope) as scope:
                grads = optimizer.compute_gradients(tower_losses[i])
                tower_grads.append(grads)

    with tf.device('/cpu:0'):
        grads_and_vars = _average_gradients(tower_grads)

        # Modify the gradients for biases and last layer variables.
        last_layers = model.get_extra_layer_scopes(
            params.last_layers_contain_logits_only)
        grad_mult = train_utils.get_model_gradient_multipliers(
            last_layers, params.last_layer_gradient_multiplier)
        if grad_mult:
            grads_and_vars = tf.contrib.training.multiply_gradients(
                grads_and_vars, grad_mult)

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Gather update_ops. These contain, for example,
        # the updates for the batch_norm variables created by model_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)

        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        # Print total loss to the terminal.
        # This implementation is mirrored from tf.slim.summaries.

        should_log = math_ops.equal(math_ops.mod(global_step, params.log_steps), 0)
        total_loss = tf.cond(
            should_log,
            lambda: tf.Print(total_loss, [global_step, total_loss], 'global_step, total_loss:'),
            lambda: total_loss)

        tf.summary.scalar('total_loss', total_loss)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Excludes summaries from towers other than the first one.
        summary_op = tf.summary.merge_all(scope='(?!clone_)')

    return train_tensor, summary_op


def run(params):
    """

    :param NewDeeplabTrainParams params:
    :return:
    """

    paramparse.to_flags(tf.app.flags.FLAGS, params, verbose=0, allow_missing=1)

    # tf.logging.set_verbosity(tf.logging.INFO)

    os.makedirs(params.log_dir, exist_ok=1)
    print('Training on set: {} with split: {}'.format(params.dataset, params.train_split))

    classes, composite_classes = read_class_info(params.class_info_path)
    class_to_color = {i: k[1] for i, k in enumerate(classes)}

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(ps_tasks=params.num_ps_tasks)):
            assert params.train_batch_size % params.num_clones == 0, (
                'Training batch size not divisble by number of clones (GPUs).')
            clone_batch_size = params.train_batch_size // params.num_clones

            dataset = data_generator.Dataset(
                dataset_name=params.dataset,
                split_name=params.train_split,
                dataset_dir=params.dataset_dir,
                batch_size=clone_batch_size,
                crop_size=[int(sz) for sz in params.train_crop_size],
                min_resize_value=params.min_resize_value,
                max_resize_value=params.max_resize_value,
                resize_factor=params.resize_factor,
                min_scale_factor=params.min_scale_factor,
                max_scale_factor=params.max_scale_factor,
                scale_factor_step_size=params.scale_factor_step_size,
                model_variant=params.model_variant,
                num_readers=1,
                is_training=True,
                should_shuffle=True,
                should_repeat=True)

            # samples = dataset.get_one_shot_iterator().get_next()

            train_tensor, summary_op = _train_deeplab_model(params, dataset.get_one_shot_iterator(),
                                                            dataset.num_of_classes,
                                                            dataset.ignore_label,
                                                            class_to_color)

            # Soft placement allows placing on CPU ops without GPU implementation.
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
            session_config.gpu_options.allow_growth = params.allow_memory_growth
            session_config.gpu_options.per_process_gpu_memory_fraction = params.gpu_memory_fraction
            last_layers = model.get_extra_layer_scopes(
                params.last_layers_contain_logits_only)
            init_fn = None
            if params.tf_initial_checkpoint:
                init_fn = train_utils.get_model_init_fn(
                    params.checkpoint_dir,
                    params.tf_initial_checkpoint,
                    params.initialize_last_layer,
                    last_layers,
                    ignore_missing_vars=True)

            scaffold = tf.train.Scaffold(
                init_fn=init_fn,
                summary_op=summary_op,
            )

            stop_hook = tf.train.StopAtStepHook(
                last_step=params.train_steps)

            profile_dir = params.profile_logdir
            if profile_dir is not None:
                tf.gfile.MakeDirs(profile_dir)

            print('saving checkpoints to: {}'.format(params.checkpoint_dir))
            print('saving TB summary to: {}'.format(params.tb_dir))

            with tf.contrib.tfprof.ProfileContext(
                    enabled=profile_dir is not None, profile_dir=profile_dir):
                with tf.train.MonitoredTrainingSession(
                        master=params.master,
                        is_chief=(params.task == 0),
                        config=session_config,
                        scaffold=scaffold,
                        checkpoint_dir=params.checkpoint_dir,
                        summary_dir=params.tb_dir,
                        log_step_count_steps=params.log_steps,
                        save_summaries_steps=params.save_summaries_secs,
                        save_checkpoint_secs=params.save_interval_secs,
                        hooks=[stop_hook]) as sess:
                    while not sess.should_stop():
                        sess.run([train_tensor])


if __name__ == '__main__':
    _params = NewDeeplabTrainParams()
    paramparse.process(_params)

    _params.process()
    run(_params)
