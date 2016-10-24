
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pathlib import Path

import math
import matplotlib
import numpy as np
import os.path
import tensorflow as tf
import time
from model_train import deepID
from libs.tfpipeline import input_pipeline



# Do not use a gui toolkit for matlotlib.
matplotlib.use('Agg')

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('checkpoint_dir', 'models/',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 3466,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """Number of examples per batch.""")
tf.app.flags.DEFINE_string('data_txt', 'tftest.txt',
                           """The text file containing test data path and annotations.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', 'the device to eval on.')

def normalized_rmse(pred, gt_truth):
    # TODO: assert shapes
    #       remove 5
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 0, :] - gt_truth[:, 1, :])**2), 1))

    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 5)




def _eval_once(saver, rmse_op, network):
  """Runs Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    rmse_op: rmse_op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:

      saver.restore(sess, ckpt.model_checkpoint_path)


      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return
    test_x, test_label = input_pipeline(['tftest.txt'], batch_size=FLAGS.batch_size, shape=[39, 39, 1], is_training=False)
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      errors = []

      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), 'tf/'))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        test_xs, label = sess.run([test_x, test_label])
        rmse = sess.run(rmse_op, feed_dict={network['x']: test_xs, network['y']: label, network['train']: False,
                network['keep_prob']: 0.5})
        errors.append(rmse)
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      errors = np.vstack(errors).ravel()
      mean_rmse = errors.mean()
      auc_at_08 = (errors < .08).mean()
      auc_at_05 = (errors < .05).mean()



      print('Errors', errors.shape)
      print('%s: mean_rmse = %.4f, auc @ 0.05 = %.4f, auc @ 0.08 = %.4f [%d examples]' %
            (datetime.now(), errors.mean(), auc_at_05, auc_at_08, total_sample_count))


    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    



def evaluate(shape=[39, 39, 1]):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    train_dir = Path(FLAGS.checkpoint_dir)
    
    images, landmarks = input_pipeline(
            [FLAGS.data_txt], batch_size=2,
            shape=shape, is_training=False)

    # mirrored_images, _, mirrored_inits, shapes = data_provider.batch_inputs(
    #     [dataset_path], reference_shape,
    #     batch_size=FLAGS.batch_size, is_training=False, mirror_image=True)

    print('Loading model...')
    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.device(FLAGS.device):
        deepid = deepID(input_shape=[None, 39, 39, 1], n_filters=[20, 40, 60, 80], 
            filter_sizes=[4, 3, 3, 2], activation=tf.nn.relu, dropout=False)

        tf.get_variable_scope().reuse_variables()



    avg_pred = deepid['pred']
    gt_truth = deepid['y']
    gt_truth = tf.reshape(gt_truth, (-1, 5, 2))
    # Calculate predictions.
    norm_error = normalized_rmse(avg_pred, gt_truth)

    # Restore the moving average version of the learned variables for eval.
    # variable_averages = tf.train.ExponentialMovingAverage(
        # 0.9999)
    # variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()


    while True:
      _eval_once(saver, norm_error, deepid)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
    evaluate()