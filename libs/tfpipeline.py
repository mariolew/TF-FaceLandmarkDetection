import tensorflow as tf
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
TXTs = ['tftest_vae.txt']



def read_my_file_format(filename):
    

    record_defaults = [[""]] + [[1.0]] * 10
    components = tf.decode_csv(filename, record_defaults=record_defaults, 
        field_delim=" ")
    imgName = components[0]
    features = components[1:]
    img_contents = tf.read_file(imgName)
    img = tf.image.decode_jpeg(img_contents, channels=1)
    return img, features

def processImage(img):
    """
        process images before feeding to CNNs
        imgs: W x H x 1
    """
    img = img.astype(np.float32)
    m = img.mean()
    s = img.std()
    img = (img - m) / s
    return img

def input_pipeline(TXTs, batch_size, shape, is_training=False):

    filename_queue = tf.train.string_input_producer(TXTs, shuffle=is_training)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    img, features = read_my_file_format(value)
    img.set_shape(shape)
    img_reshape = tf.cast(img, tf.float32)
    # float_image = tf.py_func(processImage, [img_reshape], [tf.float32])[0]
    # float_image.set_shape(shape)
    float_image = tf.image.per_image_whitening(img_reshape)
    # if is_training:
    #     float_image = distort_color(float_image)
    # img_batch, label_batch = tf.train.batch([float_image, features], batch_size=batch_size)
    min_after_dequeue = 80000 // 100

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (2 + 1) * batch_size

    # Randomize the order and output batches of batch_size.
    img_batch, label_batch = tf.train.shuffle_batch([float_image, features],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=2)
    # img_batch, label_batch = tf.train.batch([float_image, features], batch_size=batch_size)
    return img_batch, label_batch

def distort_color(image, thread_id=0, stddev=0.1, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        image += tf.random_normal(
                tf.shape(image),
                stddev=stddev,
                dtype=tf.float32,
                seed=42,
                name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
# shape = [64, 64, 1]
# im_batch, label_batch = input_pipeline(TXTs, 1, shape)
# with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    im, feat = sess.run([im_batch, label_batch])
#    print(feat[0])
#    plt.imshow(im[0].reshape((39,39)))
#    import pdb; pdb.set_trace()
#    coord.request_stop()
#    coord.join(threads)
