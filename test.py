# Sampler for CAN

import tensorflow as tf
from CAN import CAN
import numpy as np


def main(_):
    batch_size = 64
    noise_dim = 100
    tensorboard_dir = 'tensorboard'
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    # tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        can = CAN(sess)
        # Building Generator Model from CAN
        tf_random_noise = tf.placeholder(shape=[None, noise_dim], dtype=tf.float32)
        g_model = can.generator(tf_random_noise)

        generator_summary = tf.summary.image("generator_summary", g_model)
        checkpoint_dir = 'checkpoint/wikiart/'
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.Saver()
        saver.restore(sess, latest_checkpoint)
        random_noise = np.random.normal(0, 1, [batch_size, noise_dim]).astype(np.float32)
        _, summary = sess.run([g_model, generator_summary], feed_dict={
            tf_random_noise: random_noise
        })
        writer.add_summary(summary)


if __name__ == '__main__':
    tf.app.run()
