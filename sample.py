
import tensorflow as tf
# from DCGAN import *
from CAN import CAN


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        can = CAN(sess, True)
        can.build_model()
        can.sampler()


if __name__ == '__main__':
    tf.app.run()
