from CAN import CAN
import tensorflow as tf

# import
#
#
#


def main(_):
    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=run_config) as session:
        can = CAN(session)
        can.build_model()
        noise = tf.random.normal([None, 100], name="noise")
        sample_batches = can.sampler(noise)
        print(sample_batches)


if __name__ == "__main__":
    tf.compat.v1.app.run(main=main)
