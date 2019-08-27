import os
import sys
import tensorflow as tf
import numpy as np
import re
from glob import glob
import pandas

from six.moves import xrange
from random import shuffle


class CAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.data = glob(os.path.join("./", 'wikiart',
                                      '*.jpg'))
        self.sample_size = 64
        self.batch_size = 64
        self.epoch = 100

        self.d_learning_rate = 2e-4
        self.g_learning_rate = 2e-4

        self.d_decay = 0.6
        self.g_decay = 0.6

        self.label_dim = 137  # wikiart class num
        self.random_noise_dim = 100

        self.input_size = 512
        self.output_size = 512

        self.sample_dir = 'samples'

        # self.checkpoint_dir = 'drive/My Drive/checkpoint'
        self.checkpoint_dir = 'drive/My Drive/high_resolution/new_checkpoint'

        self.checkpint_dir_model = 'wikiart'
        self.data_dir = 'data'

        # self.tensorboard_dir = 'drive/My Drive/tensorboard/log'
        self.tensorboard_dir = 'tensorboard/high_resolution_g_image'
        ## get label(classification) data
        self.csv_file_path = '/content/wikiart/all_data_info.csv'

        self.df = pandas.read_csv(self.csv_file_path)
        self.label_dict = self.df['style'].unique()
        self.label_dict = dict(enumerate(self.label_dict))

        self.label_dict = dict((v, k) for k, v in self.label_dict.items())

        print(self.label_dict)

        ## Check required directory and make directory
        if not os.path.exists(self.checkpoint_dir):
            print('NO checkpoint directory => Making checkpoint directory')
            print('\nMake directory in drive first')
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.sample_dir):
            print('NO sample directory => Making sample directory')
            os.makedirs(self.sample_dir)

        if not os.path.exists(self.tensorboard_dir):
            print('No tensorboard summary directory => Making directory')
            os.makedirs(self.tensorboard_dir)

        if not os.path.exists(self.data_dir) or not self.data:
            # print(self.data)
            print('\nPROCESS END')
            print('WARNING: No data directory or No image data')
            sys.exit(1)

    def build_model(self):
        ## Creating a variable
        self.y = tf.placeholder(tf.float32, [None, self.label_dim], name='y')
        self.real_image = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3],
                                         name='real_images')
        self.random_noise = tf.placeholder(tf.float32, [None, self.random_noise_dim], name='random_noise')

        #### tensorboard
        self.random_noise_summary = tf.summary.histogram("random_noise_summary", self.random_noise)
        # z_sum

        ##  Building model
        # Creating generator / discriminator
        self.generator = self.generator(self.random_noise)

        self.discriminator_police_sigmoid, self.discriminator_police, self.discriminator_police_class_softmax, self.discriminator_police_class = self.discriminator(
            self.real_image, reuse=False)
        self.discriminator_thief_sigmoid, self.discriminator_thief, self.discriminator_thief_class_softmax, self.discriminator_thief_class = self.discriminator(
            self.generator, reuse=True)

        #### tensorboard

        # discriminator real image summary
        self.discriminator_police_summary = tf.summary.histogram("discriminator_police_summary",
                                                                 self.discriminator_police_sigmoid)
        # discriminator real image class summary
        self.discriminator_police_class_summary = tf.summary.histogram("discriminator_police_class_summary",
                                                                       self.discriminator_police_class_softmax)
        # discriminator fake image summary
        self.discriminator_thief_summary = tf.summary.histogram("discriminator_thief_summary",
                                                                self.discriminator_thief_sigmoid)
        # discriminator fake image class summary
        self.discriminator_thief_class_summary = tf.summary.histogram("discriminator_thief_class_summary",
                                                                      self.discriminator_thief_class_softmax)
        # generator summary
        self.generator_summary = tf.summary.image("generator_summary", self.generator)

        # sampler summary
        # self.sampler_summary = tf.summary.image('sampler_summary', self.sampler)

        ## Find Accuracy
        # classifcation real_label
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.discriminator_police_class, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ## Creating loss function - Find cost
        # real image discriminator cost
        self.discriminator_police_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.discriminator_police,
            labels=tf.ones_like(self.discriminator_police_sigmoid) * 0.9))

        # fake image discriminator cost
        self.discriminator_thief_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.discriminator_thief,
            labels=tf.zeros_like(self.discriminator_thief_sigmoid)))

        # real image discriminator classification cost
        self.discriminator_loss_class_real = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.discriminator_police_class,
            labels=1.0 * self.y))

        # generator image discriminator classification cost
        self.generator_loss_class_fake = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.discriminator_thief_class,
            labels=(1.0 / self.label_dim) *
                   tf.ones_like(self.discriminator_thief_class_softmax)))

        # generator cost
        self.generator_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator_thief,
                                                    labels=tf.ones_like(self.discriminator_thief_sigmoid)))

        # Generator fake image cost

        # self.generator_loss_fake = -tf.reduce_mean(tf.log(self.discriminator_thief_sigmoid))
        # generator, discriminator loss

        # Total generator loss
        self.generator_loss = self.generator_loss_fake + 1.0 * self.generator_loss_class_fake
        #
        self.discriminator_loss = self.discriminator_police_loss + self.discriminator_thief_loss + self.discriminator_loss_class_real  # 1 + 0 + 1 = 2

        #### tensorboard
        self.discriminator_police_loss_summary = tf.summary.scalar("discriminator_police_loss_summary",
                                                                   self.discriminator_police_loss)
        # d_loss_real_sum

        self.discriminator_thief_loss_summary = tf.summary.scalar("discriminator_thief_loss_summary",
                                                                  self.discriminator_thief_loss)
        # d_loss_fake_sum

        self.discriminator_police_class_loss_summary = tf.summary.scalar("discriminator_police_class_loss",
                                                                         self.discriminator_loss_class_real)
        # d_loss_class_real_sum
        self.generator_loss_class_fake_summary = tf.summary.scalar("generator_loss_class_fake",
                                                                   self.generator_loss_class_fake)
        # g_loss_class_fake_sum

        self.generator_loss_summary = tf.summary.scalar("generator_loss_summary", self.generator_loss)
        # g_loss_sum
        self.discriminator_loss_summary = tf.summary.scalar("discriminator_loss_summary", self.discriminator_loss)
        # d_loss_sum

        t_vars = tf.trainable_variables()
        self.discriminator_vars = [var for var in t_vars if 'd_' in var.name]
        self.generator_vars = [var for var in t_vars if 'g_' in var.name]
        # Creating checkpoint saver
        self.saver = tf.train.Saver()

    def train(self):
        # Creating Optimizer
        discriminator_optimizer = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.d_decay).minimize(
            self.discriminator_loss,
            var_list=self.discriminator_vars)
        generator_optimizer = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.g_decay).minimize(
            self.generator_loss,
            var_list=self.generator_vars)

        #### tensorboard
        generator_optimizer_summary = tf.summary.merge(
            [self.random_noise_summary, self.discriminator_thief_summary, self.generator_summary,
             self.discriminator_thief_loss_summary, self.generator_loss_summary])

        discriminator_optimizer_summary = tf.summary.merge(
            [self.random_noise_summary, self.discriminator_police_summary,
             self.discriminator_police_loss_summary, self.discriminator_loss_summary,
             self.discriminator_police_class_loss_summary, self.generator_loss_class_fake_summary])

        # Writing tensorboard
        self.writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        tf.global_variables_initializer().run()

        ## Creating sample -> test part
        # sample_random_noise = np.random.normal(0, 1, [self.sample_size, self.random_noise_dim]).astype(np.float32)

        # Convert (256,256) images into (256,256,3)

        shuffle(self.data)
        # sample_images_path = self.data[0: self.sample_size]
        # sample_images_ = [get_image(sample_image_path,
        #                             input_height=self.input_size,
        #                             input_width=self.input_size,
        #                             resize_height=self.output_size,
        #                             resize_width=self.output_size,
        #                             crop=False) for sample_image_path in sample_images_path]
        #
        # sample_images = np.array(sample_images_).astype(np.float32)
        # sample_labels = get_y(sample_images_path, self.label_dim, self.label_dict, self.df)  # get label(classification)

        # checkpoint variable
        counter = 1

        # checkpoint load
        checkpoint_dir_path = os.path.join(self.checkpoint_dir, self.checkpint_dir_model)
        could_load, checkpoint_counter = checkpoint_load(self.sess, self.saver, self.checkpoint_dir,
                                                         self.checkpint_dir_model)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## training
        for epoch in xrange(self.epoch):
            shuffle(self.data)

            batch_index = min(len(self.data), np.inf) // self.batch_size
            print(batch_index)
            for index in xrange(0, batch_index):
                ## Creating batch -> training part
                batch_images_path = self.data[index * self.batch_size: (index + 1) * self.batch_size]
                batch_images_ = [get_image(batch_image_path,
                                           input_height=self.input_size,
                                           input_width=self.input_size,
                                           resize_height=self.output_size,
                                           resize_width=self.output_size,
                                           crop=False) for batch_image_path in batch_images_path]

                batch_images = np.array(batch_images_).astype(np.float32)

                batch_labels = get_y(batch_images_path, self.label_dim, self.label_dict,
                                     self.df)  # get label(classification)
                batch_random_noise = np.random.normal(0, 1, [self.batch_size, self.random_noise_dim]).astype(np.float32)

                ## Update
                # Update D network
                _, summary = self.sess.run([discriminator_optimizer, discriminator_optimizer_summary],
                                           feed_dict={self.real_image: batch_images,
                                                      self.random_noise: batch_random_noise,
                                                      self.y: batch_labels})
                self.writer.add_summary(summary, counter)

                # Update G network
                _, summary = self.sess.run([generator_optimizer, generator_optimizer_summary],
                                           feed_dict={self.random_noise: batch_random_noise})
                self.writer.add_summary(summary, counter)

                errD_fake = self.discriminator_thief_loss.eval(
                    {self.random_noise: batch_random_noise, self.y: batch_labels})
                errD_real = self.discriminator_police_loss.eval({self.real_image: batch_images, self.y: batch_labels})

                # change
                # errG = self.generator_loss.eval({self.random_noise: batch_random_noise })
                errG = self.generator_loss.eval({self.random_noise: batch_random_noise, self.y: batch_labels})

                # Find cost value
                errD_class_real = self.discriminator_loss_class_real.eval(
                    {self.real_image: batch_images, self.y: batch_labels})
                errG_class_fake = self.generator_loss_class_fake.eval(
                    {self.real_image: batch_images, self.random_noise: batch_random_noise})
                accuracy = self.accuracy.eval({self.real_image: batch_images, self.y: batch_labels})

                # global value --> checkpoint value
                counter += 1
                print("Epoch: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f" % (
                    epoch, index, batch_index, errD_fake + errD_real + errD_class_real, errG))
                print("Discriminator class acc: %.2f" % (accuracy))

                ## image save
                if np.mod(counter, 300) == 1:
                    try:

                        # samples = self.sess.run(self.sampler, feed_dict={self.random_noise: sample_random_noise,
                        #                                                  self.real_image: sample_images,
                        #                                                  self.y: sample_labels})
                        # # save_images(samples, image_manifold_size(samples.shape[0]),
                        # #             './{}/train_{:02d}_{:04d}.png'.format('samples', epoch, index))
                        # print(samples.shape)
                        # save_single_image(samples,
                        #                   './{}/new_sampler_train_{:02d}_{:04d}.png'.format('samples', epoch, index))
                        samples = self.sess.run(self.generator(self.random_noise))
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format('samples', epoch, index))
                        print("[SAVE IMAGE]")
                    except Exception as e:
                        print("image save error! ", e)
                        # checkpoint save
                if np.mod(counter, 500) == 1:
                    print("[SAVE CHECKPOINT]")
                    checkpoint_save(self.sess, self.saver, checkpoint_dir_path, counter)
            # TODO:: implement move tensorboard dir to dirve
            print('{}_epoch ends', format(epoch))

    ## discriminator

    def discriminator(self, input_, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()  # for reusing variables
            # ! padding -> SAME -> VALID => ops.py
            discriminator_layer0 = lrelu(conv2d(input_, 32, name='d_h0_conv'))  # [512, 512, 3], 32 => (128, 128, 32)
            discriminator_layer1 = lrelu(
                batch_norm(conv2d(discriminator_layer0, 64, name='d_h1_conv'), 'd_bn1'))  # (?, 64, 64, 64)
            discriminator_layer2 = lrelu(
                batch_norm(conv2d(discriminator_layer1, 128, name='d_h2_conv'), 'd_bn2'))  # (?, 32, 32, 128)
            discriminator_layer3 = lrelu(
                batch_norm(conv2d(discriminator_layer2, 256, name='d_h3_conv'), 'd_bn3'))  # (?, 16, 16, 256)
            discriminator_layer4 = lrelu(
                batch_norm(conv2d(discriminator_layer3, 512, name='d_h4_conv'), 'd_bn4'))  # (?, 8, 8, 512)
            discriminator_layer5 = lrelu(
                batch_norm(conv2d(discriminator_layer4, 512, name='d_h5_conv'), 'd_bn5'))  # (?, 4, 4, 512)

            shape = np.product(discriminator_layer5.get_shape()[1:].as_list())  #
            discriminator_layer6 = tf.reshape(discriminator_layer5, [-1, shape])  #
            discriminator_output = linear(discriminator_layer6, 1, 'd_ro_lin')  # (?, 1)

            discriminator_layer7 = lrelu(linear(discriminator_layer6, 1024, 'd_h8_lin'))  #
            discriminator_layer8 = lrelu(linear(discriminator_layer7, 512, 'd_h9_lin'))  #
            discriminator_class_output = linear(discriminator_layer8, self.label_dim, 'd_co_lin')  #
            discriminator_class_output_softmax = tf.nn.softmax(discriminator_class_output)  # (?, self.label_dim)

            return tf.nn.sigmoid(
                discriminator_output), discriminator_output, discriminator_class_output_softmax, discriminator_class_output

        ## generator

    def generator(self, random_noise):
        with tf.variable_scope("generator") as scope:
            generator_linear = linear(random_noise, 64 * 4 * 4 * 16, 'g_h0_lin')  # ([?, 100], 16,384])
            generator_reshape = tf.reshape(generator_linear, [-1, 4, 4, 64 * 16])  # (?, 4, 4, 1024)
            generator_input = tf.nn.relu(batch_norm(generator_reshape, 'g_bn0'))  # (?, 4, 4, 1024)

            generator_layer1 = deconv2d(generator_input, [self.batch_size, 8, 8, 64 * 16],
                                        name='g_layer1')  # (?, 8, 8, 1024)
            generator_layer1 = tf.nn.relu(batch_norm(generator_layer1, 'g_bn1'))  # (?, 8, 8, 1024)

            generator_layer2 = deconv2d(generator_layer1, [self.batch_size, 16, 16, 64 * 8],
                                        name='g_layer2')  # (?, 16, 16, 512)
            generator_layer2 = tf.nn.relu(batch_norm(generator_layer2, 'g_bn2'))  # (?, 16, 16, 512)

            generator_layer3 = deconv2d(generator_layer2, [self.batch_size, 32, 32, 64 * 4],
                                        name='g_layer3')  # (?, 32, 32, 256)
            generator_layer3 = tf.nn.relu(batch_norm(generator_layer3, 'g_bn3'))  # (?, 32, 32, 256)

            generator_layer4 = deconv2d(generator_layer3, [self.batch_size, 64, 64, 64 * 2],
                                        name='g_layer4')  # (?, 64, 64, 128)
            generator_layer4 = tf.nn.relu(batch_norm(generator_layer4, 'g_bn4'))  # (?, 64, 64, 128)

            generator_layer5 = deconv2d(generator_layer4, [self.batch_size, 128, 128, 64],
                                        name='g_layer5')  # (?, 128, 128, 64)
            generator_layer5 = tf.nn.relu(batch_norm(generator_layer5, 'g_bn5'))  # (?, 128, 128, 64)

            generator_layer6 = deconv2d(generator_layer5, [self.batch_size, 256, 256, 3],
                                        name='g_layer6')  # (?, 256, 256, 3)
            generator_layer6 = tf.nn.relu(batch_norm(generator_layer6, 'g_bn6'))
            generator_output = deconv2d(generator_layer6, [self.batch_size, 512, 512, 3], name='g_output')
            generator_output = tf.nn.tanh(generator_output)  # (?, 512, 512, 3)

            return generator_output  # (?, 512, 512, 3)

## sampler
# def sampler(self, random_noise):

#     with tf.variable_scope("generator", reuse=True) as scope:
#         scope.reuse_variables()
#
#         sampler_linear = linear(random_noise, 64 * 4 * 4 * 16, 'g_h0_lin')  # ([?, 100], 16,384])
#         sampler_reshape = tf.reshape(sampler_linear, [-1, 4, 4, 64 * 16])  # (?, 4, 4, 1024)
#         sampler_input = tf.nn.relu(batch_norm(sampler_reshape, 'g_bn0', train=False))  # (?, 4, 4, 1024)
#
#         sampler_layer1 = deconv2d(sampler_input, [self.batch_size, 8, 8, 64 * 16],
#                                   name='g_layer1')  # (?, 8, 8, 1024)
#         sampler_layer1 = tf.nn.relu(batch_norm(sampler_layer1, 'g_bn1', train=False))  # (?, 8, 8, 1024)
#
#         sampler_layer2 = deconv2d(sampler_layer1, [self.batch_size, 16, 16, 64 * 8],
#                                   name='g_layer2')  # (?, 16, 16, 512)
#         sampler_layer2 = tf.nn.relu(batch_norm(sampler_layer2, 'g_bn2', train=False))  # (?, 16, 16, 512)
#
#         sampler_layer3 = deconv2d(sampler_layer2, [self.batch_size, 32, 32, 64 * 4],
#                                   name='g_layer3')  # (?, 32, 32, 256)
#         sampler_layer3 = tf.nn.relu(batch_norm(sampler_layer3, 'g_bn3', train=False))  # (?, 32, 32, 256)
#
#         sampler_layer4 = deconv2d(sampler_layer3, [self.batch_size, 64, 64, 64 * 2],
#                                   name='g_layer4')  # (?, 64, 64, 128)
#         sampler_layer4 = tf.nn.relu(batch_norm(sampler_layer4, 'g_bn4', train=False))  # (?, 64, 64, 128)
#
#         sampler_layer5 = deconv2d(sampler_layer4, [self.batch_size, 128, 128, 64],
#                                   name='g_layer5')  # (?, 128, 128, 64)
#         sampler_layer5 = tf.nn.relu(batch_norm(sampler_layer5, 'g_bn5', train=False))  # (?, 128, 128, 64)
#
#         sampler_layer6 = deconv2d(sampler_layer5, [self.batch_size, 256, 256, 3],
#                                   name='g_layer6')  # (?, 256, 256, 3)
#
#         sampler_layer6 = tf.nn.relu(batch_norm(sampler_layer6, 'g_bn6'))
#         sampler_output = deconv2d(sampler_layer6, [self.batch_size, 512, 512, 3], name='g_output')
#         sampler_output = tf.nn.tanh(sampler_output)  # (?, 512, 512, 3)
#
#         sampler_output = sampler_output[:1, :, :]
#
#         return sampler_output  # (1, 512, 512, 3)
