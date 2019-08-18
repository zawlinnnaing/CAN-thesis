#############################################
# Date          : 2018.03.25
# Programmer    : Seounggyu Kim
# description   : CAN 모델
# Update Date   : 2018.04.21
# Update        : 텐서 보드 추가
#############################################

import sys
from glob import glob
from random import shuffle

import pandas

from ops import *
from utils import *

data = glob(os.path.join("./data", 'wikiart',
                         '*.jpg'))  ########## '*/*.jpg' reading all files (images in directories)

sample_size = 32
batch_size = 32
epoch = 100

label_dim = 27  # wikiart class num
random_noise_dim = 100

input_size = 256
output_size = 256

sample_dir = 'samples'
checkpoint_dir = 'checkpoint'
checkpint_dir_model = 'wikiart'
data_dir = 'data'

real_image = tf.placeholder(tf.float32, [batch_size, 256, 256, 3],
                            name='real_images')
random_noise = tf.placeholder(tf.float32, [None, random_noise_dim], name='random_noise')

y = tf.placeholder(tf.float32, [None, 27], name='y')

# TODO:: modify this to suit new dataset from kaggle

# get label(classification) data
# label_dict = {}
# path_list = glob('./data/wikiart/**/', recursive=True)[1:]
# print('!!!!!11', path_list)
# for i, elem in enumerate(path_list):
#     self.label_dict[elem[15:-1]] = i

csv_file_path = '/content/data/wikiart/all_data_info.csv'

df = pandas.read_csv(csv_file_path)
label_dict = df['style'].unique()
label_dict = dict(enumerate(label_dict))

print(label_dict)
# Check required directory and make directory
if not os.path.exists(checkpoint_dir):
    print('NO checkpoint directory => Make checkpoint directory')
    os.makedirs(checkpoint_dir)

if not os.path.exists(sample_dir):
    print('NO sample directory => Make sample directory')
    os.makedirs(sample_dir)

if not os.path.exists(data_dir) or not data:
    # print(self.data)
    print('\nPROCESS END . ')
    print('Reason: No data directory or No image data')
    sys.exit(1)


class Model(object):
    pass


# discriminator

def discriminator(input_, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()  # when you share data
        # ! padding -> SAME -> VALID => ops.py (in file)
        discriminator_layer0 = lrelu(conv2d(input_, 32, name='d_h0_conv'))  # [256, 256, 3], 32 => (128, 128, 32)
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
        discriminator_class_output = linear(discriminator_layer8, 27, 'd_co_lin')  #
        discriminator_class_output_softmax = tf.nn.softmax(discriminator_class_output)  # (?, 27)

        return tf.nn.sigmoid(
            discriminator_output), discriminator_output, discriminator_class_output_softmax, discriminator_class_output


# generator

def generator(random_noise):
    with tf.variable_scope("generator") as scope:
        generator_linear = linear(random_noise, 64 * 4 * 4 * 16, 'g_h0_lin')  # ([?, 100], 16,384])
        generator_reshape = tf.reshape(generator_linear, [-1, 4, 4, 64 * 16])  # (?, 4, 4, 1024)
        generator_input = tf.nn.relu(batch_norm(generator_reshape, 'g_bn0'))  # (?, 4, 4, 1024)

        generator_layer1 = deconv2d(generator_input, [batch_size, 8, 8, 64 * 16],
                                    name='g_layer1')  # (?, 8, 8, 1024)
        generator_layer1 = tf.nn.relu(batch_norm(generator_layer1, 'g_bn1'))  # (?, 8, 8, 1024)

        generator_layer2 = deconv2d(generator_layer1, [batch_size, 16, 16, 64 * 8],
                                    name='g_layer2')  # (?, 16, 16, 512)
        generator_layer2 = tf.nn.relu(batch_norm(generator_layer2, 'g_bn2'))  # (?, 16, 16, 512)

        generator_layer3 = deconv2d(generator_layer2, [batch_size, 32, 32, 64 * 4],
                                    name='g_layer3')  # (?, 32, 32, 256)
        generator_layer3 = tf.nn.relu(batch_norm(generator_layer3, 'g_bn3'))  # (?, 32, 32, 256)

        generator_layer4 = deconv2d(generator_layer3, [batch_size, 64, 64, 64 * 2],
                                    name='g_layer4')  # (?, 64, 64, 128)
        generator_layer4 = tf.nn.relu(batch_norm(generator_layer4, 'g_bn4'))  # (?, 64, 64, 128)

        generator_layer5 = deconv2d(generator_layer4, [batch_size, 128, 128, 64],
                                    name='g_layer5')  # (?, 128, 128, 64)
        generator_layer5 = tf.nn.relu(batch_norm(generator_layer5, 'g_bn5'))  # (?, 128, 128, 64)

        generator_output = deconv2d(generator_layer5, [batch_size, 256, 256, 3],
                                    name='g_output')  # (?, 256, 256, 3)
        generator_output = tf.nn.tanh(generator_output)  # (?, 256, 256, 3)

        return generator_output  # (?, 256, 256, 3)


## sampler

def sampler(random_noise):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
        scope.reuse_variables()

        sampler_linear = linear(random_noise, 64 * 4 * 4 * 16, 'g_h0_lin')  # ([?, 100], 16,384])
        sampler_reshape = tf.reshape(sampler_linear, [-1, 4, 4, 64 * 16])  # (?, 4, 4, 1024)
        sampler_input = tf.nn.relu(batch_norm(sampler_reshape, 'g_bn0', train=False))  # (?, 4, 4, 1024)

        sampler_layer1 = deconv2d(sampler_input, [batch_size, 8, 8, 64 * 16],
                                  name='g_layer1')  # (?, 8, 8, 1024)
        sampler_layer1 = tf.nn.relu(batch_norm(sampler_layer1, 'g_bn1', train=False))  # (?, 8, 8, 1024)

        sampler_layer2 = deconv2d(sampler_layer1, [batch_size, 16, 16, 64 * 8],
                                  name='g_layer2')  # (?, 16, 16, 512)
        sampler_layer2 = tf.nn.relu(batch_norm(sampler_layer2, 'g_bn2', train=False))  # (?, 16, 16, 512)

        sampler_layer3 = deconv2d(sampler_layer2, [batch_size, 32, 32, 64 * 4],
                                  name='g_layer3')  # (?, 32, 32, 256)
        sampler_layer3 = tf.nn.relu(batch_norm(sampler_layer3, 'g_bn3', train=False))  # (?, 32, 32, 256)

        sampler_layer4 = deconv2d(sampler_layer3, [batch_size, 64, 64, 64 * 2],
                                  name='g_layer4')  # (?, 64, 64, 128)
        sampler_layer4 = tf.nn.relu(batch_norm(sampler_layer4, 'g_bn4', train=False))  # (?, 64, 64, 128)

        sampler_layer5 = deconv2d(sampler_layer4, [batch_size, 128, 128, 64],
                                  name='g_layer5')  # (?, 128, 128, 64)
        sampler_layer5 = tf.nn.relu(batch_norm(sampler_layer5, 'g_bn5', train=False))  # (?, 128, 128, 64)

        sampler_output = deconv2d(sampler_layer5, [batch_size, 256, 256, 3],
                                  name='g_output')  # (?, 256, 256, 3)
        sampler_output = tf.nn.tanh(sampler_output)  # (?, 256, 256, 3)

        return sampler_output  # (?, 256, 256, 3)


def build_model():
    """

    :rtype: object
    """
    model = Model()
    # Creating a variable

    # (?,256,256,3)

    # tensorboard
    model.random_noise_summary = tf.summary.histogram("random_noise_summary", random_noise)
    # z_sum

    #  build model
    # Creating generator / discriminator
    model.generator = generator(random_noise)

    # Discriminator for real image
    discriminator_police_sigmoid, discriminator_police, discriminator_police_class_softmax, discriminator_police_class = discriminator(
        real_image, reuse=False)

    # Discriminator for fake image (generated by generator)
    discriminator_thief_sigmoid, discriminator_thief, discriminator_thief_class_softmax, discriminator_thief_class = discriminator(
        generator, reuse=True)
    model.sampler = sampler(random_noise)

    #### tensorboard
    model.discriminator_police_summary = tf.summary.histogram("discriminator_police_summary",
                                                              discriminator_police_sigmoid)
    # d_sum

    model.discriminator_police_class_summary = tf.summary.histogram("discriminator_police_class_summary",
                                                                    discriminator_police_class_softmax)
    # d_c_sum

    model.discriminator_thief_summary = tf.summary.histogram("discriminator_thief_summary",
                                                             discriminator_thief_sigmoid)
    # d__sum
    model.discriminator_thief_class_summary = tf.summary.histogram("discriminator_thief_class_summary",
                                                                   discriminator_thief_class_softmax)
    # d_c__sum
    model.generator_summary = tf.summary.image("generator_summary", generator)
    # G_sum

    ## Find Accuracy
    # classification real_label and real discriminator labels
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(discriminator_police_class, 1))
    model.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Creating loss function - Find cost
    # real discriminator cost
    model.discriminator_police_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_police,
        labels=tf.ones_like(discriminator_police_sigmoid)))

    # fake discriminator cost
    model.discriminator_thief_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_thief,
        labels=tf.ones_like(discriminator_thief_sigmoid)))

    # style classification_discriminator cost
    model.discriminator_loss_class_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=discriminator_police_class,
        labels=1.0 * y))

    # generator style classification cost
    model.generator_loss_class_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=discriminator_thief_class,
        labels=(1.0 / 27) *
               tf.ones_like(discriminator_thief_class_softmax)))

    # generator cost
    generator_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_thief,
                                                labels=tf.ones_like(discriminator_thief_sigmoid)))

    # generator, discriminator  total loss
    model.generator_loss = generator_loss_fake + 1.0 * model.generator_loss_class_fake  #
    model.discriminator_loss = model.discriminator_police_loss + model.discriminator_thief_loss + model.discriminator_loss_class_real  # 1 + 0 + 1 = 2

    ''' Tensorboard '''
    model.discriminator_police_loss_summary = tf.summary.scalar("discriminator_police_loss_summary",
                                                                model.discriminator_police_loss)
    # d_loss_real_sum

    model.discriminator_thief_loss_summary = tf.summary.scalar("discriminator_thief_loss_summary",
                                                               model.discriminator_thief_loss)
    # d_loss_fake_sum

    model.discriminator_police_class_loss_summary = tf.summary.scalar("discriminator_police_class_loss",
                                                                      model.discriminator_loss_class_real)
    # d_loss_class_real_sum
    model.generator_loss_class_fake_summary = tf.summary.scalar("generator_loss_class_fake",
                                                                model.generator_loss_class_fake)
    # g_loss_class_fake_sum

    model.generator_loss_summary = tf.summary.scalar("generator_loss_summary", model.generator_loss)
    # g_loss_sum
    model.discriminator_loss_summary = tf.summary.scalar("discriminator_loss_summary", model.discriminator_loss)
    # d_loss_sum

    t_vars = tf.trainable_variables()
    model.discriminator_vars = [var for var in t_vars if 'd_' in var.name]
    model.generator_vars = [var for var in t_vars if 'g_' in var.name]
    # Creating checkpoint saver
    model.saver = tf.train.Saver()

    return model


def train(model, epoch, sess):
    # Creating Optimizer
    discriminator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.discriminator_loss,
                                                                                 var_list=model.discriminator_vars)
    generator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.generator_loss,
                                                                             var_list=model.generator_vars)

    #### tensorboard
    generator_optimizer_summary = tf.summary.merge(
        [model.random_noise_summary, model.discriminator_thief_summary, model.generator_summary,
         model.discriminator_thief_loss_summary, model.generator_loss_summary])

    discriminator_optimizer_summary = tf.summary.merge(
        [model.random_noise_summary, model.discriminator_police_summary,
         model.discriminator_police_loss_summary, model.discriminator_loss_summary,
         model.discriminator_police_class_loss_summary, model.generator_loss_class_fake_summary])
    writer = tf.summary.FileWriter("./logs", sess.graph)

    tf.global_variables_initializer().run()

    ## Creating sample -> test part
    sample_random_noise = np.random.normal(0, 1, [sample_size, random_noise_dim]).astype(np.float32)

    shuffle(data)
    sample_images_path = data[0: sample_size]
    sample_images_ = [get_image(sample_image_path,
                                input_height=input_size,
                                input_width=input_size,
                                resize_height=output_size,
                                resize_width=output_size,
                                crop=False) for sample_image_path in sample_images_path]
    sample_images = np.array(sample_images_).astype(np.float32)
    sample_labels = get_y(sample_images_path, label_dim, label_dict)  # get label(classification)

    # checkpoint variable
    counter = 1

    # checkpoint load
    checkpoint_dir_path = os.path.join(checkpoint_dir, checkpint_dir_model)
    could_load, checkpoint_counter = checkpoint_load(sess, model.saver, checkpoint_dir,
                                                     checkpint_dir_model)
    if could_load:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    # training

    for epoch in xrange(epoch):
        shuffle(data)

        batch_index = min(len(data), np.inf) // batch_size
        for index in xrange(0, batch_index):
            # Creating batch -> training part
            batch_images_path = data[index * batch_size: (index + 1) * batch_size]
            batch_images_ = [get_image(batch_image_path,
                                       input_height=input_size,
                                       input_width=input_size,
                                       resize_height=output_size,
                                       resize_width=output_size,
                                       crop=False) for batch_image_path in batch_images_path]
            batch_images = np.array(batch_images_).astype(np.float32)
            batch_labels = get_y(batch_images_path, label_dim, label_dict)  # get label(classification)
            batch_random_noise = np.random.normal(0, 1, [batch_size, random_noise_dim]).astype(np.float32)

            # Update
            # Update D network
            _, summary = sess.run([discriminator_optimizer, discriminator_optimizer_summary],
                                  feed_dict={real_image: batch_images,
                                             random_noise: batch_random_noise,
                                             y: batch_labels})
            writer.add_summary(summary, counter)

            # Update G network
            _, summary = sess.run([generator_optimizer, generator_optimizer_summary],
                                  feed_dict={random_noise: batch_random_noise})
            writer.add_summary(summary, counter)

            errD_fake = model.discriminator_thief_loss.eval(
                {random_noise: batch_random_noise, y: batch_labels})
            errD_real = model.discriminator_police_loss.eval({real_image: batch_images, y: batch_labels})
            ## change
            # errG = generator_loss.eval({random_noise: batch_random_noise })
            errG = model.generator_loss.eval({random_noise: batch_random_noise, y: batch_labels})

            ## Find cost value
            errD_class_real = model.discriminator_loss_class_real.eval(
                {real_image: batch_images, y: batch_labels})
            errG_class_fake = model.generator_loss_class_fake.eval(
                {real_image: batch_images, random_noise: batch_random_noise})
            accuracy = model.accuracy.eval({real_image: batch_images, y: batch_labels})

            # global value --> checkpoint value
            counter += 1
            print("Epoch: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f" % (
                epoch, index, batch_index, errD_fake + errD_real + errD_class_real, errG))
            print("Discriminator class acc: %.2f" % (accuracy))

            ## image save
            if np.mod(counter, 100) == 1:
                try:
                    samples = sess.run(sampler, feed_dict={random_noise: sample_random_noise,
                                                           real_image: sample_images,
                                                           y: sample_labels})
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format('samples', epoch, index))
                    print("[SAVE IMAGE]")
                except Exception as e:
                    print("image save error! ", e)

            ## checkpoint save
            if np.mod(counter, 500) == 1:
                print("[SAVE CHECKPOINT]")
                checkpoint_save(sess, model.saver, checkpoint_dir_path, counter)


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

with tf.Session(config=run_config) as sess:
    model = build_model()
    train(model, epoch, sess)
