from __future__ import print_function, division

# from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sagemaker_tensorflow import PipeModeDataset
import tensorflow as tf

import sys
import numpy as np
import os
import argparse
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.system('pip install -r requirements.txt')
import matplotlib.pyplot as plt

# check input data
os.system('ls /opt/ml/input/data/training |wc')

def parse_args():
    
    parser = argparse.ArgumentParser()
                                
    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.000000005)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--tmax', type=int, default=1)
    parser.add_argument('--fmin', type=int, default=0)

    # log and outfile control
    parser.add_argument('--d_skip_interval', type=int, default=1)
    parser.add_argument('--g_skip_interval', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--load_yn', dest='load_yn', default=False, action='store_true')

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
#     parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--gen_image_path', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def build_generator(latent_dim, channels):
    print("build generator --- ")
    model = Sequential()

    # In: 100
    # Out: dim x dim x depth
    model.add(Dense(256 * 32 * 32, activation="relu", input_dim=latent_dim))
    model.add(Reshape((32, 32, 256)))
    
    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    #
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    #
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_discriminator(img_shape):
    print("build discriminator --- ")
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def get_dcgan(img_rows=256, img_cols=256, channels=1, latent_dim=100, lr=0.000001, beta_1=0.5, load_yn=True, epoch=0):

    img_shape = (img_rows, img_cols, channels)
    optimizer = Adam(lr, beta_1)

    # Build and compile the discriminator
    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    # Build the generator
    generator = build_generator(latent_dim, channels)

    # The generator takes noise as input and generates imgs
    z = Input(shape=(latent_dim,))
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(img)

    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # load weight
    if load_yn :
        os.system('tar -vxf /opt/ml/input/data/model/model.tar.gz --directory /opt/ml/input/data/model')
        discriminator.load_weights('/opt/ml/input/data/model/d-{epoch:04d}.ckpt'.format(epoch=epoch))
        generator.load_weights('/opt/ml/input/data/model/g-{epoch:04d}.ckpt'.format(epoch=epoch))
    
    return generator, discriminator, combined

# 이미지리스트로부터 numpy array 리턴하는 함수
def feed_imgs(img_path, img_list):
    imgs = []
    for f in img_list:
        imgs.append(np.expand_dims(plt.imread(img_path + '/' + f), axis=3))
#         imgs.append(plt.imread(img_path + '/' + f))
    return np.stack(imgs, axis=0) * 2 -1

def save_imgs(gen_image_path, epoch, latent_dim, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("/opt/ml/output/data/gen_%d.png" % epoch)
    
    plt.close()


if __name__ == "__main__":
            
    args, _ = parse_args()
    
    device = '/gpu:0' 
    print(device)
    
    # job paramters
    print("== job parameters ======================================")
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    beta_1=args.beta_1
    tmax=args.tmax
    fmin=args.fmin
    print('batch_size = {}, epochs = {}, learning rate = {}, beta 1 = {}, tmax={}, fmin={}'.format(batch_size, epochs, lr, beta_1, tmax, fmin))
    
    # io parameters
    print("== io parameters  ======================================")
    img_path=args.train
    latent_dim=args.latent_dim
    gen_image_path=args.gen_image_path
    print('img_path = {}, latent_dim = {}, gen_image_path={}'.format(img_path, latent_dim, gen_image_path))

    # log parameters
    print("== etc parameters ======================================")
    d_skip_interval=args.d_skip_interval
    g_skip_interval=args.g_skip_interval
    log_interval=args.log_interval
    save_interval=args.save_interval
    load_yn=args.load_yn
    model_dir=args.model_dir
    print('d_skip_interval = {}, g_skip_interval = {}, log_interval = {}, load_yn={}, model_dir={}'.format(d_skip_interval, g_skip_interval, log_interval, load_yn, model_dir))
    print("========================================================")

    
    d_losses =[]
    g_losses =[]
#     d_loss = 0
#     g_loss = 0
    aimgs = os.listdir(img_path)
    n_img = len(aimgs)
    
    generator, discriminator, combined = get_dcgan(lr=lr, beta_1=beta_1, latent_dim=latent_dim, load_yn=load_yn)
    strategy = tf.distribute.MirroredStrategy()

    for epoch in range(epochs):
    # for epoch in range(3):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random half of images
        idx = np.random.randint(0, n_img, batch_size)
        img_list = [aimgs[i] for i in idx]
        imgs = feed_imgs(img_path, img_list)

        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
#         with tf.device(device):
        with strategy.scope():
            gen_imgs = generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            if epoch % d_skip_interval == 0:
                d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1))*tmax)
                d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1))+fmin)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (wants discriminator to mistake images as real)
            if epoch % g_skip_interval ==0:
                g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
        # Plot the progress
        if epoch % log_interval == 0:
            print ("%5d [D loss: %f, acc.: %.2f%%] [G loss: %f] [d+g: %f] [d+2g: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, (d_loss[0]+g_loss)/2, (d_loss[0]+2*g_loss)/3))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(gen_image_path, epoch, latent_dim, generator)
    
    print ("%5d [D loss: %f, acc.: %.2f%%] [G loss: %f] [d+g: %f] [d+2g: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, (d_loss[0]+g_loss)/2, (d_loss[0]+2*g_loss)/3))
    save_imgs(gen_image_path, epoch, latent_dim, generator)
    discriminator.save_weights('/opt/ml/model/d-{epoch:04d}.ckpt'.format(epoch=0))
    generator.save_weights('/opt/ml/model/g-{epoch:04d}.ckpt'.format(epoch=0))
    
    with open('/opt/ml/output/data/g_losses.txt', 'w') as f:
        for item in g_losses:
            f.write("%s\n" % item)
        
#     os.system('ls -al /opt/ml/model/')
#     os.system('ls -al /opt/ml/output/data/')
#     return d_losses, g_losses