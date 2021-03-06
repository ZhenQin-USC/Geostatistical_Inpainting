import util

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Flatten, LeakyReLU
from tensorflow.keras.layers import Input, Reshape, Dense, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

import numpy as np
import os
from os.path import join

def RMSE(x, y):
    return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))


class WGan80:

    def __init__(self, M, z_dim=64, name=None):

        self.M = M

        self.mx = M.shape[1]
        self.my = M.shape[2]
        self.mz = M.shape[3]

        if name is None:
            name = "wgan{}x{}".format(self.mx, self.my)
        self.name = name

        if os.path.exists("images") is not True:
            os.mkdir("images")
        if os.path.exists(join("images",self.name)) is not True:
            os.mkdir(join("images",self.name))
        if os.path.exists("losses") is not True:
            os.mkdir("losses")
        if os.path.exists("models") is not True:
            os.mkdir("models")
        self.path_to_model = join("models", self.name)
        if os.path.exists(self.path_to_model) is not True:
            os.mkdir(self.path_to_model)

        self.z_dim = z_dim

        self.critic_iter = 5
        self.clip_value = 0.01

        self.generator = self.get_generator()
        self.critic = self.get_critic()
        self.wgan = self.get_wgan()

        self.noise = np.random.normal(0, 1, (25, self.z_dim))

    def get_generator(self):

        noise = Input(shape=(self.z_dim,))

        _ = Dense(64 * 5 * 5, input_dim=self.z_dim)(noise)
        _ = Reshape((5, 5, 64))(_)

        _ = Conv2D(64, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(32, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(16, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(8, (3, 3), padding='same')(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        generated_image = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(_)

        return Model(noise, generated_image)

    def get_critic(self):

        input_image = Input(shape=(self.mx, self.my, self.mz))

        _ = Conv2D(8, (3, 3), padding='same')(input_image)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(16, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(32, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(64, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Flatten()(_)

        score = Dense(1)(_)

        return Model(input_image, score)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def get_wgan(self):

        self.critic.compile(optimizer=RMSprop(lr=5e-5), loss=self.wasserstein_loss, metrics=['accuracy'])
        self.critic.trainable = False  # check

        noise = Input(shape=(self.z_dim,))
        img = self.generator(noise)
        score = self.critic(img)

        wgan = Model(noise, score)
        wgan.compile(optimizer=RMSprop(lr=5e-5), loss=self.wasserstein_loss)
        wgan.summary()
        plot_model(wgan, to_file='wgan.png')

        return wgan

    def save_generated_images(self, i):

        images = self.generator.predict(self.noise)
        util.plot_tile(images, "images/" + self.name + "/" + str(i))

    def train_wgan(self, totalEpoch=300, batch_size=128, load=False, checkpoint=50):

        if not load:

            d_loss = 0
            d_losses = np.zeros([totalEpoch, 2])
            g_losses = np.zeros([totalEpoch, 1])

            real_label = (-1) * np.ones((batch_size, 1))
            fake_label = np.ones((batch_size, 1))

            for i in range(totalEpoch):
                for j in range(self.critic_iter):

                    real_images = self.M[np.random.randint(0, self.M.shape[0], batch_size)]
                    noise = np.random.normal(0, 1, [batch_size, self.z_dim])

                    fake_images = self.generator.predict(noise)

                    d_loss_real = self.critic.train_on_batch(real_images, real_label)
                    d_loss_fake = self.critic.train_on_batch(fake_images, fake_label)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    for layer in self.critic.layers:
                        weights = layer.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        layer.set_weights(weights)

                d_losses[i, :] = d_loss

                g_loss = self.wgan.train_on_batch(noise, real_label)
                g_losses[i, :] = g_loss

                print("%d [D loss: %f] [G loss: %f]" % (i, d_loss[0], g_loss))
                util.plotAllLosses(d_losses, g_losses, name="wgan_losses")

                if i % checkpoint == 0:
                    self.save_generated_images(i)
                    np.save("losses/" + self.name + "_d_losses.npy", np.array(d_losses))
                    np.save("losses/" + self.name + "_g_losses.npy", np.array(g_losses))

            self.wgan.save(join(self.path_to_model, "wgan.h5"))
            self.generator.save(join(self.path_to_model, "wgan_generator.h5"))
            self.critic.save(join(self.path_to_model, "wgan_critic.h5"))
        else:
            print("Trained model loaded")
            self.wgan = load_model('wgan.h5')


class WGan28:

    def __init__(self, M, z_dim=64, name=None):

        self.M = M
        # self.D = D

        self.mx = M.shape[1]
        self.my = M.shape[2]
        self.mz = M.shape[3]

        if name is None:
            name = "wgan{}x{}".format(self.mx, self.my)
        self.name = name

        if os.path.exists("images") is not True:
            os.mkdir("images")
        if os.path.exists(join("images",self.name)) is not True:
            os.mkdir(join("images",self.name))
        if os.path.exists("losses") is not True:
            os.mkdir("losses")
        if os.path.exists("models") is not True:
            os.mkdir("models")
        self.path_to_model = join("models", self.name)
        if os.path.exists(self.path_to_model) is not True:
            os.mkdir(self.path_to_model)

        self.z_dim = z_dim

        self.critic_iter = 5
        self.clip_value = 0.01

        self.generator = self.get_generator()
        self.critic = self.get_critic()
        self.wgan = self.get_wgan()

        self.noise = np.random.normal(0, 1, (25, self.z_dim))

    def get_generator(self):

        noise = Input(shape=(self.z_dim,))

        _ = Dense(64 * 4 * 4, input_dim=self.z_dim)(noise)
        _ = Reshape((4, 4, 64))(_)

        _ = Conv2D(64, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(32, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(16, (3, 3))(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        generated_image = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(_)

        return Model(noise, generated_image)

    def get_critic(self):

        input_image = Input(shape=(self.mx, self.my, self.mz))

        _ = Conv2D(16, (3, 3), padding='same')(input_image)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(32, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(64, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Flatten()(_)

        score = Dense(1)(_)

        return Model(input_image, score)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def get_wgan(self):

        self.critic.compile(optimizer=RMSprop(lr=5e-5), loss=self.wasserstein_loss, metrics=['accuracy'])
        self.critic.trainable = False  # check

        noise = Input(shape=(self.z_dim,))
        img = self.generator(noise)
        score = self.critic(img)

        wgan = Model(noise, score)
        wgan.compile(optimizer=RMSprop(lr=5e-5), loss=self.wasserstein_loss)
        wgan.summary()
        plot_model(wgan, to_file='wgan.png')

        return wgan

    def save_generated_images(self, i):

        images = self.generator.predict(self.noise)
        util.plot_tile(images, "images/" + self.name + "/" + str(i))

    def train_wgan(self, totalEpoch=300, batch_size=128, load=False, checkpoint=50):

        if not load:

            d_loss = 0
            d_losses = np.zeros([totalEpoch, 2])
            g_losses = np.zeros([totalEpoch, 1])

            real_label = (-1) * np.ones((batch_size, 1))
            fake_label = np.ones((batch_size, 1))

            for i in range(totalEpoch):
                for j in range(self.critic_iter):

                    real_images = self.M[np.random.randint(0, self.M.shape[0], batch_size)]
                    noise = np.random.normal(0, 1, [batch_size, self.z_dim])

                    fake_images = self.generator.predict(noise)

                    d_loss_real = self.critic.train_on_batch(real_images, real_label)
                    d_loss_fake = self.critic.train_on_batch(fake_images, fake_label)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    for layer in self.critic.layers:
                        weights = layer.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        layer.set_weights(weights)

                d_losses[i, :] = d_loss

                g_loss = self.wgan.train_on_batch(noise, real_label)
                g_losses[i, :] = g_loss

                print("%d [D loss: %f] [G loss: %f]" % (i, d_loss[0], g_loss))
                util.plotAllLosses(d_losses, g_losses, name="wgan_losses")

                if i % checkpoint == 0:
                    self.save_generated_images(i)
                    np.save("losses/" + self.name + "_d_losses.npy", np.array(d_losses))
                    np.save("losses/" + self.name + "_g_losses.npy", np.array(g_losses))

            self.wgan.save(join(self.path_to_model, "wgan.h5"))
            self.generator.save(join(self.path_to_model, "wgan_generator.h5"))
            self.critic.save(join(self.path_to_model, "wgan_critic.h5"))

        else:
            print("Trained model loaded")
            self.wgan = load_model('wgan.h5')
