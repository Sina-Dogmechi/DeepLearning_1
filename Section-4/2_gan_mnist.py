# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def define_generator(latent_dim= 100):

    n_nodes = 128 * 7 * 7

    model = models.Sequential([
        layers.Dense(n_nodes, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'),
        ])

    return model


def define_discriminator(in_shape=(28, 28, 1)):

    model = models.Sequential([
            layers.Conv2D(64, (3, 3), strides=(2, 2),padding='same', input_shape=in_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
    ])

    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def define_gan(g_model, d_model):

    d_model.trainable = False

    model = models.Sequential([
        g_model,
        d_model
    ])

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


def load_real_samples():

	(trainX, _), (_, _) = mnist.load_data()

	X = np.reshape(trainX, (len(trainX), 28, 28, 1))

	X = X.astype('float32')

	X = X / 255

	return X


def generate_real_samples(dataset, n_samples):

    ix = np.random.randint(0, dataset.shape[0], n_samples)

    X = dataset[ix]

    y = np.ones((n_samples, 1))

    return X, y

def generate_latent_points(latent_dim, n_samples):

    x_input = np.random.randn(latent_dim * n_samples)

    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):

    x_input = generate_latent_points(latent_dim, n_samples)

    X = g_model.predict(x_input)

    y = np.zeros((n_samples, 1))

    return X, y

def save_plot(examples, epoch, n=10):

	for i in range(n * n):

		plt.subplot(n, n, 1 + i)

		plt.axis('off')

		plt.imshow(examples[i, :, :, 0], cmap='gray_r')

	filename = 'generated_plot_e%03d.png' % (epoch+1)

	plt.savefig(filename)
    
	plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):

    X_real, y_real = generate_real_samples(dataset, n_samples)

    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'>Accuracy real: {acc_real*100}, fake: {acc_fake*100}')

    save_plot(x_fake, epoch)

    filename = f'generator_model_{epoch + 1}.h5'

    g_model.save(filename)

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):

    bat_per_epo = int(dataset.shape[0] / n_batch)

    half_batch = int(n_batch / 2)

    for i in range(n_epochs):

        for j in range(bat_per_epo):

            X_real, y_real = generate_real_samples(dataset, half_batch)

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            d_loss, _ = d_model.train_on_batch(X, y)

            X_gan = generate_latent_points(latent_dim, n_batch)

            y_gan = np.ones((n_batch, 1))

            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            print(f">{i+1}, {j+1}/{bat_per_epo}, d= {d_loss:.3f}, g={g_loss:.3f}")

        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


latent_dim = 100

d_model = define_discriminator()

g_model = define_generator(latent_dim)

gan_model = define_gan(g_model, d_model)

dataset = load_real_samples()

train(g_model, d_model, gan_model, dataset, latent_dim)