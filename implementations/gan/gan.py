


from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input,Dense,Conv2D,GlobalAveragePooling2D,BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import  LeakyReLU,ReLU,Dropout,Flatten
from tensorflow.keras.initializers import RandomNormal,Constant
from tensorflow.keras import  layers
from tensorflow.keras import metrics
from tensorflow.keras.activations import sigmoid
from tensorflow import  keras
import tensorflow as tf


tf.enable_eager_execution()

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = Dense(7*7*256)
        self.bn1  = BatchNormalization()
        self.bn2  = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.relu = ReLU()
        self.reshape = layers.Reshape((7 , 7, 256))
        self.conv2dT1 = Conv2DTranspose(128, 5, 1, padding='same', use_bias=False)
        self.conv2dT2 = Conv2DTranspose(64, 5, 2, padding='same', use_bias=False)
        self.conv2dT3 = Conv2DTranspose(1, 5, 2, padding='same', use_bias=False)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.conv2dT1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2dT2(x)
        x = self.bn3(x)
        x = self.conv2dT3(x)

        return x

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2D(64, 5, 2, padding='same')
        self.conv2 = Conv2D(128, 5, 2, padding='same')
        self.relu = ReLU()
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.flatten = Flatten()
        self.dense = Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)

        return  x


def generator_loss(fake_label):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_label),
                                                                    logits=fake_label))

    return g_loss

def discriminator_loss(real_label, fake_label):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_label),
                                                                         logits=real_label))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_label),
                                                                         logits=fake_label))
    d_loss = d_loss_fake+d_loss_real

    return  d_loss


def get_mnist():
    (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset

BUFFER_SIZE = 60000
BATCH_SIZE = 256
LATENT_DIM = 100
EPOCHS = 20

G_LOSS_METRIC = metrics.Mean('g_loss', dtype=tf.float32)
D_LOSS_METRIC = metrics.Mean('d_loss', dtype=tf.float32)


d_optimizer = keras.optimizers.Adam(1e-4)
g_optimizer = keras.optimizers.Adam(1e-4)

generator = Generator()
discriminator = Discriminator()

@tf.function
def train_one_step(images):
    noise = tf.random.normal([BATCH_SIZE,LATENT_DIM])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_imgs = generator(noise, training=True)
        real_pred = discriminator(images, training=True)
        fake_pred = discriminator(fake_imgs, training=True)

        g_loss = generator_loss(fake_imgs)
        d_loss = discriminator_loss(real_pred, fake_pred)

    g_grad = g_tape.gradient(g_loss, generator.trainable_variables)
    d_grad = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))

    G_LOSS_METRIC(g_loss)
    D_LOSS_METRIC(d_loss)


def train():
    dataste = get_mnist()
    for epoch in range(EPOCHS):
        for step, img_batch in enumerate(dataste):
            train_one_step(img_batch)

            if step%10 == 0:
                print('Epoch: {}\tstep: {}\tg_loss: {:.4f}\td_loss: {:.4f}'.format(epoch,
                                                                           step,
                                                                           G_LOSS_METRIC.result(),
                                                                           D_LOSS_METRIC.result()))


train()








