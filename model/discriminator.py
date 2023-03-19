import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.general_utils import get_weights


# Discriminate between my w's and StyleGAN's w's
class S_D(Model):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.linear1 = layers.Dense(8192, kernel_initializer=get_weights(), input_shape=(9984,))
        self.linear2 = layers.Dense(4096, kernel_initializer=get_weights())
        self.linear3 = layers.Dense(2048, kernel_initializer=get_weights())
        self.linear4 = layers.Dense(512, kernel_initializer=get_weights())
        self.linear5 = layers.Dense(1, kernel_initializer=get_weights())
        self.lrelu = layers.LeakyReLU(0.2)

        if self.args.load_checkpoint:
            self.build(input_shape=(1, 1, 9984))
            self.load_weights(str(self.args.load_checkpoint.joinpath(self.__class__.__name__ + '.h5')))

    @tf.function
    def call(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.lrelu(x)
        x = self.linear3(x)
        x = self.lrelu(x)
        x = self.linear4(x)
        x = self.lrelu(x)
        x = self.linear5(x)

        return x

    def my_save(self, reason=''):
        self.save_weights(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + '.h5')))
