import time
import sys

sys.path.append('..')

from utils import general_utils as utils
from model import id_encoder, reference_mapping, reference_autoencoder, attr_encoder,\
    generator, discriminator, landmarks

from model.stylegan import StyleGAN_G, StyleGAN_D

import tensorflow as tf
from tensorflow.keras import layers, Model


class Network(Model):
    def __init__(self, args, id_net_path, base_generator, phase,
                 landmarks_net_path=None, face_detection_model_path=None, test_id_net_path=None):
        super().__init__()
        self.args = args
        self.phase = phase
        self.G = generator.G(args, id_net_path, base_generator,
                             landmarks_net_path, face_detection_model_path, test_id_net_path, phase)

    def call(self):
        raise NotImplemented()

    def my_save(self, reason=""):
        if self.phase == "embedding":
            self.G.my_save(reason)
        elif self.phase == "regulizing":
            tf.saved_model.save(self.G, str(self.args.weights_dir.joinpath(self.__class__.__name__ + f"{reason}")))

    def my_load(self, reason=""):
        self.G.my_load(reason)
    
    def train(self):
        self._set_trainable_behavior(True)

    def test(self):
        self._set_trainable_behavior(False)

    def _set_trainable_behavior(self, trainable):
        self.G.attr_encoder.trainable = trainable
        self.G.reference_encoder.trainable = trainable
        self.G.reference_decoder.trainable = trainable
