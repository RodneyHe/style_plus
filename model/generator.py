import logging

import tensorflow as tf
from tensorflow.keras import layers, Model

from model import id_encoder
from model import attr_encoder
from model import reference_mapping, reference_autoencoder
from model import landmarks
from model.arcface.inference import MyArcFace

class G(Model):
    def __init__(self, args, id_model_path, image_G,
                 landmarks_net_path, face_detection_model_path, test_id_model_path, phase):

        super().__init__()
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        self.id_encoder = id_encoder.IDEncoder(args, id_model_path)
        self.id_encoder.trainable = False

        self.attr_encoder = attr_encoder.AttrEncoder(args)

        #self.reference_mapping = reference_mapping.ReferenceMappingNetwork(args)
        self.reference_encoder = reference_autoencoder.ReferenceEncoder(args)
        self.reference_decoder = reference_autoencoder.ReferenceDecoder(args)

        self.stylegan_s = image_G
        self.stylegan_s.trainable = False

        if phase == "embedding":
            self.phase = "embedding"
        elif phase == "regulizing":
            self.phase = "regulizing"
            orthogonal_initializer = tf.keras.initializers.Orthogonal()
            self.orthogonal_basis = tf.Variable(orthogonal_initializer(shape=(2, 2)), trainable=True, name="weights")
            # constant_initializer = tf.keras.initializers.Constant(0.1)
            # self.bias = tf.Variable(constant_initializer(shape=(self.args.batch_size, 2)), trainable=True, name="bias")

        if args.train:
            self.test_id_encoder = MyArcFace(test_id_model_path)
            self.test_id_encoder.trainable = False

            self.landmarks = landmarks.LandmarksDetector(args, landmarks_net_path, face_detection_model_path)
            self.landmarks.trainable = False

    def call(self, id_img_input, id_z_input, attr_img_input, reference_origin):
        id_embedding = self.id_encoder(id_img_input)
        attr_embedding = self.attr_encoder(attr_img_input)

        attr_lnds = self.landmarks(attr_img_input)
        
        feature_tag = tf.concat([id_embedding, attr_embedding], -1)

        if self.phase == "embedding":
            latent_embedding = self.reference_encoder(feature_tag)
        elif self.phase == "regulizing":
            sample_coords = attr_lnds[:, 30] - tf.broadcast_to(reference_origin, [self.args.batch_size, 2])
            latent_embedding = tf.tensordot(sample_coords, self.orthogonal_basis, 1)[:,None,:]

        style_control_vector = self.reference_decoder(latent_embedding)
        gen_img, _, _ = self.stylegan_s(id_z_input, style_control_vector[:, 0, :])

        # Move to roughly [0,1]
        gen_img = (gen_img + 1) / 2
        gen_img = tf.clip_by_value(gen_img, 0, 1)

        return gen_img, id_embedding, attr_embedding, attr_lnds
    
    def get_config(self):
        return {"id_encoder": self.id_encoder, 
                "attr_encoder": self.attr_encoder, 
                "test_id_encoder": self.test_id_encoder, 
                "landmarks": self.landmarks}

    def my_save(self, reason=""):
        self.attr_encoder.my_save(reason)
        self.reference_encoder.my_save(reason)
        self.reference_decoder.my_save(reason)
    
    def my_load(self, reason=""):
        self.attr_encoder.my_load(reason)
        self.reference_encoder.my_load(reason)
        self.reference_decoder.my_load(reason)
