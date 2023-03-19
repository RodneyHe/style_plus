import logging
import pickle

import numpy as np
import tensorflow as tf

from writer import Writer
from utils import general_utils as utils

import matplotlib.pyplot as plt
import PIL.Image as Image


def id_loss_func(y_gt, y_pred):
    return tf.reduce_mean(tf.keras.losses.MAE(y_gt, y_pred))

class Trainer(object):
    def __init__(self, args, model, data_loader):
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        self.model = model
        self.data_loader = data_loader

        # lrs & optimizers
        #lr = 5e-5 if self.args.resolution == 256 else 1e-5
        lr = 5e-6

        self.embedding_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.regulizing_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-8)
        #self.g_gan_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * lr)
        #self.style_d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.4 * lr)
        #self.im_d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.4 * lr)

        # Losses
        self.gan_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.pixel_loss_func = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
        self.id_loss_func = id_loss_func

        if args.pixel_mask_type == 'gaussian':
            sigma = int(80 * (self.args.resolution / 256))
            self.pixel_mask = utils.inverse_gaussian_image(self.args.resolution, sigma)
        else:
            self.pixel_mask = tf.ones([self.args.resolution, self.args.resolution])
            self.pixel_mask = self.pixel_mask / tf.reduce_sum(self.pixel_mask)

        self.pixel_mask = tf.broadcast_to(self.pixel_mask, [self.args.batch_size, *self.pixel_mask.shape])

        self.num_epoch = 1
        self.is_cross_epoch = False
        self.phase = self.model.phase

        # Lambdas
        if args.unified:
            self.lambda_gan = 0.5
        else:
            self.lambda_gan = 1

        self.lambda_pixel = 0.02

        self.lambda_id = 1
        self.lambda_attr_id = 1
        self.lambda_landmarks = 0.01
        self.r1_gamma = 10

        # Test
        self.test_not_imporved = 0
        self.max_id_preserve = 0
        self.min_lnd_dist = np.inf

        # Load the reference image to establish the coordinates
        # Using the image 00000.png No.31 landmark point as the reference point
        _ref_img = plt.imread("./dataset/dataset_256/image/00000/00000.png")
        _ref_img = tf.convert_to_tensor(_ref_img)[None, ...]
        _ref_landmark = self.model.G.landmarks(_ref_img)[0]
        
        self.ref_origin = _ref_landmark[30]

    def train(self):
        if self.phase == "regulizing":
            self.model.my_load()

        while self.num_epoch <= self.args.num_epochs:
            self.logger.info("-----------------------------------------------------")
            self.logger.info(f"Phase: {self.phase} Training epoch: {self.num_epoch}")

            if self.args.cross_frequency and (self.num_epoch % self.args.cross_frequency == 0):
                self.is_cross_epoch = True
                self.logger.info('This epoch is cross-face')
            else:
                self.is_cross_epoch = False
                self.logger.info('This epoch is same-face')

            try:
                if self.num_epoch % self.args.test_frequency == 0:
                    self.test()

                self.train_epoch()

            except Exception as e:
                self.logger.exception(e)
                raise

            # if self.test_not_imporved > self.args.not_improved_exit:
            #    self.logger.info(f'Test has not improved for {self.args.not_improved_exit} epochs. Exiting...')
            #    break

            if self.num_epoch == self.args.num_epochs:
                if self.phase == "embedding":
                    self.model.my_save()
                    break

                elif self.phase == "regulizing":
                    self.model.my_save()
                    break

            self.num_epoch += 1

    def train_epoch(self):
        
        id_loss = 0
        landmarks_loss = 0
        pixel_loss = 0

        if self.phase == "embedding":
            self.model.train()
        elif self.phase == "regulizing":
            # Frozen networks weights during regulizing phase, only train basis vectors.
            self.model.G.attr_encoder.trainable = False
            self.model.G.reference_encoder.trainable = False

        id_img, id_z_matching, attr_img, attr_img_indices = self.data_loader.get_batch(is_cross=self.is_cross_epoch)

        if not self.is_cross_epoch:
            attr_img = id_img

        # Forward that does not require grads
        # id_embedding = self.model.G.id_encoder(id_img)
        # src_landmarks = self.model.G.landmarks(attr_img)

        with tf.GradientTape(persistent=True) as g_tape:

            # attr_embedding = self.model.G.attr_encoder(attr_img)
            # self.logger.info(f"attr embedding stats- mean: {tf.reduce_mean(tf.abs(attr_embedding)):.5f},"
            #                  f" variance: {tf.math.reduce_variance(attr_embedding):.5f}")

            # feature_tag = tf.concat([id_embedding, attr_embedding], -1)

            gen_img, id_embedding, _, src_landmarks = self.model.G(id_img, id_z_matching, attr_img, self.ref_origin)

            # if self.phase == "embedding":
            #     gen_img, id_embedding, _, _ = self.model.G(id_img, id_z_matching, 
            #                                                attr_img, self.phase, self.ref_origin)
            # elif self.phase == "regulizing":
            #     gen_img, id_embedding, _, _ = self.model.G(id_img, id_z_matching, 
            #                                                attr_img, self.phase, self.ref_origin, self.orthogonal_basis)

            # if self.phase == "embedding":
            #     latent_embedding = self.model.G.reference_encoder(feature_tag)
            # elif self.phase == "regulizing":
            #     # Compute the coordinates of No.31 landmark point of samples relative to the reference origin
            #     sample_coords = src_landmarks[:, 30] - tf.broadcast_to(self.ref_origin, [self.args.batch_size, 2])
            #     # style plus latent space controlling
            #     #latent_embedding = tf.tensordot(sample_coords, self.orthogonal_basis, 1)[:,None,:] + self.bias[:,None,:]
            #     latent_embedding = tf.tensordot(sample_coords, self.orthogonal_basis, 1)[:,None,:]
            
            # style_control_vector = self.model.G.reference_decoder(latent_embedding)
            # gen_img, _, _ = self.model.G.stylegan_s(id_z_matching, style_control_vector[:,0,:])

            # # Move to roughly [0,1]
            # gen_img = (gen_img + 1) / 2
            # gen_img = tf.clip_by_value(gen_img, 0, 1)

            if self.args.id_loss:
                gen_img_id_embedding = self.model.G.id_encoder(gen_img)
                id_loss = self.lambda_id * id_loss_func(gen_img_id_embedding, tf.stop_gradient(id_embedding))
                self.logger.info(f'id loss is {id_loss:.3f}')
            
            if self.args.landmarks_loss:
                try:
                    dst_landmarks = self.model.G.landmarks(gen_img)
                except Exception as e:
                    self.logger.warning(f'Failed finding landmarks on prediction. Dont use landmarks loss. Error:{e}')
                    dst_landmarks = None

                if dst_landmarks is None or src_landmarks is None:
                    landmarks_loss = 0
                else:
                    landmarks_loss = self.lambda_landmarks * \
                                     tf.reduce_mean(tf.keras.losses.MSE(src_landmarks, dst_landmarks))
                    self.logger.info(f'landmarks loss is: {landmarks_loss:.3f}')

            if not self.is_cross_epoch and self.args.pixel_loss:
                l1_loss = self.pixel_loss_func(attr_img, gen_img, sample_weight=self.pixel_mask)
                self.logger.info(f'L1 pixel loss is {l1_loss:.3f}')

                if self.args.pixel_loss_type == 'mix':
                    mssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(attr_img, gen_img, 1.0))
                    self.logger.info(f'mssim loss is {l1_loss:.3f}')
                    pixel_loss = self.lambda_pixel * (0.84 * mssim + 0.16 * l1_loss)
                else:
                    pixel_loss = self.lambda_pixel * l1_loss

                self.logger.info(f'pixel loss is {pixel_loss:.3f}')

            total_g_not_gan_loss = id_loss \
                                   + landmarks_loss \
                                   + pixel_loss

            self.logger.info(f'total G (not gan) loss is {total_g_not_gan_loss:.3f}')

        Writer.add_scalar('loss/landmarks_loss', landmarks_loss, step=self.num_epoch)
        Writer.add_scalar('loss/total_g_not_gan_loss', total_g_not_gan_loss, step=self.num_epoch)
        Writer.add_scalar('loss/id_loss', id_loss, step=self.num_epoch)

        if not self.is_cross_epoch:
            Writer.add_scalar('loss/pixel_loss', pixel_loss, step=self.num_epoch)

        if self.args.debug or \
                (self.num_epoch < 1e3 and self.num_epoch % 1e2 == 0) or \
                (self.num_epoch < 1e4 and self.num_epoch % 1e3 == 0) or \
                (self.num_epoch % 1e4 == 0):
            utils.save_image(gen_img[0], self.args.images_results.joinpath(f'{self.phase}_{self.num_epoch}_prediction.png'))
            utils.save_image(id_img[0], self.args.images_results.joinpath(f'{self.num_epoch}_id_step.png'))
            utils.save_image(attr_img[0], self.args.images_results.joinpath(f'{self.num_epoch}_attr_step.png'))

            Writer.add_image('input/attr_image', tf.expand_dims(attr_img[0], 0), step=self.num_epoch)
            Writer.add_image('Prediction', tf.expand_dims(gen_img[0], 0), step=self.num_epoch)

        if total_g_not_gan_loss != 0:
            g_grads = g_tape.gradient(total_g_not_gan_loss, self.model.G.trainable_variables)

            g_grads_global_norm = tf.linalg.global_norm(g_grads)
            self.logger.info(f'global norm G not gan grad: {g_grads_global_norm}')

            if np.isnan(g_grads_global_norm):
                # self.model.my_save("")
                # np.save(str(self.args.weights_dir.joinpath("id_z_matching.npy")), id_z_matching.numpy())

                # with open(str(self.args.weights_dir.joinpath("step_grads")), "wb") as f:
                #     pickle.dump(g_grads, f)

                # with open(str(self.args.weights_dir.joinpath("step_noise")), "wb") as f:
                #     pickle.dump(noise_list, f)

                # for i in range(len(id_img)):
                #     utils.save_image(id_img[i], self.args.weights_dir.joinpath(f"step_id_img{i}.png"))
                #     utils.save_image(attr_img[i], self.args.weights_dir.joinpath(f"step_attr_img{i}.png"))
                
                nan_idx = tf.argmax(tf.reduce_mean(tf.keras.losses.MSE(src_landmarks, dst_landmarks), -1))
                self.logger.info(f"the gradient is exploded, cause the loss of input {nan_idx+1} is too large")

                self.data_loader.black_list.append(attr_img_indices[nan_idx])

                del g_tape
                return
            
            if self.phase == "embedding":
                self.embedding_optimizer.apply_gradients(zip(g_grads, self.model.G.trainable_variables))
            elif self.phase == "regulizing":
                self.regulizing_optimizer.apply_gradients(zip(g_grads, self.model.G.trainable_variables))
        del g_tape

    # Common

    # Test
    def test(self):
        self.logger.info(f"Phase: {self.phase}. Testing in epoch: {self.num_epoch}")
        self.model.test()

        similarities = {'id_to_pred': [], 'id_to_attr': [], 'attr_to_pred': []}

        fake_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}
        #real_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}

        if self.args.test_with_arcface:
            test_similarities = {'id_to_pred': [], 'id_to_attr': [], 'attr_to_pred': []}

        lnd_dist = []
        for i in range(self.args.test_size):
            id_img, id_z_matching, attr_img, _ = self.data_loader.get_batch(is_train=False, is_cross=True)

            gen_img, id_embedding, _, attr_lnds = self.model.G(id_img, id_z_matching, attr_img, self.ref_origin)

            gen_img_embedding = self.model.G.id_encoder(gen_img)
            attr_img_id_embedding = self.model.G.id_encoder(attr_img)

            similarities['id_to_pred'].extend(tf.keras.losses.cosine_similarity(id_embedding, gen_img_embedding).numpy())
            similarities['id_to_attr'].extend(tf.keras.losses.cosine_similarity(id_embedding, attr_img_id_embedding).numpy())
            similarities['attr_to_pred'].extend(tf.keras.losses.cosine_similarity(attr_img_id_embedding, gen_img_embedding).numpy())

            if self.args.test_with_arcface:
                try:
                    arc_id_embedding = self.model.G.test_id_encoder(id_img)
                    arc_pred_id = self.model.G.test_id_encoder(gen_img)
                    arc_attr_id = self.model.G.test_id_encoder(attr_img)

                    test_similarities['id_to_attr'].extend(
                        tf.keras.losses.cosine_similarity(arc_id_embedding, arc_attr_id).numpy())
                    test_similarities['id_to_pred'].extend(
                        tf.keras.losses.cosine_similarity(arc_id_embedding, arc_pred_id).numpy())
                    test_similarities['attr_to_pred'].extend(
                        tf.keras.losses.cosine_similarity(arc_attr_id, arc_pred_id).numpy())
                except Exception as e:
                    self.logger.warning(f'Not calculating test similarities for iteration: {i} because: {e}')

            # Landmarks
            dst_lnds = self.model.G.landmarks(gen_img)
            lnd_dist.extend(tf.reduce_mean(tf.keras.losses.MSE(attr_lnds, dst_lnds), axis=-1).numpy())

            # Fake Reconstruction
            self.test_reconstruction(id_img, id_z_matching, fake_reconstruction, display=(i==0), display_name='id_img')

            # if self.args.test_real_attr:
            #     # Real Reconstruction
            #     self.test_reconstruction(attr_img, attr_z_matching, real_reconstruction, display=(i==0), display_name='attr_img')

            if i == 0:
                utils.save_image(gen_img[0], self.args.images_results.joinpath(f'test_prediction_{self.num_epoch}.png'))
                utils.save_image(id_img[0], self.args.images_results.joinpath(f'test_id_{self.num_epoch}.png'))
                utils.save_image(attr_img[0], self.args.images_results.joinpath(f'test_attr_{self.num_epoch}.png'))

                Writer.add_image('test/prediction', gen_img, step=self.num_epoch)
                Writer.add_image('test input/id image', id_img, step=self.num_epoch)
                Writer.add_image('test input/attr image', attr_img, step=self.num_epoch)

                for j in range(np.minimum(3, attr_lnds.shape[0])):
                    src_xy = attr_lnds[j]  # GT
                    dst_xy = dst_lnds[j]  # pred

                    attr_marked = utils.mark_landmarks(attr_img[j], src_xy, color=(0, 0, 0))
                    pred_marked = utils.mark_landmarks(gen_img[j], src_xy, color=(0, 0, 0))
                    pred_marked = utils.mark_landmarks(pred_marked, dst_xy, color=(255, 112, 112))

                    Writer.add_image(f'landmarks/overlay-{j}', pred_marked, step=self.num_epoch)
                    Writer.add_image(f'landmarks/src-{j}', attr_marked, step=self.num_epoch)

        # Similarity
        self.logger.info('Similarities:')
        for k, v in similarities.items():
            self.logger.info(f'{k}: MEAN: {np.mean(v)}, STD: {np.std(v)}')

        mean_lnd_dist = np.mean(lnd_dist)
        self.logger.info(f'Mean landmarks L2: {mean_lnd_dist}')

        id_to_pred = np.mean(similarities['id_to_pred'])
        attr_to_pred = np.mean(similarities['attr_to_pred'])
        mean_disen = attr_to_pred - id_to_pred

        Writer.add_scalar('similarity/score', mean_disen, step=self.num_epoch)
        Writer.add_scalar('similarity/id_to_pred', id_to_pred, step=self.num_epoch)
        Writer.add_scalar('similarity/attr_to_pred', attr_to_pred, step=self.num_epoch)

        if self.args.test_with_arcface:
            arc_id_to_pred = np.mean(test_similarities['id_to_pred'])
            arc_attr_to_pred = np.mean(test_similarities['attr_to_pred'])
            arc_mean_disen = arc_attr_to_pred - arc_id_to_pred

            Writer.add_scalar('arc_similarity/score', arc_mean_disen, step=self.num_epoch)
            Writer.add_scalar('arc_similarity/id_to_pred', arc_id_to_pred, step=self.num_epoch)
            Writer.add_scalar('arc_similarity/attr_to_pred', arc_attr_to_pred, step=self.num_epoch)

        self.logger.info(f'Mean disentanglement score is {mean_disen}')

        Writer.add_scalar('landmarks/L2', np.mean(lnd_dist), step=self.num_epoch)

        # Reconstruction
        # if self.args.test_real_attr:
        #     Writer.add_scalar('reconstruction/real_MSE', np.mean(real_reconstruction['MSE']), step=self.num_epoch)
        #     Writer.add_scalar('reconstruction/real_PSNR', np.mean(real_reconstruction['PSNR']), step=self.num_epoch)
        #     Writer.add_scalar('reconstruction/real_ID', np.mean(real_reconstruction['ID']), step=self.num_epoch)

        Writer.add_scalar('reconstruction/fake_MSE', np.mean(fake_reconstruction['MSE']), step=self.num_epoch)
        Writer.add_scalar('reconstruction/fake_PSNR', np.mean(fake_reconstruction['PSNR']), step=self.num_epoch)
        Writer.add_scalar('reconstruction/fake_ID', np.mean(fake_reconstruction['ID']), step=self.num_epoch)

        # if mean_lnd_dist < self.min_lnd_dist:
        #     self.logger.info('Minimum landmarks dist achieved. saving checkpoint')
        #     self.test_not_imporved = 0
        #     self.min_lnd_dist = mean_lnd_dist
        #     self.model.my_save(f'_best_landmarks_epoch_{self.num_epoch}')

        # if np.abs(id_to_pred) > self.max_id_preserve:
        #     self.logger.info(f'Max ID preservation achieved! saving checkpoint')
        #     self.test_not_imporved = 0
        #     self.max_id_preserve = np.abs(id_to_pred)
        #     self.model.my_save(f'_best_id_epoch_{self.num_epoch}')

        # else:
        #     self.test_not_imporved += 1

    def test_reconstruction(self, img, z_matching, errors_dict, display=False, display_name=None):

        gen_img, id_embedding, _, _ = self.model.G(img, z_matching, img, self.ref_origin)

        recon_image = tf.clip_by_value(gen_img, 0, 1)
        recon_pred_id = self.model.G.id_encoder(recon_image)

        mse = tf.reduce_mean((img - recon_image) ** 2, axis=[1, 2, 3]).numpy()
        psnr = tf.image.psnr(img, recon_image, 1).numpy()

        errors_dict['MSE'].extend(mse)
        errors_dict['PSNR'].extend(psnr)
        errors_dict['ID'].extend(tf.keras.losses.cosine_similarity(id_embedding, recon_pred_id).numpy())

        if display:
            Writer.add_image(f'reconstruction/input_{display_name}_img', img, step=self.num_epoch)
            Writer.add_image(f'reconstruction/pred_{display_name}_img', gen_img, step=self.num_epoch)

    # Helpers
    # def generator_gan_loss(self, fake_logit):
    #     """
    #     G logistic non saturating loss, to be minimized
    #     """
    #     g_gan_loss = self.gan_loss_func(tf.ones_like(fake_logit), fake_logit)
    #     return self.lambda_gan * g_gan_loss

    # def discriminator_loss(self, fake_logit, real_logit):
    #     """
    #     D logistic loss, to be minimized
    #     verified as identical to StyleGAN' loss.D_logistic
    #     """
    #     fake_gt = tf.zeros_like(fake_logit)
    #     real_gt = tf.ones_like(real_logit)

    #     d_fake_loss = self.gan_loss_func(fake_gt, fake_logit)
    #     d_real_loss = self.gan_loss_func(real_gt, real_logit)

    #     d_loss = d_real_loss + d_fake_loss

    #     return self.lambda_gan * d_loss

    # def R1_gp(self, D, x):
    #     with tf.GradientTape() as t:
    #         t.watch(x)
    #         pred = D(x)
    #         pred_sum = tf.reduce_sum(pred)

    #     grad = t.gradient(pred_sum, x)

    #     # Reshape as a vector
    #     norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    #     gp = tf.reduce_mean(norm ** 2)
    #     gp = 0.5 * self.r1_gamma * gp

    #     return gp
