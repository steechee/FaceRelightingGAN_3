from __future__ import print_function

import os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image
# from utils import templight
from utils import getmatrix
from utils import getshading10
from utils import getshadingnp10
from utils import getshading
from utils import getshadingnp
from utils import smoothnessloss
from utils import bwsloss
from utils import swloss
import datetime

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Trainer(object):
    def __init__(self, config, rgb_loader, normal_loader, mask_loader, light_loader):
    # def __init__(self, config, rgb_loader, normal_loader, mask_loader):
        self.config = config
        self.data_loader = rgb_loader
        self.normal_loader = normal_loader
        self.mask_loader = mask_loader
        self.light_loader = light_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            # self.build_test_model()

    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

        x_fixed, normal_fixed, mask_fixed, light_fixed = self.get_image_from_loader() # 16 64 64 3

        shading_fixed = np.transpose(getshadingnp(np.transpose((normal_fixed/127.5 -1),[0, 3, 1, 2]), light_fixed),[0, 2, 3, 1])
        albedo_fixed = np.clip((x_fixed/127.5 -1)/(shading_fixed + 1e-3), -10, 10)

        shading_fixed = np.clip(((shading_fixed+1)*127.5), 0, 255)
        albedo_fixed = np.clip(((albedo_fixed+1)*127.5), 0, 255)

        save_image(x_fixed, '{}/x_fixed_rgb.png'.format(self.model_dir))
        save_image(normal_fixed, '{}/x_fixed_normal.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/x_fixed_mask.png'.format(self.model_dir))
        save_image(shading_fixed, '{}/x_fixed_shading.png'.format(self.model_dir))
        save_image(albedo_fixed, '{}/x_fixed_albedo.png'.format(self.model_dir))

        save_image(x_fixed*mask_fixed/255., '{}/x_fixed_rgb*mask.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
                "measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "k_t": self.k_t,
                    "balance": self.balance,
                    "d_loss": self.d_loss,
                    "d_loss_real": self.d_loss_real,
                    "d_loss_fake": self.d_loss_fake,
                    "g_loss": self.g_loss,
                    "renderingloss": self.renderingloss,
                    "generatorloss": self.generatorloss,
                    "normalloss": self.normalloss,
                    "maskloss": self.maskloss,
                    "albedoloss": self.albedoloss,
                    # "albedosmoothloss": self.albedosmoothloss,
                    "shadingsmoothloss": self.shadingsmoothloss,
                    "shadingbwsloss": self.shadingbwsloss,
                    # "normalsmoothloss": self.normalsmoothloss,
                    "unitnormloss": self.unitnormloss,
                    "lightloss": self.lightloss,
                    "weightloss": self.weightloss,
                    # "shadingloss": self.shadingloss,
                    "reconloss": self.reconloss,
                    # "outloss": self.outloss
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0: # every 50 steps
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                k_t = result['k_t']
                balance = result['balance']
                d_loss = result['d_loss']
                d_loss_real = result['d_loss_real']
                d_loss_fake = result['d_loss_fake']
                g_loss = result['g_loss']
                renderingloss = result['renderingloss']
                generatorloss = result['generatorloss']
                normalloss = result['normalloss']
                albedoloss = result['albedoloss']
                unitnormloss = result['unitnormloss']
                lightloss = result['lightloss']
                weightloss = result['weightloss']
                # shadingloss = result['shadingloss']
                reconloss = result['reconloss']
                shadingsmoothloss = result['shadingsmoothloss']
                shadingbwsloss = result['shadingbwsloss']
                # albedosmoothloss = result['albedosmoothloss']
                # normalsmoothloss = result['normalsmoothloss']
                maskloss = result['maskloss']
                # outloss = result['outloss']


                print("[{}/{}] k_t: {:.4f}, d_loss_real: {:.4f}, d_loss_fake: {:.4f}, Loss_G: {:.4f}, albedo: {:.4f}, normal: {:.4f}, unitnorm: {:.4f}, light: {:.4f}, weight: {:.4f}, shadingsmooth: {:.4f}, redering: {:.4f}, mask: {:.4f}, recon: {:.4f}". \
                      format(step, self.max_step, k_t, d_loss_real, d_loss_fake, g_loss, albedoloss, normalloss, unitnormloss, lightloss, weightloss, shadingsmoothloss, renderingloss, maskloss, reconloss))
                # print("[{}/{}] measure: {:.4f}, k_t: {:.4f}, balance: {:.4f}, Loss_D: {:.6f}, d_loss_real: {:.4f}, d_loss_fake: {:.4f}, Loss_G: {:.6f}, generator: {:.4f}, redering: {:.4f}, albedo: {:.4f}, normal: {:.4f}, unitnorm: {:.4f}, light: {:.4f},  weight: {:.4f}, shadingsmooth: {:.4f}, shadingbws: {:.4f}, mask: {:.4f}, recon: {:.4f}". \
                 # print("[{}/{}] measure: {:.3f}, k_t: {:.3f}, balance: {:.3f}, Loss_D: {:.3f}, d_loss_real: {:.3f}, d_loss_fake: {:.3f}, Loss_G: {:.3f}, generator: {:.3f}, redering: {:.3f}, albedo: {:.3f}, normal: {:.3f}, unitnorm: {:.3f}, light: {:.3f},  weight: {:.3f}, shadingsmooth: {:.3f}, shadingbws: {:.3f}, mask: {:.3f}, recon: {:.3f}". \
                # print("[{}/{}] measure: {:.4f}, k_t: {:.4f}, balance: {:.4f}, Loss_D: {:.6f}, d_loss_real: {:.4f}, d_loss_fake: {:.4f}, Loss_G: {:.6f}, redering: {:.4f},  generator: {:.4f}, albedo: {:.4f}, normal: {:.4f}, unitnorm: {:.4f}, light: {:.4f}, shading: {:.4f}, recon: {:.4f}". \
                #     format(step, self.max_step, measure, k_t, balance, d_loss, d_loss_real, d_loss_fake, g_loss, renderingloss, generatorloss, albedoloss, normalloss, unitnormloss, lightloss, shadingloss, reconloss))

            if step % (self.log_step * 10) == 0: # every 500 steps
            # if step % (self.log_step) == 0: #
                self.generate(x_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                #cur_measure = np.mean(measure_history)
                #if cur_measure > prev_measure * 0.99:
                #prev_measure = cur_measure

    def build_model(self):
        ## load input data
        self.x = self.data_loader #rgb #16 3 64 64
        # print (self.x.get_shape())
        self.normalgt = self.normal_loader #16 3 64 64
        # print (self.normalgt.get_shape())
        self.maskgt = self.mask_loader #16 3 64 64
        # print (self.maskgt.get_shape())
        self.lightgt = self.light_loader #16 27
        # print (self.lightgt.get_shape())

        ## normalize data range from [0 255] to [-1, 1] 16 3 64 64
        x = norm_img(self.x) #16 3 64 64
        # print (x.get_shape())
        normalgt = norm_img(self.normalgt) #16 3 64 64
        # print (normalgt.get_shape())
        maskgt = norm_img(self.maskgt) #16 3 64 64
        # print (maskgt.get_shape())


        ## get ground truth shading and albedo to calculate loss
        shadinggt = getshading(normalgt, self.lightgt)
        # print (shadinggt.get_shape()) #16 3 64 64
        albedogt = tf.clip_by_value(x/(shadinggt + 1e-3), -10, 10)
        # print (albedogt.get_shape()) #16 3 64 64

        ## define k_t
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        ## encoder, resblock and decoder consist generatorCNN
        albedo, normal, mask, self.light, self.shadingweight, shading, recon, pointrecon, self.newlight, newshading, newrecon, self.G_var = GeneratorCNN(
                self.x, normalgt, albedogt, self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format, reuse=False)
        # print (albedo.get_shape()) #16 3 64 64
        # print (normal.get_shape()) #16 3 64 64
        # print (mask.get_shape()) #16 3 64 64
        # print (light.get_shape()) #16 27 --> 16 30
        # print (shading.get_shape()) #16 3 64 64
        # print (recon.get_shape()) #16 3 64 64
        # print (light2.get_shape()) #16 27
        # print (shading2.get_shape()) #16 3 64 64
        # print (recon2.get_shape()) #16 3 64 64


        ## denormalize data range from [-1 1] to [0 255] to visualization 16 64 64 3
        self.albedo = denorm_img(albedo, self.data_format)
        self.albedogt = denorm_img(albedogt, self.data_format)
        self.normal = denorm_img(normal, self.data_format)
        self.mask = denorm_img(mask, self.data_format)
        self.shading = denorm_img(shading, self.data_format)
        self.recon = denorm_img(recon, self.data_format)
        self.pointrecon = denorm_img(pointrecon, self.data_format)
        self.newshading = denorm_img(newshading, self.data_format)
        self.newrecon = denorm_img(newrecon, self.data_format)

        # print (self.albedo.get_shape()) #16 64 64 3
        # print (self.albedogt.get_shape()) #16 64 64 3
        # print (self.normal.get_shape()) #16 64 64 3
        # print (self.mask.get_shape()) #16 64 64 3
        # print (self.shading.get_shape()) #16 64 64 3
        # print (self.recon.get_shape()) #16 64 64 3
        # print (self.shading2.get_shape()) #16 64 64 3
        # print (self.recon2.get_shape()) #16 64 64 3


        # self.out = self.mask/255.*self.recon + (1-(self.mask/255.))*tf.transpose(self.x,[0, 2, 3, 1])
        # self.out2 = self.mask/255.*self.recon2 + (1-(self.mask/255.))*tf.transpose(self.x,[0, 2, 3, 1])
        # out = tf.transpose(norm_img(self.out), [0, 3, 1, 2])
        # out2 = tf.transpose(norm_img(self.out2), [0, 3, 1, 2])
        # print (out.get_shape()) # 16 3 64 64
        # print (out2.get_shape()) # 16 3 64 64

        ## rendering loss
        # pointlight =     # 12 3
        # newshading = getpointshading(normal, pointlight) #
        # AE_newshading = DiscriminatorCNN(newshading, )



        d_out, self.D_z, self.D_var = DiscriminatorCNN(
        tf.concat([x, recon, pointrecon], 0), self.channel, self.z_num, self.repeat_num,
        self.conv_hidden_num, self.data_format)
        # AE_x, AE_recon = tf.split(d_out, 2)
        # print (d_out.get_shape()) # 16+16+16*12 224 3 64 64
        # AE_x, AE_recon = tf.split(d_out, 2)
        # AE_x, AE_recon, AE_pointrecon1, = tf.split(d_out, 2)
        AE_x = d_out[:16]
        AE_recon = d_out[16:32]
        AE_pointrecon = d_out[32:]


        # print (AE_x.get_shape()) #16 3 64 64
        # print (AE_recon.get_shape()) #16 3 64 64
        # print (AE_pointrecon.get_shape()) #192 3 64 64


        self.AE_x, self.AE_recon, self.AE_pointrecon = denorm_img(AE_x, self.data_format), denorm_img(AE_recon, self.data_format), denorm_img(AE_pointrecon, self.data_format)

        # print (self.AE_x.get_shape()) #16 64 64 3
        # print (self.AE_recon.get_shape()) #16 64 64 3
        # print (self.AE_pointrecon.get_shape()) #192 64 64 3

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)



        ## define losses
        # d_loss
        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x)) # 16 3 64 64
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_recon - recon)) # 16 3 64 64
        # self.d_loss_fake = 0.5*(tf.reduce_mean(tf.abs(AE_recon - recon)) + tf.reduce_mean(tf.abs(AE_recon2 - recon2)))
        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake

        # g_loss
        self.generatorloss = tf.reduce_mean(tf.abs(AE_recon - recon)) # 16 3 64 64
        # self.renderingloss = tf.reduce_mean(tf.abs(AE_recon2 - recon2))

        # albedo #
        # self.albedoloss = 0.2*tf.reduce_mean(tf.abs(albedo*tf.transpose(self.mask/255.,[0,3,1,2]) - albedogt*tf.transpose(self.mask/255.,[0,3,1,2]))) # 16 3 64 64
        self.albedoloss = tf.reduce_mean(tf.abs(albedo*tf.transpose(self.mask/255.,[0,3,1,2]) - albedogt*tf.transpose(self.mask/255.,[0,3,1,2])))
        # self.albedosmoothloss = smoothnessloss(albedo*tf.transpose(self.mask/255.,[0,3,1,2])) # albedo or self.albedo?

        # normal: assume both gt and estimated normal are normalized to unit norm
        self.normalloss = tf.losses.cosine_distance(normal*tf.transpose(self.mask/255.,[0,3,1,2]), normalgt*tf.transpose(self.mask/255.,[0,3,1,2]), dim=1) # 16 3 64 64
        # self.normalloss = tf.reduce_mean(tf.abs(normal*tf.transpose(self.mask/255.,[0,3,1,2]) - normalgt*tf.transpose(self.mask/255.,[0,3,1,2]))) # 16 3 64 64
        # self.normalsmoothloss = 0.004*smoothnessloss(self.normal) # albedo or self.albedo?
        self.gt_Nnm = tf.ones([16,64,64])
        self.unitnormloss = tf.reduce_mean(tf.abs(tf.norm(normal,axis=1) - self.gt_Nnm)) # 16 3 64 64

        # light

        # light9 = tf.concat([self.light[:,:9], self.light[:,10:19], self.light[:,20:29]], 1)
        # print (light[:,:9].get_shape())
        # print (light9.get_shape())
        # self.lightloss = tf.reduce_mean(tf.abs(light9 - self.lightgt)) # 16 27
        self.lightloss = 2*tf.reduce_mean(tf.abs(self.light - self.lightgt)) # 16 27
        # self.swloss = shadingweightloss(self.shadingweight)
        # self.weightloss = tf.reduce_mean(tf.abs(self.shadingweight - 1.5))
        self.weightloss = tf.reduce_mean(swloss(self.shadingweight))

        # shading
        # self.shadingloss = tf.reduce_mean(tf.abs(shading - shadinggt))
        self.shadingsmoothloss = smoothnessloss(shading*tf.transpose(self.mask/255.,[0,3,1,2])) # 16 3 64 64 # shading or self.shading
        self.shadingbwsloss, self.avg_r, self.avg_g, self.avg_b = bwsloss(shading,tf.transpose(self.mask/255.,[0,3,1,2]))

        # recon
        self.reconloss = 5*tf.reduce_mean(tf.abs(recon*tf.transpose(self.mask/255.,[0,3,1,2]) - x*tf.transpose(self.mask/255.,[0,3,1,2]))) # 16 3 64 64

        # mask
        self.maskloss = 0.00005*tf.reduce_sum(tf.abs(mask - maskgt))
        # self.maskloss = 0.004*tf.reduce_mean(tf.abs(mask - maskgt))

        # self.outloss = tf.reduce_mean(tf.abs(out - x))

        # rendering loss
        self.renderingloss = tf.reduce_mean(tf.abs(AE_pointrecon - pointrecon))


        self.g_loss = self.generatorloss + self.albedoloss + self.normalloss + self.unitnormloss + self.lightloss + self.shadingsmoothloss + self.reconloss + self.maskloss + self.weightloss + self.renderingloss
        # self.g_loss = self.generatorloss + self.albedoloss + self.normalloss + self.unitnormloss + self.lightloss + self.shadingsmoothloss + self.shadingbwsloss + self.reconloss + self.maskloss + self.weightloss
        # self.g_loss = self.renderingloss + self.generatorloss + self.albedoloss + self.normalloss + self.unitnormloss + self.lightloss + self.shadingloss + self.reconloss


        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss # gamma = 0.5
        # self.balance = self.gamma * self.d_loss_real - 0.5*(self.renderingloss + self.generatorloss)# gamma = 0.5
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("AE_input", self.AE_x),
            tf.summary.image("AE_recon", self.AE_recon),
            # tf.summary.image("AE_recon2", self.AE_recon2),
            tf.summary.image("G_normal", self.normal),
            tf.summary.image("G_mask", self.mask),
            tf.summary.image("G_albedo", self.albedo),
            tf.summary.image("G_shading", self.shading),
            # tf.summary.image("G_shading2", self.shading2),
            tf.summary.image("G_recon", self.recon),
            # tf.summary.image("G_recon2", self.recon2),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/renderingloss", self.renderingloss),
            tf.summary.scalar("loss/generatorloss", self.generatorloss),
            tf.summary.scalar("loss/normalloss", self.normalloss),
            tf.summary.scalar("loss/albedoloss", self.albedoloss),
            tf.summary.scalar("loss/maskloss", self.maskloss),
            # tf.summary.scalar("loss/albedosmoothloss", self.albedosmoothloss),
            tf.summary.scalar("loss/shadingsmoothloss", self.shadingsmoothloss),
            tf.summary.scalar("loss/shadingbwsloss", self.shadingbwsloss),
            # tf.summary.scalar("loss/normalsmoothloss", self.normalsmoothloss),
            tf.summary.scalar("loss/unitnormloss", self.unitnormloss),
            tf.summary.scalar("loss/lightloss", self.lightloss),
            tf.summary.scalar("loss/weightloss", self.weightloss),
            # tf.summary.scalar("loss/shadingloss", self.shadingloss),
            tf.summary.scalar("loss/reconloss", self.reconloss),
            # tf.summary.scalar("loss/outloss", self.outloss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

    # def build_test_model(self):
        # with tf.variable_scope("test") as vs:
        #     # Extra ops for interpolation
        #     z_optimizer = tf.train.AdamOptimizer(0.0001)
        #
        #     self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
        #     self.z_r_update = tf.assign(self.z_r, self.z)
        #
        # G_z_r, _ = GeneratorCNN(
        #         self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)
        #
        # with tf.variable_scope("test") as vs:
        #     self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
        #     self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])
        #
        # test_variables = tf.contrib.framework.get_variables(vs)
        # self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, path, idx=None):
        inputs = inputs.transpose([0, 3, 1, 2])

        lightgt, light, weight, normal, mask, shading, albedo, recon, pointrecon, avgr, avgg, avgb = self.sess.run([self.lightgt, self.light, self.shadingweight, self.normal, self.mask, self.shading, self.albedo, self.recon, self.pointrecon, self.avg_r, self.avg_g, self.avg_b], {self.x: inputs})


        mask_path = os.path.join(path, '{}_M.png'.format(idx))
        save_image(mask, mask_path)
        print("[*] Samples saved: {}".format(mask_path))

        normal_path = os.path.join(path, '{}_N.png'.format(idx))
        save_image(normal, normal_path)
        print("[*] Samples saved: {}".format(normal_path))

        normal2_path = os.path.join(path, '{}_N2.png'.format(idx))
        save_image(normal*mask/255., normal2_path)

        shading_path = os.path.join(path, '{}_S.png'.format(idx))
        save_image(shading, shading_path)
        print("[*] Samples saved: {}".format(shading_path))

        shading2_path = os.path.join(path, '{}_S2.png'.format(idx))
        save_image(shading*mask/255., shading2_path)

        albedo_path = os.path.join(path, '{}_A.png'.format(idx))
        save_image(albedo, albedo_path)
        print("[*] Samples saved: {}".format(albedo_path))

        albedo2_path = os.path.join(path, '{}_A2.png'.format(idx))
        save_image(albedo*mask/255., albedo2_path)

        recon_path = os.path.join(path, '{}_R.png'.format(idx))
        save_image(recon, recon_path)
        print("[*] Samples saved: {}".format(recon_path))

        recon2_path = os.path.join(path, '{}_R2.png'.format(idx))
        save_image(recon*mask/255., recon2_path)

        # light coefficient print and save as txt
        lightgt_path = os.path.join(path, '{}_lightgt.txt'.format(idx))
        light_path = os.path.join(path, '{}_light.txt'.format(idx))
        weight_path = os.path.join(path, '{}_weight.txt'.format(idx))
        np.savetxt(lightgt_path, lightgt)
        np.savetxt(light_path, light)
        np.savetxt(weight_path, weight)


        print (avgr)
        print (avgg)
        print (avgb)

        for step in range(12):
            pointrecon_path = os.path.join(path, '{}_PR{}.png'.format(idx, step))
            save_image(pointrecon[step*16:(step+1)*16], pointrecon_path)
            print("[*] Samples saved: {}".format(pointrecon_path))


        # shading2_path = os.path.join(path, '{}_S2.png'.format(idx))
        # save_image(shading2, shading2_path)
        # print("[*] Samples saved: {}".format(shading2_path))
        #
        # recon2_path = os.path.join(path, '{}_R2.png'.format(idx))
        # save_image(recon2, recon2_path)
        # print("[*] Samples saved: {}".format(recon2_path))



    # def autoencode(self, inputs, path, idx=None, x_fake=None):
    def autoencode(self, inputs, path, idx=None):

        inputs = inputs.transpose([0, 3, 1, 2])

        AE_x, AE_recon, mask = self.sess.run([self.AE_x, self.AE_recon, self.mask], {self.x: inputs})

        AE_x_path = os.path.join(path, '{}_D_real.png'.format(idx))
        save_image(AE_x, AE_x_path)
        print("[*] Samples saved: {}".format(AE_x_path))

        AE_x2_path = os.path.join(path, '{}_D_real2.png'.format(idx))
        save_image(AE_x*mask/255., AE_x2_path)
        print("[*] Samples saved: {}".format(AE_x2_path))


        AE_recon_path = os.path.join(path, '{}_D_recon.png'.format(idx))
        save_image(AE_recon, AE_recon_path)
        print("[*] Samples saved: {}".format(AE_recon_path))

        AE_recon2_path = os.path.join(path, '{}_D_recon2.png'.format(idx))
        save_image(AE_recon*mask/255., AE_recon2_path)
        print("[*] Samples saved: {}".format(AE_recon2_path))


        # AE_recon2_path = os.path.join(path, '{}_D_recon2.png'.format(idx))
        # save_image(AE_recon2, AE_recon2_path)
        # print("[*] Samples saved: {}".format(AE_recon2_path))

    def relight(self, inputs, path, idx=None):

        inputs = inputs.transpose([0, 3, 1, 2])

        newshading, newrecon, newlight, mask = self.sess.run([self.newshading, self.newrecon, self.newlight, self.mask], {self.x: inputs})

        newshading_path = os.path.join(path, '{}_newshading.png'.format(idx))
        save_image(newshading, newshading_path)
        print("[*] Samples saved: {}".format(newshading_path))

        newshading2_path = os.path.join(path, '{}_newshading2.png'.format(idx))
        save_image(newshading*mask/255., newshading2_path)
        print("[*] Samples saved: {}".format(newshading2_path))

        newrecon_path = os.path.join(path, '{}_newrecon.png'.format(idx))
        save_image(newrecon, newrecon_path)
        print("[*] Samples saved: {}".format(newrecon_path))

        newrecon2_path = os.path.join(path, '{}_newrecon2.png'.format(idx))
        save_image(newrecon*mask/255., newrecon2_path)
        print("[*] Samples saved: {}".format(newrecon2_path))

        # light coefficient print and save as txt
        newlight_path = os.path.join(path, '{}_newlight+weight.txt'.format(idx))
        np.savetxt(newlight_path, newlight)



    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size/2)

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test(self):
        # root_path = "./"#self.model_dir

        all_G_z = None
        for step in range(3):
            # real1_batch = self.get_image_from_loader()
            # real2_batch = self.get_image_from_loader()
            x_fixed, normal_fixed, mask_fixed, light_fixed = self.get_image_from_loader() # 16 64 64 3


            print (self.model_dir)
            # print (x_fixed.shape) # 16 64 64 3
            now = datetime.datetime.now()
            # result_dir = os.path.join(self.model_dir,'_testresult/%s'%(now.strftime('%m%d_%H%M')))
            result_dir = self.model_dir + '_testresult/%s'%(now.strftime('%m%d_%H%M'))

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            # save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))
            save_image(x_fixed, os.path.join(result_dir, 'test{}_input.png'.format(step)))
            save_image(x_fixed*mask_fixed/255., os.path.join(result_dir, 'test{}_input*mask.png'.format(step)))

            # self.autoencode(
            #         real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
            # self.autoencode(
            #         real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.autoencode(
                    x_fixed, result_dir, idx="test{}".format(step))


            # self.interpolate_G(real1_batch, step, root_path)
            #self.interpolate_D(real1_batch, real2_batch, step, root_path)

            # z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            # G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))
            self.generate(x_fixed, result_dir, idx="test{}".format(step))

            self.relight(x_fixed, result_dir, idx="test{}".format(step))


            # if all_G_z is None:
            #     all_G_z = G_z
            # else:
            #     all_G_z = np.concatenate([all_G_z, G_z])
            # save_image(all_G_z, '{}/G_z{}.png'.format(root_path, step))

        # save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def get_image_from_loader(self):

        rgb, normal, mask, light = self.sess.run([self.data_loader, self.normal_loader, self.mask_loader, self.light_loader])

        if self.data_format == 'NCHW':
            rgb = rgb.transpose([0, 2, 3, 1]) # for image saving 16 3 64 64 to 16 64 64 3
            normal = normal.transpose([0, 2, 3, 1])
            mask = mask.transpose([0, 2, 3, 1])

        return rgb, normal, mask, light
