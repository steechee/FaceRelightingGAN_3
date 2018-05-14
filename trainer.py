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
from utils import templight
from utils import getmatrix
from utils import getshading
from utils import getshadingnp

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

            self.build_test_model()

    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

        x_fixed, normal_fixed, mask_fixed, light_fixed = self.get_image_from_loader() # 16 64 64 3


        shading_fixed = np.transpose(getshadingnp(np.transpose((normal_fixed/127.5 -1),[0, 3, 1, 2]), light_fixed),[0, 2, 3, 1])
        albedo_fixed = np.clip((x_fixed/127.5 -1)/(shading_fixed + 1e-3), 0, 10)

        shading_fixed = np.clip(((shading_fixed+1)*127.5), 0, 255)
        albedo_fixed = np.clip(((albedo_fixed+1)*127.5), 0, 255)

        save_image(x_fixed, '{}/x_fixed_rgb.png'.format(self.model_dir))
        save_image(normal_fixed, '{}/x_fixed_normal.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/x_fixed_mask.png'.format(self.model_dir))
        save_image(shading_fixed, '{}/x_fixed_shading.png'.format(self.model_dir))
        save_image(albedo_fixed, '{}/x_fixed_albedo.png'.format(self.model_dir))
        # save_image((1-mask_fixed/255.)*x_fixed, '{}/x_fixed_bg.png'.format(self.model_dir))

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
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                    "d_loss_real": self.d_loss_real,
                    "d_loss_fake": self.d_loss_fake,
                    "balance": self.balance,
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0: # every 50 steps
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']
                d_loss_real = result['d_loss_real']
                d_loss_fake = result['d_loss_fake']
                balance = result['balance']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}, d_loss_real: {:.4f}, d_loss_fake: {:.4f}, balance: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, measure, k_t, d_loss_real, d_loss_fake, balance))

            # if step % (self.log_step * 10) == 0: # every 500 steps
            if step % (self.log_step) == 0: #
                # x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.generate(x_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step)
                # self.autoencode(x_fixed[0], self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                #cur_measure = np.mean(measure_history)
                #if cur_measure > prev_measure * 0.99:
                #prev_measure = cur_measure

    def build_model(self):
        self.x = self.data_loader #rgb #16 3 64 64

        self.normalgt = self.normal_loader
        self.maskgt = self.mask_loader
        self.lightgt = self.light_loader #16 27

        # self.bggt = self.x * (255. - self.maskgt)/255.
        # print (self.shadinggt.get_shape()) #16 3 64 64
        # print (self.albedogt.get_shape()) #16 3 64 64

        # print (self.x.dtype)
        # print (self.shadinggt.dtype)
        # print (self.albedogt.dtype)


        x = norm_img(self.x)
        normalgt = norm_img(self.normalgt)
        maskgt = norm_img(self.maskgt)
        # bggt = norm_img(self.bggt)

        # shadinggt = getshading(normalgt, self.lightgt)
        # albedogt = tf.clip_by_value(x/(shadinggt + 1e-3), 0, 10)

        # shadinggt = norm_img(self.shadinggt)
        # albedogt = norm_img(self.albedogt)

        # print (x.dtype)
        # print (shadinggt.dtype)
        # print (albedogt.dtype)

        self.z = tf.random_uniform(
                (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        G, mask, albedo, light, shading, recon, self.G_var = GeneratorCNN(
                self.x, self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format, reuse=False)


        # print (G.dtype)
        # print (shading.dtype)
        # print (albedo.dtype)

        # Z, z_n, self.Enc_var = Encoder(
        #         self.x, self.channel, self.z_num,
        #         self.repeat_num, self.conv_hidden_num, self.data_format)
        #
        # G, self.Dec_var = Decoder(
        #         Z, self.conv_hidden_num, self.channel,
        #         self.repeat_num, self.data_format, self.x, self.maskgt)
                # self.repeat_num, self.data_format, self.x, self.maskgt, self.albedogt, self.lightgt)

        # self.G_var = self.Enc_var + self.Dec_var


        self.G = denorm_img(G, self.data_format)
        self.mask = denorm_img(mask, self.data_format)

        self.shading = denorm_img(shading, self.data_format)
        self.albedo = denorm_img(albedo, self.data_format)
        self.recon = denorm_img(recon, self.data_format)

        self.out = self.mask/255.*self.recon + (1-(self.mask/255.))*tf.transpose(self.x,[0, 2, 3, 1])
        # self.out = self.mask/255.*self.recon + (1-(self.mask/255.))*tf.transpose(self.bggt,[0, 2, 3, 1])

        out = tf.transpose(norm_img(self.out), [0, 3, 1, 2])


        d_out, self.D_z, self.D_var = DiscriminatorCNN(
        tf.concat([recon, x, out], 0), self.channel, self.z_num, self.repeat_num,
        self.conv_hidden_num, self.data_format)
        AE_G, AE_x, AE_mat = tf.split(d_out, 3)
        # print (AE_x.get_shape) # 16 3 64 64

        self.AE_G, self.AE_x, self.AE_mat = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format), denorm_img(AE_mat, self.data_format)
        # print (self.AE_x.get_shape) # 16 64 64 3

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        # self.g_loss = tf.reduce_mean(tf.abs(AE_G - G))
        # self.g_loss = tf.reduce_mean(tf.abs(G - normalgt))
        # self.g_loss = tf.reduce_mean(tf.abs(AE_G - G)) + tf.reduce_mean(tf.abs(G - x))


        self.normalloss = tf.reduce_mean(tf.abs(G - normalgt))
        self.maskloss = tf.reduce_mean(tf.abs(mask - maskgt))
        # self.albedoloss = tf.reduce_mean(tf.abs(albedo - albedogt))
        self.lightloss = tf.reduce_mean(tf.abs(tf.concat([light[:,:9],light[:,10:19],light[:,20:29]],axis=-1) - self.lightgt))
        # self.lightloss = tf.reduce_mean(tf.abs(light - self.lightgt))
        # self.shadingloss = tf.reduce_mean(tf.abs(shading - shadinggt))
        self.reconloss = tf.reduce_mean(tf.abs(recon - x))

        self.g_loss = tf.reduce_mean(tf.abs(AE_G - G)) + self.normalloss + self.maskloss + self.lightloss + self.reconloss
        # self.g_loss = tf.reduce_mean(tf.abs(AE_G - G)) + self.normalloss + self.maskloss + self.albedoloss + self.lightloss + self.shadingloss + self.reconloss

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss # gamma = 0.5
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("AE_G", self.AE_G),
            tf.summary.image("AE_x", self.AE_x),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/normalloss", self.normalloss),
            tf.summary.scalar("loss/maskloss", self.maskloss),
            # tf.summary.scalar("loss/albedoloss", self.albedoloss),
            tf.summary.scalar("loss/lightloss", self.lightloss),
            # tf.summary.scalar("loss/shadingloss", self.shadingloss),
            tf.summary.scalar("loss/reconloss", self.reconloss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r, _ = GeneratorCNN(
                self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, path, idx=None):
        # print (inputs.shape)
        inputs = inputs.transpose([0, 3, 1, 2])
        # print (inputs.shape)
        x, mask, shading, albedo, recon, out = self.sess.run([self.G, self.mask, self.shading, self.albedo, self.recon, self.out], {self.x: inputs})
        print (np.amax(mask))
        print (np.amin(mask))

        x_path = os.path.join(path, '{}_N.png'.format(idx))
        save_image(x, x_path)
        print("[*] Samples saved: {}".format(x_path))

        mask_path = os.path.join(path, '{}_M.png'.format(idx))
        save_image(mask, mask_path)
        print("[*] Samples saved: {}".format(mask_path))

        albedo_path = os.path.join(path, '{}_A.png'.format(idx))
        save_image(albedo, albedo_path)
        print("[*] Samples saved: {}".format(albedo_path))

        shading_path = os.path.join(path, '{}_S.png'.format(idx))
        save_image(shading, shading_path)
        print("[*] Samples saved: {}".format(shading_path))

        recon_path = os.path.join(path, '{}_R.png'.format(idx))
        save_image(recon, recon_path)
        print("[*] Samples saved: {}".format(recon_path))

        out_path = os.path.join(path, '{}_out.png'.format(idx))
        # bggt = np.clip(np.transpose((bggt + 1)*127.5, [0, 2, 3, 1]), 0, 255)
        save_image(out, out_path)
        print("[*] Samples saved: {}".format(out_path))



    # def autoencode(self, inputs, path, idx=None, x_fake=None):
    def autoencode(self, inputs, path, idx=None):
        # items = {
        #     'real': inputs,
        #     'fake': x_fake,
        # }
        # for key, img in items.items():
        #     if img is None:
        #         continue
        #     if img.shape[3] in [1, 3]:
        #         img = img.transpose([0, 3, 1, 2])

            inputs = inputs.transpose([0, 3, 1, 2])

            # x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x, recond, out = self.sess.run([self.AE_x, self.AE_G, self.AE_mat], {self.x: inputs})
            x_path = os.path.join(path, '{}_D_real.png'.format(idx))
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

            recond_path = os.path.join(path, '{}_D_fake.png'.format(idx))
            save_image(recond, recond_path)
            print("[*] Samples saved: {}".format(recond_path))

            out_path = os.path.join(path, '{}_D_matting.png'.format(idx))
            save_image(out, out_path)
            print("[*] Samples saved: {}".format(out_path))


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
        root_path = "./"#self.model_dir

        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(
                    real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
            self.autoencode(
                    real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch, step, root_path)
            #self.interpolate_D(real1_batch, real2_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))

            if all_G_z is None:
                all_G_z = G_z
            else:
                all_G_z = np.concatenate([all_G_z, G_z])
            save_image(all_G_z, '{}/G_z{}.png'.format(root_path, step))

        save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def get_image_from_loader(self):
        # rgb = self.data_loader.eval(session=self.sess)
        # normal = self.normal_loader.eval(session=self.sess)
        # mask = self.mask_loader.eval(session=self.sess)
        # light = self.light_loader.eval(session=self.sess)

        rgb, normal, mask, light = self.sess.run([self.data_loader, self.normal_loader, self.mask_loader, self.light_loader])

        if self.data_format == 'NCHW':
            rgb = rgb.transpose([0, 2, 3, 1]) # for image saving 16 3 64 64 to 16 64 64 3
            normal = normal.transpose([0, 2, 3, 1])
            mask = mask.transpose([0, 2, 3, 1])

        return rgb, normal, mask, light
