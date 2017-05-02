import tensorflow as tf
from Utils import ops

prince = True

def generate_gray_noise_audio(transform):
    s = transform.get_shape()[1].value
    bs = 10
    base = tf.random_uniform([bs,1,2], -.9, .9)
    base = tf.tile(base, [1, s, 1])
    noise = tf.random_uniform([bs,s,2],-.5,.5)
    random_mult = tf.random_uniform([bs,1,1],0.1,1)
    random_mult = tf.tile(random_mult,[1,s,2])
    return tf.clip_by_value(base + random_mult * noise,-.99,.99)


def moment4(x):
    s = x.get_shape()[1].value
    means = tf.tile(tf.expand_dims(tf.reduce_mean(x,axis=[1]),1), [1, s, 1, 1])
    diffs_fourthed = tf.pow(x - means, 4)
    return tf.pow( tf.reduce_mean(diffs_fourthed,axis=[1]),1/4)
def moment4_1d(x):
    s = x.get_shape()[1].value
    means = tf.tile(tf.expand_dims(tf.reduce_mean(x,axis=[1]),1), [1, s])
    diffs_fourthed = tf.pow(x - means, 4)
    return tf.pow( tf.reduce_mean(diffs_fourthed,axis=[1]),1/4)

def reshape(x, arr):
    return tf.reshape(x, [int(a) for a in arr])

def tf_resize(x, size):
    return tf.image.resize_images(x, (size, size))

def concat(dim, objects, name=None):
    if name is None:
        return tf.concat(objects, dim)
    else:
        return tf.concat(objects, dim, name = None)
def c_e(logits, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
def pack(x):
    return tf.stack(x)

class GAN:
    def __init__(self):
        
        self.batch_size = 10
        self.g_idx = 0
        
        self.d_idx = 0
        self.g_bn_idx = 0
        
    def d_bn(self):
        d_bn = ops.batch_norm(name='d_bn_clean' + str(self.d_bn_idx))
        self.d_bn_idx += 1
        return d_bn

    def g_bn(self):
        g_bn = ops.batch_norm(name='g_bn_clean' + str(self.g_bn_idx))
        self.g_bn_idx += 1
        return g_bn
        
    def add_residual_pre(self, prev_layer, k_h = 10, k_w = 1, hidden_filters = None,name_func=None):
        
        filters = prev_layer.get_shape()[3].value
        if name_func is None:
            name_func = self.g_name
        if hidden_filters ==None:
            hidden_filters = filters * 2
        s = prev_layer.get_shape()[1].value
        
        bn0 = self.g_bn()
        bn1 = self.g_bn()
        #ops.lrelu(ops.conv2d(audio, ddim, k_h = 100, k_w = 2, d_h = 4, d_w = 1, name='d_a1'))
        low_dim = ops.conv2d(ops.lrelu(bn0(prev_layer)), hidden_filters, k_h=k_h, k_w=k_w, d_h = 2, d_w = 1, name = name_func())
        
        residual = ops.deconv2d_audio(ops.lrelu(bn1(low_dim)), 
            [self.batch_size, s, 2, filters], k_h=k_h, k_w=k_w, d_h = 2, d_w = 1, name=name_func())
        
        next_layer = prev_layer + residual
        return next_layer
    def build_model(self):
        img_size = 16
        self.lr = 1e-5
        beta1 = .5
        bs = self.batch_size
        audio_freq = 4000
        vid_len = 33
        self.video = tf.placeholder('float32', [bs,vid_len, img_size, img_size, 3 ])
        self.wrong_video = tf.placeholder('float32', [bs,vid_len, img_size, img_size, 3 ])
        vid = tf.transpose(self.video, [0, 2, 3, 1, 4])
        w_vid = tf.transpose(self.wrong_video, [0, 2, 3, 1, 4])
        vid = tf.reshape(vid, [bs, img_size, img_size, vid_len * 3])
        w_vid = tf.reshape(w_vid, [bs, img_size, img_size, vid_len * 3])
        self.real_audio = tf.placeholder('float32', [bs,audio_freq,2])
        self.z_noise = tf.placeholder('float32', [bs,10 ])
        self.gray_noise = generate_gray_noise_audio(self.real_audio)
        self.gray2_noise = generate_gray_noise_audio(self.real_audio)
        self.gray3_noise = generate_gray_noise_audio(self.real_audio)
        
        self.gen_audio, l2_d1_grad, l2_d2_grad, l2_d3_grad, l2_d4_grad = self.generator(self.z_noise, vid)
        with tf.variable_scope('scope_disc'):
            real_img_logit, real_txt_logit, real_acts = self.discriminator(self.real_audio, vid)
            wrong_img_logit, wrong_txt_logit, wrong_acts = self.discriminator(self.real_audio, w_vid, reuse=True)
            gen_img_logit, gen_txt_logit, gen_acts = self.discriminator(self.gen_audio, vid, reuse=True)
        
        
        pos_ex = tf.ones_like(real_img_logit)
        neg_ex = tf.zeros_like(real_img_logit)
        
        self.d_loss_real = tf.reduce_mean(c_e(real_img_logit, pos_ex)) + tf.reduce_mean(c_e(real_txt_logit, pos_ex))*2
            
        self.d_loss_wrong = tf.reduce_mean(c_e(wrong_img_logit, pos_ex)) + tf.reduce_mean(c_e(wrong_txt_logit, neg_ex))
            
        self.d_loss_gen = tf.reduce_mean(c_e(gen_img_logit, neg_ex)) + tf.reduce_mean(c_e(gen_txt_logit, neg_ex))

        real_acts = [tf.reshape(r, [-1]) for r in real_acts]
        gen_acts = [tf.reshape(r, [-1]) for r in gen_acts]

        real_acts = concat(0, real_acts)
        gen_acts = concat(0, gen_acts)

        self.g_loss = tf.reduce_mean(c_e(gen_img_logit, pos_ex)) + tf.reduce_mean(c_e(gen_txt_logit, pos_ex))
        self.g_reg = tf.reduce_mean(tf.square(self.gen_audio - self.real_audio)) * 0 + \
                tf.reduce_mean(tf.square(real_acts - gen_acts)/tf.abs(real_acts)) * 3e3
        
        d_loss = self.d_loss_real + self.d_loss_wrong + self.d_loss_gen

        self.d_loss = d_loss
        
        g_loss = self.g_loss
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        
        d_l2_reg = tf.reduce_sum([tf.reduce_sum(tf.square(var)) * 1e0 for var in d_vars])
        g_l2_reg = tf.reduce_sum([tf.reduce_sum(tf.square(var))*1e-6 for var in g_vars])
        
        self.g_reg += g_l2_reg
        self.d_reg = d_l2_reg
        
        g_reg = self.g_reg
        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1 = beta1)
        gvs = optimizer.compute_gradients(g_loss + g_reg, var_list=g_vars)
        clip_max = 1
        clip = .01
        capped_gvs = [(tf.clip_by_value(grad, -1*clip,clip), var) for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_max), var) for grad, var in capped_gvs if grad is not None]
        self.g_optim = optimizer.apply_gradients(capped_gvs)
        self.g_gvs = [grad for grad, var in gvs if grad is not None]
        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1 = beta1)
        gvs = optimizer.compute_gradients(d_loss + d_l2_reg,var_list=d_vars)
        clip_max = 1
        clip = .01
        capped_gvs = [(tf.clip_by_value(grad, -1*clip,clip), var) for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_max), var) for grad, var in capped_gvs if grad is not None]
        self.d_optim = optimizer.apply_gradients(capped_gvs)
        self.d_gvs = [grad for grad, var in gvs if grad is not None]
        
    def g_name(self):
        self.g_idx += 1
        return 'g_' + str(self.g_idx)
    
    def d_name(self):
        self.d_idx += 1
        return 'd_generatedname_' + str(self.d_idx)
    
    
    def generator(self, z_noise, video):
        bs = self.batch_size

        bnv1 = self.g_bn()
        bnv2 = self.g_bn()
        bnv3 = self.g_bn()

        bna0 = self.g_bn()
        bna1 = self.g_bn()
        bna2 = self.g_bn()
        bna3 = self.g_bn()
        bna4 = self.g_bn()
        
        bnr0 = self.g_bn()
        bnr1 = self.g_bn()
        bnr2 = self.g_bn()
        
        bnl1 = self.g_bn()
        
        a_g_dim  = 4000
        a = a_g_dim
        a160, a80, a32, a16, a8, a4, a2 = int(a/160), int(a/80), int(a/32), int(a/16), int(a/8), int(a/4), int(a/2)
        vid = ops.lrelu(bnv1(ops.conv2d(video, 10, k_h = 10, k_w = 10, d_h = 2, d_w = 2, name = self.g_name())))
        vid = ops.lrelu(bnv2(ops.conv2d(vid, 20, k_h = 10, k_w = 10, d_h = 2, d_w = 2, name = self.g_name())))
        vid = ops.lrelu(bnv3(ops.conv2d(vid, 40, k_h = 10, k_w = 10, d_h = 2, d_w = 2, name = self.g_name())))
        hidden = reshape(vid, [bs, -1])
        hidden = concat(1, [hidden, z_noise])
        audio = ops.lrelu(bnl1(ops.linear(hidden, a80 * 10, self.g_name())))
        audio = reshape(audio, [bs, a160, 2, 10])
        g_dim = 50
        k_def = 10
        audio = ops.lrelu(bna0(ops.deconv2d_audio(audio, [bs, a80, 2, g_dim*4],k_h = 5, k_w = 1, d_h=2, d_w=1, name=self.g_name())))
        audio = self.add_residual_pre(audio, k_h = k_def, name_func = self.g_name)
        audio = ops.lrelu(bna1(ops.deconv2d_audio(audio, [bs, a16, 2, g_dim],k_h = 20, k_w = 1, d_h=5, d_w=1, name=self.g_name())))
        
        audio = reshape(audio, [bs, -1])
        audiores = ops.lrelu(bnr0(ops.linear(audio, 1000, self.g_name())))
        audiores = ops.linear(audiores, a16 * 2 * g_dim, self.g_name())
        audio = audio + audiores
        audio = reshape(audio, [bs, a16, 2, g_dim])
        
        audio = self.add_residual_pre(audio, k_h = k_def, name_func = self.g_name)
        audio = ops.lrelu(bna2(ops.deconv2d_audio(audio, [bs, a4, 2, g_dim],k_h = 30, k_w = 1, d_h=4, d_w=1, name=self.g_name())))
        
        audio = reshape(audio, [bs, -1])
        audiores = ops.lrelu(bnr1(ops.linear(audio, 1000, self.g_name())))
        audiores = ops.linear(audiores, a4 * 2 * g_dim, self.g_name())
        audio = audio + audiores
        audio = reshape(audio, [bs, a4, 2, g_dim])
        
        audio = self.add_residual_pre(audio, k_h = 20, name_func = self.g_name)
        audio = ops.lrelu(bna3(ops.deconv2d_audio(audio, [bs, 5000, 2, 1],k_h = 40, k_w = 1, d_h=5, d_w=1, name=self.g_name())))
        audio = self.add_residual_pre(audio, k_h = 40, name_func = self.g_name)
        audio = reshape(audio, [bs, -1])
        audiores = ops.lrelu(bnr2(ops.linear(audio, 1000, self.g_name())))
        audiores = ops.linear(audiores, 10000, self.g_name())
        audio = audio + audiores
        audio = reshape(audio, [bs, 5000, 2, 1])
        
        audio = ops.deconv2d_audio(audio, [bs, 5000, 2, 1],k_h = 10, k_w = 2, d_h=1, d_w=1, name=self.g_name())
        audio = self.add_residual_pre(audio, k_h = 10, name_func = self.g_name)
        audio = audio[:,1000:5000,:,:]
        
        audio = reshape(audio, [bs, a, 2])
        dist = 1
        shifted = tf.slice(audio, [0, 1, 0], [bs, 4000 - dist, 2])
        shifted2 = tf.slice(audio, [0,0,0],[bs,dist,2])
        shifted1 = tf.concat([shifted, shifted2], 1)
        diff = shifted1 - audio
        l2_d1_grad = tf.reduce_mean(tf.square(diff))
        dist = 2
        shifted = tf.slice(audio, [0, 1, 0], [bs, 4000 - dist, 2])
        shifted2 = tf.slice(audio, [0,0,0],[bs,dist,2])
        shifted1 = tf.concat([shifted, shifted2], 1)
        diff = shifted1 - audio
        l2_d2_grad = tf.reduce_mean(tf.square(diff))
        dist = 3
        shifted = tf.slice(audio, [0, 1, 0], [bs, 4000 - dist, 2])
        shifted2 = tf.slice(audio, [0,0,0],[bs,dist,2])
        shifted1 = tf.concat([shifted, shifted2], 1)
        diff = shifted1 - audio
        l2_d3_grad = tf.reduce_mean(tf.square(diff))
        dist = 4
        shifted = tf.slice(audio, [0, 1, 0], [bs, 4000 - dist, 2])
        shifted2 = tf.slice(audio, [0,0,0],[bs,dist,2])
        shifted1 = tf.concat([shifted, shifted2], 1)
        diff = shifted1 - audio
        l2_d4_grad = tf.reduce_mean(tf.square(diff))


        return tf.tanh(audio), l2_d1_grad, l2_d2_grad, l2_d3_grad, l2_d4_grad

    def blur_audio(self, audio):
        audio = (audio + self.shift_audio(audio, 1)) / 2
        return audio

    def shift_audio(self, audios, left):
        total_size = 4000
        audio = tf.map_fn(lambda img: tf.image.pad_to_bounding_box(
                    tf.image.crop_to_bounding_box(img, 1, total_size - left), left + 1, total_size), audios)
        return audio
    
    def discriminator(self, audio, video, reuse=False):
        tf.get_variable_scope()._reuse = None
        bs = self.batch_size
        a_len = audio.get_shape()[1].value
        a_sh = audio.get_shape()
        audio = reshape(audio, [a_sh[0], a_sh[1], a_sh[2], 1])
        s = video.get_shape()[1].value
        
        ddim = 50
        
        a1 = ops.lrelu(ops.conv2d(audio, ddim, k_h = 15, k_w = 1, d_h = 4, d_w = 1, name='d_a1'))
        a2 = ops.lrelu(ops.conv2d(a1, ddim*2, k_h = 15, k_w = 1, d_h = 4, d_w = 2, name='d_a2'))
        a3 = ops.lrelu(ops.conv2d(a2, ddim*4, k_h = 15, k_w = 1, d_h = 2, d_w = 1, name='d_a3'))
        a4 = ops.lrelu(ops.conv2d(a3, ddim*8, k_h = 15, k_w = 1, d_h = 5, d_w = 1, name='d_a4'))
        a5 = ops.lrelu(ops.conv2d(a4, ddim*16, k_h = 15, k_w = 1, d_h = 5, d_w = 1, name='d_a5'))
        
        a1o = reshape(a1, [bs, -1])
        a2o = reshape(a2, [bs, -1])
        a3o = reshape(a3, [bs, -1])
        a4o = reshape(a4, [bs, -1])
        a5o = reshape(a5, [bs, -1])
        a_flat = tf.concat([a1o, a2o, a3o, a4o, a5o], 1)
        
        v1 = ops.lrelu(ops.conv2d(video, 10, name='d_v1'))
        v2 = ops.lrelu(ops.conv2d(v1, 20, name='d_v2'))
        
        v_flat = reshape(v2, [bs, -1])
        
        av_flat = concat(1, [a_flat, v_flat])
        
        a_hidden = ops.lrelu(ops.linear(a_flat, 300, 'd_ah'))
        a_out = ops.linear(a_hidden, 1, 'd_aout')
        
        av_hidden = ops.lrelu(ops.linear(av_flat, 300, 'd_avh'))
        av_out = ops.linear(av_hidden, 1, 'd_avout')
        dist_params = []
        
        for v in [a1, a2, a3, a4, a5]:
            m, s = tf.nn.moments(v, axes=[1])
            m4 = moment4(v)
            dist_params = dist_params + [m, s, m4]
            
        for v in [a_hidden, av_hidden]:
            m4 = moment4_1d(v)
            m, s = tf.nn.moments(v, axes=[1])
            dist_params = dist_params + [m, s, m4]
            
        dist_len = len(dist_params)
        means = [None]*dist_len
        stds = [None]*dist_len
        for idx, dp in enumerate(dist_params):
            m, s = tf.nn.moments(dp, axes=[0])
            means[idx] = m
            stds[idx] = s
            
        return a_out, av_out, means + stds
    
