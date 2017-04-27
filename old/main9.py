
from scipy.io.wavfile import read, write
import matplotlib
import scipy
import numpy as np
import pandas
import model9 as model
import os
import sys
from scipy.misc import imresize

#import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import pylab
import imageio


import tensorflow as tf
from Utils import ops
from tensorflow.contrib.distributions import MultivariateNormalDiag as tf_norm

prince = True

def reshape(x, arr):
    return tf.reshape(x, [int(a) for a in arr])

def tf_resize(x, size):
    return tf.image.resize_images(x, (size, size))

audio = [None]*10
vid = [None]*10
files = os.listdir('data/audios')
max_len = 20
audio_second = 8000
video_second = 33
for i, n in enumerate(files):
    print(i, n)
    n = n[:-4]
    video_name = 'data/videos/' + n + '.mp4'
    audio_name = 'data/audios/' + n + '.wav'
    sr, audio[i] = read(audio_name)
    audio[i] = audio[i][8000:8000*max_len,:]
    vid[i] = imageio.get_reader(video_name,  'ffmpeg')
    vid[i] = [imresize(vid[i].get_data(j), [16,16,3]) for j in range(33, max_len*33)]
#b=np.array([(ele/2**8.)*2-1 for ele in audio[start:end,:]])# this is 8-bit track, b is now normalized on [-1,1)
vid = np.array(vid)
####PROCESSING AND PROOF OF CONCEPT:
aud_saved = audio
audio = (np.array(audio).astype('float') - np.min(audio))
audio = (2 * audio/np.max(audio)) - 1
a =  audio * 26600
a = a.astype('int16')
a2 = aud_saved[7]
a3 = a[7,:,:]
write('debug2.wav', 8000, a3[:24000,:])
gan = model.GAN()
gan.build_model()

g_optim = gan.g_optim
d_optim = gan.d_optim
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)
bs = 10
z_len = 100
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
g_loss = 1
g_reg = 1
for i in range(10000):
    for batch_no in range(3, 18):
        batch_no = 10
        audio_start = batch_no * audio_second
        audio_end = (batch_no + 1) * audio_second
        video_start = batch_no * video_second
        video_end = (batch_no+1) * video_second
        videos = vid[:,video_start:video_end,:,:,:]
        shuf = np.random.randint(2, 6)
        wrongvideos = np.concatenate([videos[shuf:,:,:,:,:], videos[:shuf,:,:,:,:]], 0)
        #audios = [audio[j][audio_start:audio_end,:] for j in range(10)]
        #audio_clip = np.array(audios)
        audio_clip = audio[:,audio_start:audio_end,:]
        if 1:#g_loss < 1.7:
            _, _, g_loss, g_reg, d_loss   = sess.run(
                    [g_optim, d_optim, gan.g_loss, gan.g_reg, gan.d_loss],
                    feed_dict = {
                        gan.video : videos,
                        gan.wrong_video : wrongvideos,
                        gan.real_audio : audio_clip,
                        gan.z_noise : np.random.rand(bs, z_len)
                    })
        print('epoch: ', i, 'batch: ', batch_no, 'g_loss:', g_loss, 'g_reg', g_reg, 'd_loss', d_loss)
        for _ in range(1):
            _, g_loss, g_reg   = sess.run(
                        [g_optim, gan.g_loss, gan.g_reg],
                        feed_dict = {
                            gan.video : videos,
                            gan.wrong_video : wrongvideos,
                            gan.real_audio : audio_clip,
                            gan.z_noise : np.random.rand(bs, z_len)
                        })
        if batch_no % 10 == 0:
            _, g_loss, g_reg, gen_audio_out, half_audio_out, real_audio_out   = sess.run(
                        [g_optim, gan.g_loss, gan.g_reg, gan.gen_audio, gan.half_audio, gan.real_audio],
                        feed_dict = {
                            gan.video : videos,
                            gan.wrong_video : wrongvideos,
                            gan.real_audio : audio_clip,
                            gan.z_noise : np.random.rand(bs, z_len)
                        })
            gen_audio_out = gen_audio_out * 26600
            gen_audio_out = gen_audio_out.astype('int16')
            half_audio_out = half_audio_out * 26600
            half_audio_out = half_audio_out.astype('int16')
            real_audio_out = real_audio_out * 26600
            real_audio_out = real_audio_out.astype('int16')
            for idx in range(4,7):
                try:
                    write('gen_audio_out' + str(idx) + '.wav', 8000, gen_audio_out[idx,:,:])
                    write('real_audio_out' + str(idx) + '.wav', 8000, real_audio_out[idx,:,:])
                    write('half_audio_out' + str(idx) + '.wav', 8000, half_audio_out[idx,:,:])
                except:
                    pass

