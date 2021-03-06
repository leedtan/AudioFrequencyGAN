
from scipy.io.wavfile import read, write
import matplotlib
import scipy
import numpy as np
import pandas
import model as model
import os
import sys
import matplotlib.pyplot as plt
from scipy.misc import imresize

from scipy.fftpack import fft
from scipy.io import wavfile
import pylab
import imageio


import tensorflow as tf
from Utils import ops
from tensorflow.contrib.distributions import MultivariateNormalDiag as tf_norm

prince = True

def plot10(x, row, col, fig_name, title_label):
    f, a = plt.subplots(row, col*2, figsize=(col*5, row*1.8))
    for j in range(row):
        for i in range(col):
            idx = i + j*col
            a[j][i].plot(np.fft.irfft(x[idx,:,0] + x[idx,:,1] * 1j,axis=0).astype('int16'), label = 'audio amplitudes')
            #a[j,i].axis('off')
            a[j,i].set_title(title_label + ' time-domain')
            a[j,i].set_xticks([])
            a[j,i].legend()
    for j in range(row):
        for i in range(col):
            idx = i + j*col
            a[j][i+col].plot(x[idx,:,0], c='red', label='real axis')
            a[j][i+col].plot(x[idx,:,1], c='green', label='imaginary axis')
            #a[j,i].axis('off')
            a[j,i+col].set_title(title_label + ' freq-domain')
            a[j,i+col].set_xticks([])
            a[j,i+col].legend()
    f.savefig(fig_name)
    plt.close()
audio = [None]*10
vid = [None]*10
files = os.listdir('data/audios')
max_len = 15
audio_second = 4000
video_second = 33
a_time_clips = []
a_freq_clips = []
for i, n in enumerate(files):
    print(i, n)
    n = n[:-4]
    video_name = 'data/videos/' + n + '.mp4'
    audio_name = 'data/audios/' + n + '.wav'
    sr, audio[i] = read(audio_name)
    audio[i] = audio[i][8000*10:8000*(max_len+11),:]
    a_time_clips.append([])
    a_freq_clips.append([])
    for c_idx in range(max_len - 2):
        a_time_clips[-1].append(audio[i][c_idx * 8000: (c_idx + 1)*8000 - 2,:])
        a_freq_clips[-1].append(np.fft.rfft(a_time_clips[-1][-1][:,0], axis=0))
    vid[i] = imageio.get_reader(video_name,  'ffmpeg')
    vid[i] = [imresize(vid[i].get_data(j), [16,16,3]) for j in range(33, max_len*33)]

vid = np.array(vid)

####PROCESSING AND PROOF OF CONCEPT:
aud_saved = audio
freq_saved = a_freq_clips

fr = a_freq_clips[6][9][:]
au = np.fft.irfft(fr, axis=0)
au.shape #8000, 2
au = au * 3
au = au.astype('int16')
write('debug.wav', 8000, au)

a_freq = np.array(a_freq_clips)#10, 13, 4001, 2

audio = (np.array(audio).astype('float') - np.min(audio))
audio = (2 * audio/np.max(audio)) - 1
a =  audio * 26600
a = a.astype('int16')
a2 = aud_saved[7]
a3 = a[7,:,:]
write('debug2.wav', 8000, a3[:24000,:])

#a freq dimension batch, time index, time signal
a_from_freq = np.fft.irfft(a_freq[6, 7, :], axis=0)
write('debug4.wav', 8000, a_from_freq.astype("int16"))

audio = np.concatenate([np.expand_dims(a_freq.real,3), np.expand_dims(a_freq.imag,3)], axis=3)
home_output_folder = 'outputs4/'
if not os.path.exists(home_output_folder):
    os.makedirs(home_output_folder)
#This reversable transformation maps the audio files to [-1,1] cleanly.
#For a full release upon more success, this should be a function of the data.
scale_divisor = 57.5
audio_power = 4
audio_raw = audio
audio = np.sign(audio_raw)*np.power(np.abs(audio_raw), 1/audio_power)/scale_divisor
print('audio processed')
gan = model.GAN()
gan.build_model()
print('model defined')

g_optim = gan.g_optim
d_optim = gan.d_optim
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
bs = 10
z_len = 10
sess = tf.Session()
print('initializing variables')
sess.run(tf.global_variables_initializer())
print('variables initialized')
saver = tf.train.Saver()
g_loss = 1
g_reg = 1

saver = tf.train.Saver()
if 0:
    print('restoring model')
    saver.restore(sess, '/media/lee/datapart/AudioFrequencyGAN/saved_model4.ckpt')
    print('model restored')
for i in range(10000):
    if (i + 1) % 10 == 0:
        print('saving updated model')
        saver.save(sess, 'saved_model4.ckpt')
        print('done saving updated model')
    for batch_no in range(3, 10):
        print('batch started', batch_no)
        audio_start = batch_no * audio_second
        audio_end = (batch_no + 1) * audio_second
        video_start = batch_no * video_second
        video_end = (batch_no+1) * video_second
        videos = vid[:,video_start:video_end,:,:,:]
        shuf = np.random.randint(2, 6)
        wrongvideos = np.concatenate([videos[shuf:,:,:,:,:], videos[:shuf,:,:,:,:]], 0)
        audio_clip = audio[:,batch_no,:,:]
        if g_loss < 1.8:
            print('running discriminator')
            _, _, g_loss, g_reg, d_loss   = sess.run(
                    [g_optim, d_optim, gan.g_loss, gan.g_reg, gan.d_loss],
                    feed_dict = {
                        gan.video : videos,
                        gan.wrong_video : wrongvideos,
                        gan.real_audio : audio_clip,
                        gan.z_noise : np.random.rand(bs, z_len)
                    })
        _, g_loss, g_reg, d_loss   = sess.run(
                [g_optim, gan.g_loss, gan.g_reg, gan.d_loss],
                feed_dict = {
                    gan.video : videos,
                    gan.wrong_video : wrongvideos,
                    gan.real_audio : audio_clip,
                    gan.z_noise : np.random.rand(bs, z_len)
                })
        print('epoch: ', i, 'batch: ', batch_no, 'g_loss:', g_loss, 'g_reg', g_reg, 'd_loss', d_loss)
        if batch_no == 3:
            print('printing some audios')
            _, g_loss, g_reg, gen_audio, real_audio   = sess.run(
                        [g_optim, gan.g_loss, gan.g_reg, gan.gen_audio, gan.real_audio],
                        feed_dict = {
                            gan.video : videos,
                            gan.wrong_video : wrongvideos,
                            gan.real_audio : audio_clip,
                            gan.z_noise : np.random.rand(bs, z_len)
                        })
            gen_audio_out = np.power(gen_audio * scale_divisor, audio_power) * np.sign(gen_audio)
            real_audio_out = np.power(real_audio * scale_divisor, audio_power)* np.sign(real_audio)
            for _ in range(2):
                idx = np.random.randint(3,9)
                ade = audio_clip[idx, :,:]
                ade2 = np.fft.irfft(ade[:,0] + ade[:,1] * 1j,axis=0)
                hdr = 'outputs4/ep_' + str(i) + '_b_' + str(batch_no) + '_'
                
                #Start by reversing the transformation from [-1,1] to the real frequency signal:
                #Then reverse the Fourier Transform
                gen_audio_spl = np.fft.irfft(gen_audio_out[idx,:,0] + gen_audio_out[idx,:,1] * 1j,axis=0)
    
                real_audio_spl = np.fft.irfft(real_audio_out[idx,:,0] + real_audio_out[idx,:,1] * 1j,axis=0)
                
                #Write the Audio to file
                write(hdr + 'gen_audio_out' + str(idx) + '.wav', 7998, gen_audio_spl.astype('int16'))
                write(hdr + 'real_audio_out' + str(idx) + '.wav', 7998, real_audio_spl.astype('int16'))
    
                #Write the time-series amplitudes to file
                np.savetxt(hdr + 'np_gen_audio_out' + str(idx) + '.txt', gen_audio_spl.astype('int16'))
                np.savetxt(hdr + 'np_real_audio_out' + str(idx) + '.txt', real_audio_spl.astype('int16'))
                
                #generate image of audio wave
                plt.plot(gen_audio_spl.astype('int16'))
                plt.title('generated audio')
                plt.savefig(hdr + 'generated_audio_img' + str(idx))
                plt.close()
                plt.plot(real_audio_spl.astype('int16'))
                plt.title('real audio')
                plt.savefig(hdr + 'real_audio_img' + str(idx))
                plt.close()
                plot10(gen_audio_out, 5, 2, fig_name='outputs4/global_picture_generated' + str(i), title_label = 'Gen')
                plot10(real_audio_out, 5, 2, fig_name='outputs4/global_picture_real' + str(i), title_label = 'Real')
            print('done printing batch')
            



