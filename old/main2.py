
from scipy.io.wavfile import read, write
import matplotlib
import scipy
import numpy as np
import pandas
import model

import matplotlib.pyplot as plt
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

video_name0 = 'data/sao_op1.mp4'
audio_name0 = 'data/sao_op1.wav'
sr, audio0 = read(audio_name0)
#s2, test_audio = read('test.wav')
#write('test.wav', 44100, test_audio)
#sr: 44100, audio: (3926016, 2). audio/sr = 90
#sample rate is 44100. 90 seconds of audio
#video.get_length() = 2668
#video sample rate = 30
#1470 audio amplitudes per video frame
vid = imageio.get_reader(video_name0,  'ffmpeg')
vid_len = vid.get_length()
aud_len = audio0.shape[0]
audio_per_vid_clip = int(round(aud_len / vid_len))
alen = audio_per_vid_clip
start = 1000000
end = 2000000
clip_len = end - start
b=np.array([(ele/2**8.)*2-1 for ele in audio[start:end,:]])# this is 8-bit track, b is now normalized on [-1,1)

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
audio_second = 44100
video_second = 33
for i in range(100):
    for batch_no in range(10,100):
        audio_start = batch_no * audio_second
        audio_end = (batch_no + 1) * audio_second
        video_start = batch_no * video_second
        video_end = (batch_no+1) * video_second
        video1
        _, _, g_loss, d_loss, audio   = sess.run(
                    [g_optim, d_optim, gan.g_loss, gan.d_loss, gan.gen_audio],
                    feed_dict = {
                        gan.video : vid_clip,
                        gan.wrong_video : wrong_video,
                        gan.real_audio : audio,
                        gan.z_noise : np.random.rand(bs, z_len)
                    })



for b_idx in range(1000,vid_len):
    vid_clip = vid.get_data(b_idx)
    audio_clip = audio[b_idx * alen:(b_idx+1)*alen,:]
#c = np.fft.rfft(b, axis=0, n=10000) # calculate fourier transform (complex numbers list)
c = np.fft.rfft(b, axis=0) # calculate fourier transform (complex numbers list)
frequencies = [n*2 for n in range(2,10000)]
idxes = [1 if i not in frequencies else 0 for i in range(int(clip_len/2))]
idxes = np.where(idxes)
c[idxes[0],:] = 0
#c = np.fft.rfft(b, axis=0)
d = np.fft.irfft(c, axis=0, n=clip_len)
#d = np.fft.irfft(np.fft.rfft(b, axis=0), axis=0)
write('test.wav', 44100,d)
c2 = np.fft.rfft(b, axis=0,n=10)
c3 = np.fft.hfft(b, n=10)
d = int(round(len(c)/2))
plt.plot(abs(c[:(d-1)]),'r') 
plt.close()
target = c[:]
plt.plot(audio)
audio = audio[10000:11000,:]
audio_scaled = [(ele/2**8.)*2-1 for ele in audio]
freq_domain_audio = np.fft.rfft(audio_scaled)
reverse_audio = np.fft.irfft(freq_domain_audio)
write('test.wav', 44100, reverse_audio)
nums = [10, 287]
for num in nums:
    image = vid.get_data(num)
    fig = pylab.figure()
    fig.suptitle('image #{}'.format(num), fontsize=20)
    pylab.imshow(image)
pylab.show()
