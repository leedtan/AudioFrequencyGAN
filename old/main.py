
from scipy.io.wavfile import read, write
import matplotlib
import scipy
import numpy as np
import pandas

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import pylab
import imageio
filename = 'data/videos/sao_op1.mp4'
audio_name = 'data/audios/sao_op1.wav'
sr, audio = read(audio_name)
s2, test_audio = read('test.wav')
write('test.wav', 8000, audio)
write('test1.wav',8000,audio[100000:108000,:])
'''
audio
array([[ 0,  0],
       [ 0,  0],
       [ 0,  0],
       ..., 
       [-6, -8],
       [-6, -8],
       [-6, -8]], dtype=int16)
audio.max()
15912
'''

#sr: 44100, audio: (3926016, 2). audio/sr = 90
#video.get_length() = 2668
vid = imageio.get_reader(filename,  'ffmpeg')
vid_len = vid.get_length()
aud_len = audio.shape[0]
audio_per_vid_clip = int(round(aud_len / vid_len))
alen = audio_per_vid_clip
for b_idx in range(1000,vid_len):
    vid_clip = vid.get_data(b_idx)
    audio_clip = audio[b_idx * alen:(b_idx+1)*alen,:]
    #maybe unnecessary or wrong:
start = 1000000
end = 2000000
clip_len = end - start
b=np.array([(ele/2**8.)*2-1 for ele in audio[start:end,:]])# this is 8-bit track, b is now normalized on [-1,1)
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
