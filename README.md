# AudioFrequencyGAN
The following is a GAN that generates Frequency domain Audio signals. Currently it generates 1 second of 8KHz audio, represented as a 2X4000 matrix of frequencies strengths real and imaginary signals.

The generated frequencies, when converted to time-domain, sound like noise at first, but that will change to tunes and/or beeps. but not yet to music or words. I have also tried time domain, also to no avail.

To install, install Tensorflow, imageio, ffmpeg, and run imageio.plugins.ffmpeg.download()

We also provide two helper scripts: `install-dependencies.sh` for installing the necessary dependencies.
