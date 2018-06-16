import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write
from numpy import exp, abs, angle
from scipy import signal
import utils
def audio_fft():

    rate, data = wav.read('31beethovens3a.wav')
    fft_out = fft(data)
    signs = data/np.absolute(data)
    return [fft_out, signs]


#mrate, data = wav.read('mond_1.wav')
# #print data[0,:]
# chunk = data[:,0]
# kernel = [-2,0,-2]
# conv1 = np.convolve(chunk, kernel)

# chunk = data[:,1]
# kernel = [-2,0,-2]
# conv2 = np.convolve(chunk, kernel)
# conv = np.hstack((conv1, conv2))
# conv = conv.astype(np.float32)
# conv = conv/10000
# write('conv.wav', 44100, conv)



