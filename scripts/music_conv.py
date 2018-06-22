import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write
from numpy import exp, abs, angle
from scipy import signal
import utils
import librosa
def audio_fft():

    rate, data = wav.read('31beethovens3a.wav')
    fft_out = fft(data)
    signs = data/np.absolute(data)
    return [fft_out, signs]


filepath = '../test_audio/poem.mp3' 
data, sr = librosa.load(filepath)
chunk = data
kernel = [-1,-10, -1, -10, -1]
conv1 = np.convolve(chunk, kernel)

# chunk = data[:,1]
# kernel = [1,0,1]
# conv2 = np.convolve(chunk, kernel)
# conv = np.hstack((conv1, conv2))
# conv = conv.astype(np.float32)
conv1 = conv1/100
write('conv.wav', sr, conv1)



