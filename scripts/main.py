import utils
import librosa
import os
import numpy as np
filepath = '../test_audio/440Hz_44100Hz_16bit_05sec.wav' 

filename = os.path.basename(filepath)

y, sr = librosa.load(filepath)
# y= y[80000:]
# y = y[0:200000]


# hop_length = 32
# utils.spectrogram(y, hop_length, sr, plotFlag=False,flag_hp=False, save_flag=True, filename=filename)
# n_mels = 128
# utils.mfcc(y, sr, n_mels, plotFlag=False, save_flag=True, filename=filename)
# utils.plotTimeSeries(y,sr, downsampleF=1, flag_hp=False, plotFlag=False, save_flag=True, filename=filename)
# r, _ =  utils.zero_crossing(y,sr)
# print utils.spectral_centroid(y,sr)
utils.plotSpectrum(y,sr, flag_hp=False, plotFlag=False, save_flag=True,  filename=filename)
y = utils.cepstral_analysis(y, sr, plotFlag=True)
# print utils.getPitch(y,sr)
# freq, max_freq, min_freq = utils.getFreq(y,sr)
# utils.melSpectrogram(y, sr, n_mels, max_freq, plotFlag=False,flag_hp=False, save_flag=True, filename=filename)


# Filter requirements.
order = 6
fs = sr       # sample rate, Hz
cutoff = 500  # desired cutoff frequency of the filter, Hz

utils.lpf(y, sr, order, fs, cutoff, freq_resp_plot=False, plotFlag=True)