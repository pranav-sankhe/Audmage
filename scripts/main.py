import utils
import librosa
import os
import numpy as np
filepath = '../test_audio/poem.mp3' 

filename = os.path.basename(filepath)

y, sr = librosa.load(filepath)
y= y[80000:]
# y = y[0:200000]


# hop_length = 32
# utils.spectrogram(y, hop_length, sr, plotFlag=False,flag_hp=False, save_flag=True, filename=filename)
n_mels = 128
# utils.mfcc(y, sr, n_mels, plotFlag=False, save_flag=True, filename=filename)
utils.plotTimeSeries(y,sr, downsampleF=1, flag_hp=True, plotFlag=True, save_flag=True, filename=filename)
# r, _ =  utils.zero_crossing(y,sr)
# print utils.spectral_centroid(y,sr)
# utils.plotSpectrum(y,sr, flag_hp=True, plotFlag=False, save_flag=True,  filename=filename)
# # utils.cepstral_analysis(y, sr, plotFlag=True)
# print utils.getPitch(y,sr)
# freq, max_freq, min_freq = utils.getFreq(y,sr)
# utils.melSpectrogram(y, sr, n_mels, max_freq, plotFlag=True,flag_hp=False, save_flag=True, filename=filename)

# length = len(y)
# chunk_size = 100 
# num_parts = length/chunk_size
# idx = np.arange(num_parts)
# l = np.split(y, idx*chunk_size)

# for i in range(num_parts):
# 	if len(l[i]) >0:
# 		utils.melSpectrogram(l[i], sr, n_mels, max_freq, plotFlag=False,flag_hp=False, save_flag=True, filename= 'poem/' + filename + str(i))


