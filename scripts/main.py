import utils
import librosa


filepath = '../test_audio/turkish_march.mp3' 
y, sr = librosa.load(filepath)
y = y[0:200000]

hop_length = 32
utils.spectrogram(y, hop_length, sr, plotFlag=False,flag_hp=True)
# n_coeffs = 40
# utils.mfcc(y, sr, n_coeffs, plotFlag=True)
# utils.plotTimeSeries(y,sr, flag_hp=True)
# r, _ =  utils.zero_crossing(y,sr)
# print utils.spectral_centroid(y,sr)
# utils.plotSpectrum(y,sr, flag_hp=True)
# utils.cepstral_analysis(y, sr, plotFlag=True)