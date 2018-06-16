import utils
import librosa


filepath = '../test_audio/turkish_march.mp3' 
y, sr = librosa.load(filepath)
y = y[:200000]

utils.spectrogram(y, 32, sr, plotFlag=False)
utils.mfcc(y,sr, 40, plotFlag=False)


