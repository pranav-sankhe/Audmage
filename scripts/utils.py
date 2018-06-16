import os
import pylab
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write
from scipy import signal
import librosa
import librosa.display

def covariance(seq1, seq2, plot_true):
	seq1 = np.array(seq1)
	seq2 = np.array(seq2)

	cov = np.cov(seq1, seq1)
	if plot_true == True:
		plt.plot(cov)
	return cov	



def correlation(array1, array2, input_size, plot_true):
	if input_size == 1:
		cor = np.correlate(array1, array2, "full")
		if plot_true == True: 
			plt.plot(cor)

	if input_size == 2: 

		cor = signal.correlate2d(array1, array2, boundary='symm', mode='same')		
		if plot_true == True:
			plt.plot(corr)

	return cor		 	


def mfcc(y,sr, n_mfcc,plotFlag):
	
	mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
	write('fut.wav', sr, y)		#write file under test
	if plotFlag:
		librosa.display.specshow(mfccs, x_axis='time')
		plt.colorbar()
		plt.title('MFCC')
		plt.tight_layout()
		plt.show()


def spectrogram(y, hop_length, sr, plotFlag):


	write('fut.wav', sr, y)		#write file under test
	D = librosa.stft(y, hop_length=hop_length)
	#D_left = librosa.stft(y, center=False)

	#D_short = librosa.stft(y, hop_length=64)

	if plotFlag:
		librosa.display.specshow(librosa.amplitude_to_db(D,
		                                                 ref=np.max),
		                         y_axis='log', x_axis='time')
		plt.title('Power spectrogram')
		plt.colorbar(format='%+2.0f dB')
		plt.tight_layout()
		
		plt.show()
