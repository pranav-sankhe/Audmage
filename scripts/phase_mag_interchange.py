import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write
from numpy import exp, abs, angle
from scipy import signal

def img_fft():
	img = cv2.imread('lenna.jpeg',0)
	f = np.fft.fft2(img)                            #fourier transform
	#fshift = np.fft.fftshift(f)                     # Shift the zero-frequency component to the center of the spectrum.
	# magnitude_spectrum = 20*np.log(np.abs(fshift))

	#f_ishift = np.fft.ifftshift(fshift)             #The inverse of fftshift

	img_back = np.fft.ifft2(f)               #This function computes the inverse of the 2-dimensional discrete Fourier Transform  
	img_back = np.abs(img_back)

	return f

def polar2z(r,theta):
    return r * exp( 1j * theta )

def audio_fft():

	rate, data = wav.read('/Users/sankhe/Documents/projects/Perceived-Music-Information-Retrieval/audio/full.v1/S22_Harry Potter Theme.wav')
	fft_out = fft(data)
	signs = data/np.absolute(data)
	return [fft_out, signs]


def output_audio(audio_fft,sign,img_fft):


	mag_img = np.absolute(img_fft)
	phase_img = np.angle(img_fft)

	mag_audio = np.absolute(audio_fft)
	phase_audio = np.angle(audio_fft)

	#print audio_fft, img_fft
	#print mag_img, mag_audio
	#print phase_img, phase_audio

	mag_img = (mag_img - mag_img.min())/(mag_img.max() - mag_img.min())
	mag_audio = (mag_audio - mag_audio.min())/(mag_audio.max() - mag_audio.min())


	mag_img_fl = mag_img.ravel()
	img_l = len(mag_img_fl)
	audio_l = len(mag_audio)
	mag_img_new = np.zeros((audio_l))
	
	for i in range(audio_l):
		mag_img_new[i] = mag_img_fl[ i - (i/img_l)*img_l]
		#print mag_img_new[i]

	for i in range(mag_audio.shape[0]):
		mag_audio[i,0] = mag_audio[i,0]  + mag_img_new[i] 
		mag_audio[i,1] = mag_audio[i,1]  + mag_img_new[i]




	phase_img = (phase_img - phase_img.min())/(phase_img.max() - phase_img.min())
	phase_audio = (phase_audio - phase_audio.min())/(phase_audio.max() - phase_audio.min())
	
	phase_img_fl = phase_img.ravel()
	img_l = len(phase_img_fl)
	audio_l = len(phase_audio)
	phase_img_new = np.zeros((audio_l))
	
	for i in range(audio_l):
		phase_img_new[i] = phase_img_fl[ i - (i/img_l)*img_l]
		#print phase_img_new[i]

	for i in range(phase_audio.shape[0]):
		phase_audio[i,0] = phase_audio[i,0]  + phase_img_new[i] 
		phase_audio[i,1] = phase_audio[i,1]  + phase_img_new[i]


	#print mag_audio, mag_audio.shape
	#print phase_audio, phase_audio.shape
	
	y = polar2z(mag_audio*100,phase_audio)
	yinv = ifft(y)
	yinv = np.absolute(yinv)
	yinv = yinv*sign
	write('test.wav', 44100, yinv)


def output_img(audio_fft,img_fft):

	mag_img = np.absolute(img_fft)
	phase_img = np.angle(img_fft)

	mag_audio = np.absolute(audio_fft)
	phase_audio = np.angle(audio_fft)

	# mag_img = (mag_img - mag_img.min())/(mag_img.max() - mag_img.min())
	# mag_audio = (mag_audio - mag_audio.min())/(mag_audio.max() - mag_audio.min())

	# phase_img = (phase_img - phase_img.min())/(phase_img.max() - phase_img.min())
	# phase_audio = (phase_audio - phase_audio.min())/(phase_audio.max() - phase_audio.min())


	mag_audio_trunc1 = mag_audio[:,0]
	mag_audio_trunc2 = mag_audio[:,1]	

	mag_audio_trunc1 = mag_audio_trunc1[0:mag_img.shape[0]*mag_img.shape[1]]
	mag_audio_2D1 = np.reshape(mag_audio_trunc1, mag_img.shape)	

	mag_audio_trunc2 = mag_audio_trunc2[0:mag_img.shape[0]*mag_img.shape[1]]
	mag_audio_2D2 = np.reshape(mag_audio_trunc2, mag_img.shape)	


	mag_img =  mag_img +  mag_audio_2D1*1000
 	#mag_img =  np.convolve(mag_img ,mag_audio_2D1*10)
 	#mag_img = signal.convolve2d(mag_img ,mag_audio_2D1*100, boundary='symm', mode='same')

	phase_audio_trunc1 = phase_audio[:,0]
	phase_audio_trunc2 = phase_audio[:,1]	

	phase_audio_trunc1 = phase_audio_trunc1[0:phase_img.shape[0]*phase_img.shape[1]]
	phase_audio_2D1 = np.reshape(phase_audio_trunc1, phase_img.shape)	

	phase_audio_trunc2 = phase_audio_trunc2[0:phase_img.shape[0]*phase_img.shape[1]]
	phase_audio_2D2 = np.reshape(phase_audio_trunc2, phase_img.shape)	

	phase_img = phase_img + phase_audio_2D1
	y = polar2z(mag_img,phase_img)
	y = np.fft.ifftshift(y)             #The inverse of fftshift

	img_back = np.fft.ifft2(y)               #This function computes the inverse of the 2-dimensional discrete Fourier Transform  
	img_back = np.abs(img_back)
	
	img = cv2.imread('lenna.jpeg',0)
	plt.subplot(121),plt.imshow(img, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
	plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

	plt.show()

output_img(audio_fft()[0],img_fft())



