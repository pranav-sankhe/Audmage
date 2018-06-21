# Scripts Description 

## ```utils.py```

- ```python class ZoomPan ```: Class to enable interactive zoom by scrolling.
- ```python def covariance(seq1, seq2, plot_true)```: Function to compute covariance between ```seq1``` and ```seq2```. ```plt_true``` is a boolean argument which controls the plots. 
- ```python def rmsEnergy(y) ```: Roughly corresponds to how loud the music is. 
- ```python def spectral_centroid(y, sr) ```: Used to characterize spectrum. Indicates where the centre of mass of the spectru is. Perceptually, it has an impression of the brightness of the sound. 
- ```python def zero_crossing(y, sr) ```: Rate at which signal changes its signs. Highly used to classify percussive sounds.  
- ```pyhton def plotTimeSeries(y,sr, flag_hp) ```: Plots the signal in time domain. The flag 'flag_hp' is enabled, plots the harmonic and percussive components of the audio. 
- ```python plotSpectrum(y,sr,flag_hp) ```: Computes and plots the fourier transform of the given signal. [fft]
- ```python def spectrogram(y, hop_length, sr, plotFlag) ```: Computes and plots the spectrogram [STFT] of the given audio signal. 
- ``` python def mfcc(y,sr, n_mfcc,plotFlag) ```: Computes and plots the Mel Frequency Cepstral Coefficients of the given audio signal. 

**Note**
Harmonic and Percussion components of music: The aim of the Harmonic/Percussive separation is to decompose the original music signal to the harmonic (i.e. pitched instruments) and the percussive (non pitched instruments, percussion) parts of the signal. Such methods can be applied to audio mixing software, or can be adopted as preprocessing on other Music Information Retrieval Methods such as rhythm analysis or chord/tonality recognition.

The method is based on the assumption that harmonic components exhibit horizontal lines on the spectrogram while the percussive sounds are evident as vertical lines. By adopting Non-Linear image filters applied to the spectrogram in order to filter out these component, the proposed method is simple, intuitive and does not make any prior assumption of the genre,style or instrumentation of the target music. 
_Reference: [Click Here](http://mir.ilsp.gr/harmonic_percussive_separation.html)_

