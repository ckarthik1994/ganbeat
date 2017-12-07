import numpy as np
from scipy import interpolate
import scipy.io.wavfile as wf

fmin = 0
fmax=0

lines = open('./giantsteps-tempo-dataset-master/splits/files.txt','r').readlines()
for line in lines:
    fil_path= "./giantsteps-tempo-dataset-master/audio/"+line[:-4]+"wav"
    dst = "./giantsteps-tempo-dataset-master/audio/processed/"+line[:-4]+"wav"
    NEW_SAMPLERATE = 11025
    rate, data = wf.read(fil_path)
    if rate != NEW_SAMPLERATE:
        duration = data.shape[0] / rate

        time_old  = np.linspace(0, duration, data.shape[0])
        time_new  = np.linspace(0, duration, int(data.shape[0] * NEW_SAMPLERATE / rate))

        interpolator = interpolate.interp1d(time_old, data.T)
        new_audio = interpolator(time_new).T

        wf.write(dst, NEW_SAMPLERATE, np.round(new_audio).astype(data.dtype))
        
fil_path= "./giantsteps-tempo-dataset-master/audio/processed/1479462.LOFI.wav"
rate, data = wf.read(fil_path)
print(data.shape[0])
f, t, Zxx = signal.stft(data, rate)
print(Zxx.shape)
song = signal.istft(Zxx,rate)
wav_data = song[1]
wf.write("Sample2.wav",rate,wav_data)
