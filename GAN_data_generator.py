
# coding: utf-8

# In[3]:


import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

fmin = 0
fmax=0

lines = open('./giantsteps-tempo-dataset-master/splits/files.txt','r').readlines()
for line in lines:
    fil_path= "./giantsteps-tempo-dataset-master/audio/"+line[:-4]+"wav"
    rate, data = wf.read(fil_path)
    amp = 20 * np.sqrt(20)
    f, t, Zxx = signal.stft(data, rate)
    print(line[:-5],len(f))
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


# In[11]:


fil_path= "./giantsteps-tempo-dataset-master/audio/1479462.LOFI.wav"
rate, data = wf.read(fil_path)
f, t, Zxx = signal.stft(data, rate)
song = signal.istft(Zxx,rate)
wav_data = song[1]
wf.write("Sample2.wav",rate,wav_data)

