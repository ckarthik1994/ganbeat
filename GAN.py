from glob import glob

# Fetching Paths
main_path = './giantsteps-tempo-dataset-master/audio/processed/'
paths = glob(main_path+'*.wav')

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from scipy import signal
import scipy.io.wavfile as wf
import numpy as np

# fetching data
def read_wav_from_path(path):
    rate, data = wf.read(path)
    f, t, Zxx = signal.stft(data, rate)
    Zxx = Zxx.transpose()
    #X = Zxx[:-1]
    #y = Zxx[1:]
    return Zxx

from multiprocessing import Pool

p = Pool(8)
def read_set(paths):
    resultSet = p.map(read_wav_from_path, paths)
    return resultSet
    
#creating test and train splits for generator
trainPaths = paths[:int(0.7*len(paths))]
testPaths = paths[int(0.7*len(paths)):]
trainData = read_set(trainPaths)
#train_x = trainData[:-1]
testData = read_set(testPaths)

def get_data(data):
    _set = np.zeros((len(data), 10342, 129))
    for i in range(len(data)):
        try:
            _set[i,:, :] = data[i]
        except:
            continue
    return _set
    
test = get_data(testData)
testx = test[:, :-1, :]
testy = test[:, 1:, :]

train = get_data(trainData)
trainx = train[:, :-1, :]
trainy = train[:, 1:, :]

# Generator Model
inp = Input(shape=trainx[0].shape)
my_lstm = LSTM(512,return_sequences=True)(inp)
out = TimeDistributed(Dense(129))(my_lstm)
model = Model(input=inp,output=out)
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(trainx, trainy, epochs=3, batch_size=4)


y_ = model.predict(testx)

# Regenerating Songs from predicted Stft
i=0
for y in y_:
    song = signal.istft(y.transpose(),11025)
    song = np.asarray(song)
    wav_data = song[1]
    wf.write(testPaths[i][:-5]+'_test.wav',11025,wav_data)

trainx,trainy = [],[]
#making train_x,train_y for Discriminator
for x in testx:
  trainx.append(x)
  trainy.append([1])
for y in y_:
  trainx.append(y)
  trainy.append([0])

#generate Train and test splits for Discriminatior
trainx,trainy=zip(*(random.shuffle(zip(trainx,trainy))))
testx = trainx[int((0.7)*len(trainx)):]
trainx = trainx[:int((0.7)*len(trainx))]
testy = trainy[int((0.7)*len(triany)):]
train y = trainy[:int((0.7)*len(triany))]

# Discriminator Model
inp = Input(shape=y_[0].shape)
my_lstm = LSTM(512,return_sequences=False)(inp)
out = Dense(1,activation=sigmoid)(my_lstm)
model = Model(input=inp,output=out)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(trainx, trainy, epochs=3, batch_size=4)
    i+=1
loss=model.evaluate(testx,testy)
print("Dicriminator test loss is %d"%(loss))
