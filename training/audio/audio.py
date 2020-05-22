
import os
import librosa
import pysptk
import numpy as np
import database as db #collecting training, testing, and validation data 

ids_train, labels_train, genres_trains = db.get_split('train')
ids_valid, labels_valid, genres_valid = db.get_split('valid')
ids_test, labels_test, genres_test = db.get_split('test')

audio_path = # audio files' location 
audio_ext = '.mp3'
os.chdir(audio_path)
files = os.listdir(audio_path)
files.sort() #sort audio by there number
train_data = []
valid_data = []
test_data = []
nframe_mfcc = 5
i = 0
for f in files:
    if not f.endswith(audio_ext):
        continue
    y, sr = librosa.load(audio_path+f, offset=5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    n = mfccs.shape[1]//nframe_mfcc
    Mfcc = mfccs[:,0:int(n*nframe_mfcc)]
    Mfcc = Mfcc.reshape(13,n,nframe_mfcc)
    mfcc_mean = Mfcc.mean(axis =1).reshape(13*nframe_mfcc)#mfcc features
    #mfcc_dev = Mfcc.var(axis = 1).reshape(13*nframe_mfcc)
    mfcc_delta = librosa.feature.delta(Mfcc, width=(n//2)*2-1 , axis=1, order=1)
    mfcc_delta_mean = mfcc_delta.mean(axis =1).reshape(13*nframe_mfcc)#mfcc delta features
    lpc = pysptk.sptk.lpc(y, order=9) #LPC feature
    audio_feature = np.concatenate((np.concatenate((mfcc_delta_mean,mfcc_mean)),lpc))
    f2 = f.replace(audio_ext,'')
    if(sum(ids_train==f2)):
        train_data.append(audio_feature)
    elif(sum(ids_valid==f2)):
        valid_data.append(audio_feature)
    elif(sum(ids_test==f2)):
        test_data.append(audio_feature)
    else:
        print(f)
    i = i+1
    if(i%500==0):
        print(i) #tracking number
#saving acoustic features
np.save('valid_data_delta',valid_data) 
np.save('train_data_delta',train_data)
np.save('test_data_delta',test_data)
