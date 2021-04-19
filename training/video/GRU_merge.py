import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Concatenate, Input, GRU, Embedding, BatchNormalization, MaxPooling1D
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
from lmtd9 import LMTD
import database as db
import evaluation
import numpy as np
from seperation import seperation_genre

number_of_frames = 9 
input_dim = 2048
nb_classes = 4
max_epochs = 100
lmtd = LMTD()

LMTD_PATH = #the dataset location

features_path = os.path.join(LMTD_PATH, 'lmtd9_resnet152.pickle')
lmtd.load_precomp_features(features_file=features_path)
x_valid, x_valid_len, y_valid, valid_ids = lmtd.get_split('valid')
x_train, x_train_len, y_train, train_ids = lmtd.get_split('train')
x_test,  x_test_len,  y_test,  test_ids  = lmtd.get_split('test')

if (nb_classes == 9):
    def print_result(result):
        for k, v in result.items():
            try:
                print ('{:<15s}'.format(lmtd.genres[k][2:]))
            except IndexError:
                print ('{:<15s}'.format(k.title()))
            print ('{:5.4f}'.format(v))
else:
    def print_result(result):
        genres = ['action','drama','horor','romance']
        for k, v in result.items():
            try:
                print ('{:<15s}'.format(genres[k]))
            except TypeError:
                print ('{:<15s}'.format(k.title()))
            print ('{:5.4f}'.format(v)) 
            
merge = False

#GRU network parallel
In_merge = []
x_merge = []
for i in range(12):
    input1 = Input(shape=(number_of_frames, input_dim))
    x1 = GRU(120, return_sequences=True, stateful=False)(input1)
    x1 = MaxPooling1D(pool_size=3)(x1)
    x1 = (GRU(64, return_sequences=False, stateful=False))(x1)
    x1 = (Dense(nb_classes, activation='relu'))(x1)
    In_merge.append(input1)
    x_merge.append(x1)
added = Concatenate()([x_merge[0],x_merge[1],x_merge[2],x_merge[3],x_merge[4],x_merge[5],
                   x_merge[6],x_merge[7],x_merge[8],x_merge[9],x_merge[10],x_merge[11]]) 
out = (Dense(nb_classes, activation='sigmoid'))(added)

#GRU network sequential
In = Input(shape=(240, input_dim))
x = GRU(120, return_sequences=True, stateful=False)(In)
x = MaxPooling1D(pool_size=3)(x)
x = (GRU(64, return_sequences=False, stateful=False))(x)
x = (Dense(nb_classes, activation='sigmoid'))(x)

if(merge):
    model = Model(inputs = [In_merge[0],In_merge[1],In_merge[2],In_merge[3],In_merge[4],In_merge[5],
                   In_merge[6],In_merge[7],In_merge[8],In_merge[9],In_merge[10],In_merge[11]], outputs = out)
    model_name = 'GRU_merge'
else:
    model = Model(inputs = In,outputs = x)
    model_name = 'GRU'
if(nb_classes==4):
    x_valid, y_valid = seperation_genre(x_valid, y_valid)
    x_train, y_train = seperation_genre(x_train, y_train)
    x_test,  y_test  = seperation_genre(x_test,  y_test)
    model_name = model_name + '_4genre'
    
print(model.summary())
"""
#saving model summary

from contextlib import redirect_stdout.0`
with open('/Users/tina/Downloads/model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
"""

model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy')
history = History()
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
mcp_save = ModelCheckpoint(model_name+'.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

if(merge):          
    model.fit([x_train[:,9*1:9*2,:],x_train[:,9*2:9*3,:],x_train[:,9*3:9*4,:], x_train[:,9*4:9*5,:], x_train[:,9*5:9*6,:], x_train[:,9*6:9*7,:],x_train[:,9*7:9*8,:],x_train[:,9*8:9*9,:], x_train[:,9*9:9*10,:],x_train[:,9*10:9*11,:],x_train[:,9*11:9*12,:],x_train[:,9*12:9*13,:]],
              y_train,
              validation_data=([x_valid[:,9*1:9*2,:],x_valid[:,9*2:9*3,:],x_valid[:,9*3:9*4,:], x_valid[:,9*4:9*5,:], x_valid[:,9*5:9*6,:], x_valid[:,9*6:9*7,:],x_valid[:,9*7:9*8,:],x_valid[:,9*8:9*9,:], x_valid[:,9*9:9*10,:],x_valid[:,9*10:9*11,:],x_valid[:,9*11:9*12,:],x_valid[:,9*12:9*13,:]], 
                               y_valid), 
              epochs=max_epochs,
              batch_size=32,
              callbacks=[early_stopping, mcp_save, history])
    y_pred_train = model.predict([x_train[:,9*1:9*2,:],x_train[:,9*2:9*3,:],x_train[:,9*3:9*4,:], x_train[:,9*4:9*5,:], x_train[:,9*5:9*6,:], x_train[:,9*6:9*7,:],x_train[:,9*7:9*8,:],x_train[:,9*8:9*9,:], x_train[:,9*9:9*10,:],x_train[:,9*10:9*11,:],x_train[:,9*11:9*12,:],x_train[:,9*12:9*13,:]])    
else:
    model.fit(x_train,
          y_train,
          validation_data=(x_valid, 
                           y_valid), 
          epochs=max_epochs,
          batch_size=32,
          callbacks=[early_stopping, mcp_save, history])
    y_pred_train = model.predict(x_train)    
          
_, _, thresh = evaluation.prauc(y_train, y_pred_train, False)
model_json = model.to_json()

np.save(model_name+'_val_loss',history.history['val_loss'])
np.save(model_name+'_loss',history.history['loss'])
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
np.save(model_name, thresh)

if(merge):
    score = model.evaluate([x_test[:,9*1:9*2,:],x_test[:,9*2:9*3,:],x_test[:,9*3:9*4,:], x_test[:,9*4:9*5,:], x_test[:,9*5:9*6,:], x_test[:,9*6:9*7,:],x_test[:,9*7:9*8,:],x_test[:,9*8:9*9,:], x_test[:,9*9:9*10,:],x_test[:,9*10:9*11,:],x_test[:,9*11:9*12,:],x_test[:,9*12:9*13,:]],
                           y_test,batch_size=32)
    y_pred = model.predict([x_test[:,9*1:9*2,:],x_test[:,9*2:9*3,:],x_test[:,9*3:9*4,:], x_test[:,9*4:9*5,:], x_test[:,9*5:9*6,:], x_test[:,9*6:9*7,:],x_test[:,9*7:9*8,:],x_test[:,9*8:9*9,:], x_test[:,9*9:9*10,:],x_test[:,9*10:9*11,:],x_test[:,9*11:9*12,:],x_test[:,9*12:9*13,:]])    
else:
    score = model.evaluate(x_test,y_test,batch_size=32)
    y_pred = model.predict(x_test)
    
result, y_final, _ = evaluation.prauc(y_test, y_pred, True, thresh)   
print('test_error: ',score)
print_result(result)
print('')
if(nb_classes==4):
    classes = ['action','drama','horor','romance']
else:
    classes = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama',
               'Horror', 'Romance', 'SciFi', 'Thriller']
