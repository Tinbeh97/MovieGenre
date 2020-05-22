import tensorflow as tf
from keras import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Concatenate, Input, LSTM, Embedding, BatchNormalization, MaxPooling1D
from keras.layers.convolutional import Convolution2D, Convolution3D, MaxPooling2D
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
from lmtd9 import LMTD
import database as db
import evaluation
import numpy as np
from seperation import seperation_genre
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from sklearn.metrics import hamming_loss, label_ranking_loss

number_of_frames = 9
input_dim = 2048
number_of_classes = 4 #number of genres (4 or 9)
nb_classes = 4
max_epochs = 100
lmtd = LMTD()

LMTD_PATH = #where LMTD dataset is saved

features_path = os.path.join(LMTD_PATH, 'lmtd9_resnet152.pickle')
lmtd.load_precomp_features(features_file=features_path)
x_valid, x_valid_len, y_valid, valid_ids = lmtd.get_split('valid')
x_train, x_train_len, y_train, train_ids = lmtd.get_split('train')
x_test,  x_test_len,  y_test,  test_ids  = lmtd.get_split('test')

#LSTM network
input1 = Input(shape=(number_of_frames, input_dim))
x1 = LSTM(64, return_sequences=True, stateful=False)(input1)
x1 = MaxPooling1D()(x1)
x1 = (LSTM(64, return_sequences=True, stateful=False))(x1)
x1 = MaxPooling1D()(x1)
x1 = (Flatten())(x1)
x1 = (Dropout(0.5))(x1)
x1 = (Dense(32, activation='relu'))(x1)
x1 = (Dense(number_of_classes, activation='relu'))(x1)
input2 = Input(shape=(number_of_frames, input_dim))
x2 = LSTM(64, return_sequences=True, stateful=False)(input2)
x2 = MaxPooling1D()(x2)
x2 = (LSTM(64, return_sequences=True, stateful=False))(x2)
x2 = MaxPooling1D()(x2)
x2 = (Flatten())(x2)
x2 = (Dropout(0.5))(x2)
x2 = (Dense(32, activation='relu'))(x2)
x2 = (Dense(number_of_classes, activation='relu'))(x2)
input3 = Input(shape=(number_of_frames, input_dim))
x3 = LSTM(64, return_sequences=True, stateful=False)(input3)
x3 = MaxPooling1D()(x3)
x3 = (LSTM(64, return_sequences=True, stateful=False))(x3)
x3 = MaxPooling1D()(x3)
x3 = (Flatten())(x3)
x3 = (Dropout(0.5))(x3)
x3 = (Dense(32, activation='relu'))(x3)
x3 = (Dense(number_of_classes, activation='relu'))(x3)
input4 = Input(shape=(number_of_frames, input_dim))
x4 = LSTM(64, return_sequences=True, stateful=False)(input4)
x4 = MaxPooling1D()(x4)
x4 = (LSTM(64, return_sequences=True, stateful=False))(x4)
x4 = MaxPooling1D()(x4)
x4 = (Flatten())(x4)
x4 = (Dropout(0.5))(x4)
x4 = (Dense(32, activation='relu'))(x4)
x4 = (Dense(number_of_classes, activation='relu'))(x4)
input5 = Input(shape=(number_of_frames, input_dim))
x5 = LSTM(64, return_sequences=True, stateful=False)(input5)
x5 = MaxPooling1D()(x5)
x5 = (LSTM(64, return_sequences=True, stateful=False))(x5)
x5 = MaxPooling1D()(x5)
x5 = (Flatten())(x5)
x5 = (Dropout(0.5))(x5)
x5 = (Dense(32, activation='relu'))(x5)
x5 = (Dense(number_of_classes, activation='relu'))(x5)
input6 = Input(shape=(number_of_frames, input_dim))
x6 = LSTM(64, return_sequences=True, stateful=False)(input6)
x6 = MaxPooling1D()(x6)
x6 = (LSTM(64, return_sequences=True, stateful=False))(x6)
x6 = MaxPooling1D()(x6)
x6 = (Flatten())(x6)
x6 = (Dropout(0.5))(x6)
x6 = (Dense(32, activation='relu'))(x6)
x6 = (Dense(number_of_classes, activation='relu'))(x6)
input7 = Input(shape=(number_of_frames, input_dim))
x7 = LSTM(64, return_sequences=True, stateful=False)(input7)
x7 = MaxPooling1D()(x7)
x7 = (LSTM(64, return_sequences=True, stateful=False))(x7)
x7 = MaxPooling1D()(x7)
x7 = (Flatten())(x7)
x7 = (Dropout(0.5))(x7)
x7 = (Dense(32, activation='relu'))(x7)
x7 = (Dense(number_of_classes, activation='relu'))(x7)
input8 = Input(shape=(number_of_frames, input_dim))
x8 = LSTM(64, return_sequences=True, stateful=False)(input8)
x8 = MaxPooling1D()(x8)
x8 = (LSTM(64, return_sequences=True, stateful=False))(x8)
x8 = MaxPooling1D()(x8)
x8 = (Flatten())(x8)
x8 = (Dropout(0.5))(x8)
x8 = (Dense(32, activation='relu'))(x8)
x8 = (Dense(number_of_classes, activation='relu'))(x8)
input9 = Input(shape=(number_of_frames, input_dim))
x9 = LSTM(64, return_sequences=True, stateful=False)(input9)
x9 = MaxPooling1D()(x9)
x9 = (LSTM(64, return_sequences=True, stateful=False))(x9)
x9 = MaxPooling1D()(x9)
x9 = (Flatten())(x9)
x9 = (Dropout(0.5))(x9)
x9 = (Dense(32, activation='relu'))(x9)
x9 = (Dense(number_of_classes, activation='relu'))(x9)
input10 = Input(shape=(number_of_frames, input_dim))
x10 = LSTM(64, return_sequences=True, stateful=False)(input10)
x10 = MaxPooling1D()(x10)
x10 = (LSTM(64, return_sequences=True, stateful=False))(x10)
x10 = MaxPooling1D()(x10)
x10 = (Flatten())(x10)
x10 = (Dropout(0.5))(x10)
x10 = (Dense(32, activation='relu'))(x10)
x10 = (Dense(number_of_classes, activation='relu'))(x10)
input11 = Input(shape=(number_of_frames, input_dim))
x11 = LSTM(64, return_sequences=True, stateful=False)(input11)
x11 = MaxPooling1D()(x11)
x11 = (LSTM(64, return_sequences=True, stateful=False))(x11)
x11 = MaxPooling1D()(x11)
x11 = (Flatten())(x11)
x11 = (Dropout(0.5))(x11)
x11 = (Dense(32, activation='relu'))(x11)
x11 = (Dense(number_of_classes, activation='relu'))(x11)
input12 = Input(shape=(number_of_frames, input_dim))
x12 = LSTM(64, return_sequences=True, stateful=False)(input12)
x12 = MaxPooling1D()(x12)
x12 = (LSTM(64, return_sequences=True, stateful=False))(x12)
x12 = MaxPooling1D()(x12)
x12 = (Flatten())(x12)
x12 = (Dropout(0.5))(x12)
x12 = (Dense(32, activation='relu'))(x12)
x12 = (Dense(number_of_classes, activation='relu'))(x12)
added = Concatenate()([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]) 
out = (Dense(number_of_classes, activation='sigmoid'))(added)
model = Model(inputs = [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12], outputs = out)
print(model.summary())

#for indicating each label corresponds to which genre
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

model.compile(optimizer = Adam(lr=0.00001), loss = 'binary_crossentropy')
history = History()
#stop simulation if validation loss still decreasing after 5 iteration
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#save the best validation loss's network weights for 5 consecutive iteration
if(nb_classes ==9):
    mcp_save = ModelCheckpoint('LSTM_merge.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
else:
    x_valid, y_valid = seperation_genre(x_valid, y_valid)
    x_train, y_train = seperation_genre(x_train, y_train)
    x_test,  y_test  = seperation_genre(x_test,  y_test)
    mcp_save = ModelCheckpoint('LSTM_merge_4genre.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
model.fit([x_train[:,9*1:9*2,:],x_train[:,9*2:9*3,:],x_train[:,9*3:9*4,:], x_train[:,9*4:9*5,:], x_train[:,9*5:9*6,:], x_train[:,9*6:9*7,:],x_train[:,9*7:9*8,:],x_train[:,9*8:9*9,:], x_train[:,9*9:9*10,:],x_train[:,9*10:9*11,:],x_train[:,9*11:9*12,:],x_train[:,9*12:9*13,:]],
          y_train,
          validation_data=([x_valid[:,9*1:9*2,:],x_valid[:,9*2:9*3,:],x_valid[:,9*3:9*4,:], x_valid[:,9*4:9*5,:], x_valid[:,9*5:9*6,:], x_valid[:,9*6:9*7,:],x_valid[:,9*7:9*8,:],x_valid[:,9*8:9*9,:], x_valid[:,9*9:9*10,:],x_valid[:,9*10:9*11,:],x_valid[:,9*11:9*12,:],x_valid[:,9*12:9*13,:]], 
                           y_valid), 
          epochs=max_epochs,
          batch_size=32,
          callbacks=[early_stopping, mcp_save, history])
          
y_pred_train = model.predict([x_train[:,9*1:9*2,:],x_train[:,9*2:9*3,:],x_train[:,9*3:9*4,:], x_train[:,9*4:9*5,:], x_train[:,9*5:9*6,:], x_train[:,9*6:9*7,:],x_train[:,9*7:9*8,:],x_train[:,9*8:9*9,:], x_train[:,9*9:9*10,:],x_train[:,9*10:9*11,:],x_train[:,9*11:9*12,:],x_train[:,9*12:9*13,:]])    
_, _, thresh = evaluation.prauc(y_train, y_pred_train, False)

#save model loss and weights
model_json = model.to_json()
if(nb_classes == 9):
    np.save('LSTM_merge_val_loss',history.history['val_loss'])
    np.save('LSTM_merge_loss',history.history['loss'])
    with open("LSTM_merge.json", "w") as json_file:
        json_file.write(model_json)
    np.save('LSTM_merge', thresh)
else:
    np.save('LSTM_merge_val_loss_4genre',history.history['val_loss'])
    np.save('LSTM_merge_loss_4genre',history.history['loss'])
    with open("LSTM_merge_4genre.json", "w") as json_file:
        json_file.write(model_json)
    np.save('LSTM_merge_4genre', thresh)

score = model.evaluate([x_test[:,9*1:9*2,:],x_test[:,9*2:9*3,:],x_test[:,9*3:9*4,:], x_test[:,9*4:9*5,:], x_test[:,9*5:9*6,:], x_test[:,9*6:9*7,:],x_test[:,9*7:9*8,:],x_test[:,9*8:9*9,:], x_test[:,9*9:9*10,:],x_test[:,9*10:9*11,:],x_test[:,9*11:9*12,:],x_test[:,9*12:9*13,:]],
                       y_test,batch_size=32)
print('test_error: ',score)
y_pred = model.predict([x_test[:,9*1:9*2,:],x_test[:,9*2:9*3,:],x_test[:,9*3:9*4,:], x_test[:,9*4:9*5,:], x_test[:,9*5:9*6,:], x_test[:,9*6:9*7,:],x_test[:,9*7:9*8,:],x_test[:,9*8:9*9,:], x_test[:,9*9:9*10,:],x_test[:,9*10:9*11,:],x_test[:,9*11:9*12,:],x_test[:,9*12:9*13,:]])    
_, y_final, _ = evaluation.prauc(y_test, y_pred, True, thresh)   

if(nb_classes==4):
    classes = ['action','drama','horor','romance']
else:
    classes = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama',
               'Horror', 'Romance', 'SciFi', 'Thriller']

#report hamming loss
print('hamming_loss: ',hamming_loss(y_test,y_final))

y_test = y_test.astype(int)
y_final = y_final.astype(int)

#confusion matrix 

first_matrix = np.zeros((nb_classes,nb_classes))
pred_matrix = np.zeros((nb_classes,nb_classes))
for i in range(nb_classes):
    for j in range(nb_classes):
        pos = (~np.logical_xor(y_test[:,i],y_test[:,j]))
        first_matrix[i,j] = np.sum(pos)
        pred_matrix[i,j] = np.sum(~np.logical_xor(y_test[pos,i],y_final[pos,i]))
plt.imshow(first_matrix, interpolation='nearest')
plt.title('data_distribution')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.figure()
plt.imshow(pred_matrix, interpolation='nearest')
plt.title('data_correct_prediction')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

np.set_printoptions(precision=3)
fault = ((first_matrix-pred_matrix).T/np.diag(first_matrix)).T
plt.figure()
plt.imshow(fault, interpolation='nearest')
plt.title('data_fault_prediction')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

np.set_printoptions(precision=3)
fault = (first_matrix-pred_matrix)/first_matrix
plt.figure()
plt.imshow(fault, interpolation='nearest')
plt.title('data_fault_prediction')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)