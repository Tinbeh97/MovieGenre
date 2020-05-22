import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
from lmtd9 import LMTD
import database as db
import evaluation
from keras import Model
from keras.layers import Input, Convolution1D, GlobalMaxPooling1D, merge, Dense, Dropout
from keras.layers import BatchNormalization, Activation, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.optimizers import Adam
from sklearn.metrics import hamming_loss, label_ranking_loss
import numpy as np
from seperation import seperation_genre

time_steps = 240
nb_features = 2048                                          
nb_classes = 4 #determine number of genres
conv_filters = 384
dropout = 0.5
max_epochs = 100
lmtd = LMTD()
LMTD_PATH = #Where LMTD dataset is saved
features_path = os.path.join(LMTD_PATH, 'lmtd9_resnet152.pickle')
lmtd.load_precomp_features(features_file=features_path)
x_valid, x_valid_len, y_valid, valid_ids = lmtd.get_split('valid')
x_train, x_train_len, y_train, train_ids = lmtd.get_split('train')
x_test,  x_test_len,  y_test,  test_ids  = lmtd.get_split('test')
if (nb_classes == 4):
    x_valid, y_valid = seperation_genre(x_valid, y_valid)
    x_train, y_train = seperation_genre(x_train, y_train)
    x_test,  y_test  = seperation_genre(x_test,  y_test)

#1D convolutional model 
inputs = Input(shape=(time_steps, nb_features))
x = BatchNormalization()(inputs)
x = Convolution1D(conv_filters, kernel_size=3)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=7)(x)
x = Convolution1D(128, kernel_size=3)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dropout(dropout)(x)
x = Dense(100,activation='relu')(x)
x = Dropout(dropout)(x)
out = Dense(nb_classes, activation='sigmoid')(x)
model = Model(inputs, out)
print( model.summary())

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

model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy')
history = History()
#stop simulation if validation loss still decreasing after 5 iteration
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#save the best validation loss's network weights for 5 consecutive iteration
mcp_save = ModelCheckpoint('model_mm_9_genre_1.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
model.fit(x_train, y_train,
          validation_data=(x_valid, y_valid), 
          epochs=max_epochs,
          batch_size=32,
          callbacks=[early_stopping, mcp_save, history])

print(history.history.keys())
np.save('model_mm_9_genre_1_val_loss',history.history['val_loss'])
np.save('model_mm_9_genre_1_loss',history.history['loss'])
y_pred_val = model.predict(x_valid)    
y_pred_train = model.predict(x_train)    

_, _, thresh = evaluation.prauc(y_train, y_pred_train, False)
model_json = model.to_json()
with open("model_mm_9_genre_1.json", "w") as json_file:
    json_file.write(model_json)
# save traing threshold
np.save('threshold_CNN1D_mm_9genre_1', thresh)

score = model.evaluate(x_test,y_test,batch_size=32)
print('test_error: ',score)
y_pred = model.predict(x_test)    
_, y_final, _ = evaluation.prauc(y_test, y_pred, True, thresh)   

if(nb_classes==4):
    classes = ['action','drama','horor','romance']
else:
    classes = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama',
               'Horror', 'Romance', 'SciFi', 'Thriller']

#final hamming loss
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