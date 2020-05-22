import os
import database as db
from lmtd9 import LMTD
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss, label_ranking_loss
import pickle
import numpy as np
from seperation import seperation_genre
from keras.models import model_from_json
import evaluation

lmtd = LMTD()
LMTD_PATH = # where LMTD resnet data is saved 
features_path = os.path.join(LMTD_PATH, 'lmtd9_resnet152.pickle')
lmtd.load_precomp_features(features_file=features_path)
# derive testing and training data
x_train, x_train_len, y_train, train_ids = lmtd.get_split('train')
x_test,  x_test_len,  y_test,  test_ids  = lmtd.get_split('test')

num_of_classes = 9 # number of classes can be 4 or 9
    
# indicate each label correspond to which genre
if(num_of_classes==9):
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

ids_train, labels_train, genres_trains = db.get_split('train')
ids_test, labels_test, genres_test = db.get_split('test')

# model can be CNN1D or LSTM
model = 'CNN1D'

# derive data and label that have 4 main genre from 9 genre data
if (num_of_classes == 4):
    x_train, _ = seperation_genre(x_train, y_train)
    x_test,  _  = seperation_genre(x_test,  y_test)

# get high-level movies' visual features 
if(model=='CNN1D'):
    if (num_of_classes == 9):
        #open(where you saved 9 genre CNN1D visual only network)
        json_file = open('./visual_weight/model_mm_9_genre_1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./visual_weight/model_CNN1D_mm_9genre_1.h5")
    else:    
        #open(where you saved 4 genre CNN1D visual only network)
        json_file = open('./visual_weight/model_mm_4_genre_4.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./visual_weight/model_CNN1D_mm_4genre_4.h5")
    x_train_2 = loaded_model.predict(x_train)
    x_test_2 = loaded_model.predict(x_test)
else:
    if (num_of_classes == 9):
        #open(where you saved 9 genre LSTM visual only network)
        json_file = open('./visual_weight/LSTM_merge.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./visual_weight/LSTM_merge.h5")
    else:
        #open(where you saved 4 genre LSTM visual only network)
        json_file = open('./visual_weight/LSTM_merge_4genre.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./visual_weight/LSTM_merge_4genre.h5")
    x_train_2 = loaded_model.predict([x_train[:,9*1:9*2,:],x_train[:,9*2:9*3,:],x_train[:,9*3:9*4,:], x_train[:,9*4:9*5,:], x_train[:,9*5:9*6,:], x_train[:,9*6:9*7,:],x_train[:,9*7:9*8,:],x_train[:,9*8:9*9,:], x_train[:,9*9:9*10,:],x_train[:,9*10:9*11,:],x_train[:,9*11:9*12,:],x_train[:,9*12:9*13,:]])
    x_test_2 = loaded_model.predict([x_test[:,9*1:9*2,:],x_test[:,9*2:9*3,:],x_test[:,9*3:9*4,:], x_test[:,9*4:9*5,:], x_test[:,9*5:9*6,:], x_test[:,9*6:9*7,:],x_test[:,9*7:9*8,:],x_test[:,9*8:9*9,:], x_test[:,9*9:9*10,:],x_test[:,9*10:9*11,:],x_test[:,9*11:9*12,:],x_test[:,9*12:9*13,:]])
    
# import audio features 
train_data = np.load('./audio_features/train_data_2.npy')
test_data = np.load('./audio_features/test_data_2.npy')

if (num_of_classes == 4):
    train_data, y_train = seperation_genre(train_data, y_train)
    test_data,  y_test  = seperation_genre(test_data,  y_test)

# normalize audio features between 0 and 1
l1, l2 = len(y_train), len(y_test)
train_data = (train_data-np.amin(train_data,axis=1).reshape(l1,1))/(np.amax(train_data,axis=1).reshape(l1,1)-np.amin(train_data,axis=1).reshape(l1,1))
test_data = (test_data-np.amin(test_data,axis=1).reshape(l2,1))/(np.amax(test_data,axis=1).reshape(l2,1)-np.amin(test_data,axis=1).reshape(l2,1))

# concatenate audio and visual features 
train_data = np.concatenate((train_data,x_train_2),axis=1)
test_data = np.concatenate((test_data,x_test_2),axis=1)


y_pred_2 = np.zeros((len(test_data),num_of_classes))
y_pred = np.zeros((len(test_data),num_of_classes))
y_t_pred = np.zeros((len(train_data),num_of_classes))
y_train_pred = np.zeros((len(train_data),num_of_classes))
for i in range(num_of_classes):
    if (num_of_classes == 9):
        if(model=='CNN1D'):
            filename = 'svm_'+str(i)+'_mm_9genre_1Dconv_1.sav'
        else:
            filename = 'svm_'+str(i)+'_9genre_lstm_prob_merge.sav'
    else:
        if(model=='CNN1D'):
            filename = 'svm_'+str(i)+'_mm_4genre_1Dconv_02.sav'
        else:
            filename = 'svm_'+str(i)+'_4genre_lstm_prob_merge.sav'
    clf = pickle.load(open(filename, 'rb'))
    test_result = clf.predict_proba(test_data)
    train_result = clf.predict_proba(train_data)
    y_train_pred[:,i] = np.asarray(train_result[:,1])
    y_pred_2[:,i] = np.asarray(test_result[:,1])
    y_pred[:,i] = np.asarray(clf.predict(test_data))
    y_t_pred[:,i] = np.asarray(clf.predict(train_data))

# use training threshold
_, _ , threshold= evaluation.prauc(y_train, y_train_pred,False)
# test prediction
_, y_final, _ = evaluation.prauc(y_test, y_pred_2, True, threshold)   

# record final hamming loss
print('hamming_loss: ',hamming_loss(y_test,y_pred))

# genres considered based on number of classes 
if(num_of_classes==4):
    classes = ['action','drama','horor','romance']
else:
    classes = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama',
               'Horror', 'Romance', 'SciFi', 'Thriller']

# create confusion matrix for audio+visual classification 
y_test = y_test.astype(int)
first_matrix = np.zeros((num_of_classes,num_of_classes))
pred_matrix = np.zeros((num_of_classes,num_of_classes))
for i in range(num_of_classes):
    for j in range(num_of_classes):
        pos = (~np.logical_xor(y_test[:,i],y_test[:,j]))
        first_matrix[i,j] = np.sum(pos)
        pred_matrix[i,j] = np.sum(~np.logical_xor(y_test[pos,i],y_pred[pos,i]))
np.set_printoptions(precision=3)
fault = ((first_matrix-pred_matrix).T/np.diag(first_matrix)).T
plt.figure()
plt.imshow(fault, interpolation='nearest', vmin=0.0, vmax = 0.4)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

saveloc = './results/9genre_visual_audio_confusion.pdf'# where you want to save your confusion matrix
plt.savefig(saveloc, format='pdf', dpi=300, bbox_inches = 'tight')

# create confusion matrix for visual only classification 
_, _, thresh = evaluation.prauc(y_train, x_train_2, False)
_, y_pred_3, _ = evaluation.prauc(y_test, x_test_2, True, thresh)   
first_matrix = np.zeros((num_of_classes,num_of_classes))
pred_matrix = np.zeros((num_of_classes,num_of_classes))
for i in range(num_of_classes):
    for j in range(num_of_classes):
        pos = (~np.logical_xor(y_test[:,i],y_test[:,j]))
        first_matrix[i,j] = np.sum(pos)
        pred_matrix[i,j] = np.sum(~np.logical_xor(y_test[pos,i],y_pred_3[pos,i]))
np.set_printoptions(precision=3)
fault2 = ((first_matrix-pred_matrix).T/np.diag(first_matrix)).T
plt.figure()
plt.imshow(fault2, interpolation='nearest', vmin=0.0, vmax = 0.4)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)
saveloc = './results/9genre_visualonly_confusion.pdf'# where you want to save your confusion matrix
plt.savefig(saveloc, format='pdf', dpi=300, bbox_inches = 'tight')

