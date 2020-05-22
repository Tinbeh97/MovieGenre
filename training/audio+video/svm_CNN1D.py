import os
import database as db
from lmtd9 import LMTD
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
import pickle
from seperation import seperation_genre
import evaluation
import numpy as np
from keras.models import model_from_json
from sklearn import svm
from sklearn.model_selection import GridSearchCV

lmtd = LMTD()
LMTD_PATH = #where you saved LMTD dataset
audio_path = #where you saved audio features
features_path = os.path.join(LMTD_PATH, 'lmtd9_resnet152.pickle')
lmtd.load_precomp_features(features_file=features_path)
x_valid, x_valid_len, y_valid, valid_ids = lmtd.get_split('valid')
x_train, x_train_len, y_train, train_ids = lmtd.get_split('train')
x_test,  x_test_len,  y_test,  test_ids  = lmtd.get_split('test')

num_of_classes = 9 #number of genres can be 4 or 9 

if (num_of_classes == 4):
    x_valid, _ = seperation_genre(x_valid, y_valid)
    x_train, _ = seperation_genre(x_train, y_train)
    x_test,  _  = seperation_genre(x_test,  y_test)
    
#for indicating each label corresponds to which genre
if (num_of_classes == 9):
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
        
# which video features to use
if (num_of_classes == 9):
    json_file = open('model_mm_9_genre_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_CNN1D_mm_9genre_1.h5")
else:    
    json_file = open('model_mm_4_genre_4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_CNN1D_mm_4genre_4.h5")

x_valid_2 = loaded_model.predict(x_valid)
x_train_2 = loaded_model.predict(x_train)
x_test_2 = loaded_model.predict(x_test)

# load audio features
valid_data = np.load(audio_path+'valid_data_delta.npy')
train_data = np.load(audio_path+'train_data_delta.npy')
test_data = np.load(audio_path+'test_data_delta.npy')

if (num_of_classes==4):
    valid_data, y_valid = seperation_genre(valid_data, y_valid)
    train_data, y_train = seperation_genre(train_data, y_train)
    test_data,  y_test  = seperation_genre(test_data,  y_test)
l1, l2 = len(y_train), len(y_test)

#normalize acoustic features between 0 and 1 
train_data = (train_data-np.amin(train_data,axis=1).reshape(l1,1))/(np.amax(train_data,axis=1).reshape(l1,1)-np.amin(train_data,axis=1).reshape(l1,1))
test_data = (test_data-np.amin(test_data,axis=1).reshape(l2,1))/(np.amax(test_data,axis=1).reshape(l2,1)-np.amin(test_data,axis=1).reshape(l2,1))

#concatinate audio and video features
x_train_3 = np.concatenate((train_data,x_train_2),axis=1)
x_test_3 = np.concatenate((test_data,x_test_2),axis=1)

y_pred_2 = np.zeros((len(x_test),num_of_classes))
y_pred = np.zeros((len(x_test),num_of_classes))
y_train_pred = np.zeros((len(x_train),num_of_classes))
# SVM model 
for i in range(num_of_classes):
    C_range = np.logspace(-2,5,14)
    gamma_range = np.logspace(-5,0,14)
    param_grid = dict(gamma=gamma_range, C=C_range)
    svr = svm.SVC(kernel='rbf',decision_function_shape='ovo', probability=True)
    clf = GridSearchCV(svr, param_grid=param_grid)
    clf.fit(x_train_3, y_train[:,i])
    test_result = clf.predict_proba(x_test_3)
    if (num_of_classes == 9):
        filename = 'svm_'+str(i)+'_mm_9genre_1Dconv_01.sav'
    else:
        filename = 'svm_'+str(i)+'_mm_4genre_1Dconv_02.sav'   
    pickle.dump(clf, open(filename, 'wb'))
    print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
    train_result = clf.predict_proba(x_train)
    y_train_pred[:,i] = np.asarray(train_result[:,1])
    y_pred_2[:,i] = np.asarray(test_result[:,1])
    y_pred[:,i] = np.asarray(clf.predict(x_test))
    
_, _ , threshold = evaluation.prauc(y_train, y_train_pred,False)
#use training threshold
_, y_final,_ = evaluation.prauc(y_test, y_pred_2, True, threshold)   

#final loss
print('hamming_loss: ',hamming_loss(y_test,y_pred))
if(num_of_classes==4):
    classes = ['action','drama','horor','romance']
else:
    classes = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama',
               'Horror', 'Romance', 'SciFi', 'Thriller']

#confusion matrix
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)
first_matrix = np.zeros((num_of_classes,num_of_classes))
pred_matrix = np.zeros((num_of_classes,num_of_classes))
for i in range(num_of_classes):
    for j in range(num_of_classes):
        pos = (~np.logical_xor(y_test[:,i],y_test[:,j]))
        first_matrix[i,j] = np.sum(pos)
        pred_matrix[i,j] = np.sum(~np.logical_xor(y_test[pos,i],y_pred[pos,i]))
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
fault = (((first_matrix-pred_matrix)).T/np.diag(first_matrix)).T
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
