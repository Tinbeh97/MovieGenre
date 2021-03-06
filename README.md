# Movie's genre classification

**Please refer to (paper) for detail explanation**

In this project, both acoustic and visual feature of the movie is used to classify the multi-label movie's genre. The network defined for genre classification based on frames is shown below. The final class is decided by feeding the fusion of audio and visual features into the SVM model.  
![Image of network](https://github.com/Tinbeh97/MovieGenre/blob/master/conv.png)

These codes are written for both 9 and 4 genre dataset. Action, adventure, comedy, crime, drama, horror, romance, sci-fi, and thriller genres are considered for 9 genre data. In 4 genre classification, action, drama, horror, and romance are separated from 9 class dataset.

## Requirements

All codes are written for Python 3 (https://www.python.org/) on Linux platform. 

The packages that are needed: keras, librosa, sklearn, pickel, itertools, sqlite3, subprocess, os, matplotlib, sys, _pickle, and pysptk.

## Dataset

The dataset can be collected from the https://github.com/jwehrmann/lmtd/tree/master/lmtd9.

## Implementation

In all codes the dictionary or data location is commented and should be defined by the user.  

### Training

#### Visual based

If you want to extract Vgg or Resnet high-level features from videos go to [training/frames_features](./training/frames_features) folder and run high_level_video.py. The network (in this case vgg16) should be saved on Keras beforehand. Using GPU for running this code is highly recommended.

LSTM and CNN1D model that are defined on the paper are saved on [training/video](./training/video). You can implement each network by running lstm_model.py and CNN1D_model.py, respectively. The new model which is the sequential and parallel GRU network is added to this file. The model is implemented in GRU_merge.py and codes are based on Tensorflow 2.3.1.

The error_epoch.py can be used to plot validation and training loss per epoch.

#### Movie's audio + frames

For extracting audio feature go to [training/audio](./training/audio) folder. Run mp3maker.py for deriving background audio of trailers. Then run audio.py for extracting audio features. The location of data and storage should be determined by user. 

For detecting final genre based on both acoustic and frames features run svm_CNN1D.py in [training/audio+video](./training/audio+video).

### Testing

For testing the method you should put your audio features in [testing/audio_features](./testing/audio_features) folder. Also, [testing/visual_weight](./testing/visual_weight) and [testing/svm_result](./testing/svm_result) folders are where the visual model weights and audio-visual SVM network is saved. You can also change directory of them in testing.py if prefered to.

Run [testing.py](./testing) to test the model performance. You can decide whether you want 4 or 9 genre classification by changing the num_of_classes value.

## Citation

If you find this repository useful, please consider citing following paper:



## Acknowledgments

LMTD9 data and codes to derive testing, training, and validation data is used from https://github.com/jwehrmann/lmtd/tree/master/lmtd9.
