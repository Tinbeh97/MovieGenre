# Movie's genre classification

**Please refer to (paper) for detail explanation**

In this project, both acoustic and visual feature of the movie is used to classify the multi-label movie's genre. The network defined for genre classification based on frames is shown below. The final class is decided by feeding the fusion of audio and visual features into the SVM model.  
![Image of network](https://github.com/Tinbeh97/MovieGenre/blob/master/conv.png)

## Requirements

All codes are written for Python 2.7+ (https://www.python.org/) on Linux platform. 

The libraries that are needed: keras, librosa, sklearn, pickel, itertools, sqlite3, subprocess, os, matplotlib, sys, _pickle, and pysptk.

## Dataset

The dataset can be collected from the https://github.com/jwehrmann/lmtd/tree/master/lmtd9.

## Implementation

In all codes the dictionary or data location is commented and should be defined by the user.  

### training

#### visual based

If you want to extract Vgg or Resnet high-level features from videos go to [training/frames_features](./training/frames_features) folder and run high_level_video.py. The network (in this case vgg16) should be saved on Keras beforehand. Using GPU for running this code is highly recommended.

LSTM and CNN1D model that are defined on the paper are saved on [training/video](./training/video). You can implement each network by running lstm_model.py and CNN1D_model.py, respectively.

The error_epoch.py can be used to plot validation and training loss per epoch.

#### movie's audio + frames

For extracting audio feature go to [training/audio](./training/audio) folder. Run mp3maker.py for deriving background audio of trailers. Then run audio.py for extracting audio features. The location of data and storage should be determined by user. 

For detecting final genre based on both acoustic and frames features run svm_CNN1D.py in [training\audio+video](.\training\audio+video).

### testing


## Citation

If you find these codes usefull please cite following paper:



## Acknowledgments

