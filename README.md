# Movie's genre classification

**Please refer to (paper) for detail explanation**

In this project, both acoustic and visual feature of the movie is used to classify the multi-label movie's genre. The network defined for genre classification based on frames is shown below. The final class is decided by feeding the fusion of audio and visual features into the SVM model.  
![Image of network](https://github.com/Tinbeh97/MovieGenre/blob/master/conv.png)

## Requirements

All codes are written for Python 2.7+ (https://www.python.org/) on Linux platform. 

The libraries that are needed: keras, librosa, sklearn, pickel, itertools, sqlite3, subprocess, os, matplotlib, and pysptk.

## Dataset

The dataset can be collected from the https://github.com/jwehrmann/lmtd/tree/master/lmtd9.

## Implementation

In all codes the dictionary or data location is commented and should be defined by the user.  

### training

#### visual based

#### movie's audio + frames

For extracting audio feature go to /training/audio folder. Run mp3maker.py for deriving background audio of trailers. Then run audio.py for extracting audio features. The location of data and storage should be determined by user. 



### testing


## Citation

If you find these codes usefull please cite following paper:



## Acknowledgments

