import numpy as np
from config.global_parameters import default_model_name
from model_utils import features_batch
import database as db
from _pickle import dump


def get_frames(videoPath, start_time=5000, end_time=120000, time_step=2000):
    print ("Getting frames for ",videoPath)
    try:
        cap = cv2.VideoCapture(videoPath)
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if(t<=120000):
            time_step = int(1000*(t-2000)/120000)
            end_time = start_time + time_step*115
        for k in range(start_time, end_time+1, time_step):
            cap.set(0, k)
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame,(frameWidth, frameHeight))
                yield frame
    except Exception as e:
        print (e)
        return
    
def dump_pkl(data, pklName, verbose = True):
    if verbose:
        print( "Dumping data into",pklName)
    dump(data, open(pklName+'.p', 'wb'))
    
def gather_data(model_name=default_model_name):
    """Driver function to collect frame features for a genre"""
    ids_train, labels_train, genres_trains = db.get_split('train')
    ids_valid, labels_valid, genres_valid = db.get_split('valid')
    ids_test, labels_test, genres_test = db.get_split('test')
    ids_train = [int(x) for x in ids_train]
    ids_valid = [int(x) for x in ids_valid]
    ids_test = [int(x) for x in ids_test]
    train_data = []
    valid_data = []
    test_data = []
    
    VIDEOS_PATH = # where raw videos are saved
    VIDEOS_EXTENSION = '.mp4'  # for example
    os.chdir(VIDEOS_PATH)
    videoPaths = os.listdir(VIDEOS_PATH)
    videoPaths.sort()
    i=0
    outPath = #where to save high-level frames' features
    
    for videoPath in videoPaths:
        if not videoPath.endswith(VIDEOS_EXTENSION):
            continue
        #print (videoPath,":")
        path=os.path.join(VIDEOS_PATH,videoPath)
        frames =list(get_frames(path, time_step=1000))
        print (len(frames))
        if len(frames)==0:
            print( "corrupt.")
            continue
        videoFeatures = features_batch(frames, model_name)
        print (videoFeatures.shape)
        f2 = videoPath.replace(VIDEOS_EXTENSION,'')
        print(f2)
        f2 = int(f2)
        if(sum(np.equal(ids_train,f2))):
            train_data.append(videoFeatures)
        elif(sum(np.equal(ids_valid,f2))):
            valid_data.append(videoFeatures)
        elif(sum(np.equal(ids_test,f2))):
            test_data.append(videoFeatures)
        else:
            print(f2)
        i = i+1
        if(i%500==0): #use seperate files to avoid taking to much storage
            dump_pkl(train_data, outPath+'/train_'+str(i//500))
            dump_pkl(valid_data, outPath+'/valid_'+str(i//500))
            dump_pkl(test_data, outPath+'/test_'+str(i//500))
            train_data = []
            valid_data = []
            test_data = []
        if(i%100==0): #tracking number 
            print(i)

    
    #save features
    dump_pkl(train_data, outPath+'/train_end')
    dump_pkl(valid_data, outPath+'/valid_end')
    dump_pkl(test_data, outPath+'/test_end')
    
if __name__=="__main__":
    gather_data()
