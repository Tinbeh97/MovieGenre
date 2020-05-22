from keras.applications.vgg16 import VGG16
import numpy as np

def remove_last_layers(model):
    """To remove the last FC layers of VGG and get the 4096 dim features"""
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

vgg_model_16 = VGG16(include_top=True, weights="imagenet")
#vgg_model_19 = VGG19(include_top=True, weights="imagenet")

remove_last_layers(vgg_model_16)
#remove_last_layers(vgg_model_19)

def features_batch(frames, model_name="vgg16"):

    if model_name.lower() in ["vgg16", "vgg_16"]:
        model = vgg_model_16
      
    imageTensor = np.array(frames)

    modelFeature =  model.predict(imageTensor, verbose=1)
    return modelFeature

