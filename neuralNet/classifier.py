#system libraries
import os
import sys
import math
import configparser
import json

#pyPi libraries
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
from numpy import argmax
from pandas import read_csv
from keras.models import load_model

#This configures GPUs so that memory issues don't occur
def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            raise e





def Classify( config_file_loc = "./neuralNet/config/config.ini", csv_file_loc: "The csv file containing input to the LSTM" = ''):
    if not csv_file_loc[-3:] != ".csv":
        raise Exception("Please provide a valid csv file for classification")
        
    if not os.path.isfile(csv_file_loc):
        raise Exception("Please provide a valid csv file location")
        
    print("Classifying data in csv: ", csv_file_loc)


    config_parser = configparser.ConfigParser()
    config_parser.read(config_file_loc)
    print("sections: ",config_parser.sections())
    classification_params = config_parser["CLASSIFICATION"]
    model_params = config_parser["MODEL"]
    
    trained_h5file_loc = classification_params["trained_h5file_loc"]
    print("Using trained h5 file at location {}".format(trained_h5file_loc))

    threshold = float(classification_params["threshold"])
    
    if not os.path.isfile( trained_h5file_loc ):
        raise Exception("{} is not a valid h5 file location".format(trained_h5file_loc))

    model = load_model( trained_h5file_loc )
    print( "Model: " + trained_h5file_loc + " loaded" )
    
    pageSize = int(model_params["pageSize"])

    #read the mapping from one-hot encoding to labels
    label_dict = json.loads(model_params["label_dict"])
    #labels must be kept in order
    labels = [label_dict[key] for key in sorted(label_dict.keys())[::-1]]
        
    #nJoints = int(trainingRootNode[0][0].attrib["njoints"] )
    nJoints = int(model_params["nJoints"])

    dsShape = ( pageSize, 3 + nJoints * 3 )
    
    #read the input csv
    data_set = np.array( read_csv( csv_file_loc ) )

    #possible answers contains the counter for the classifiction for each set of frames
    #of size pageSize. The counter a label is incremented if pageSize consecutive frames are
    #classified as that label. Each counter obviously starts at 0
    #possible_answers = [0 for _ in  labels]
    possible_answers = {string_label : 0 for string_label in label_dict.values()}
                            
    
    answers_distribution = []


    #first configure GPU so don't run out of memory 
    configure_gpus()

    #iterate over frames in sets of pageSize, which will be used as input to the LSTM
    for offset in range( 0, len( data_set ), pageSize ):
        #take the next slice of pageSize frames
        newInput  = np.array( data_set )[ offset : offset + pageSize ]
        #this checks there are enough frames to classify (LSTM is trained to classify pageSize frames at a time)
        if newInput.shape[0] == pageSize:
            #make sure that the input is formatted to the number of frames to classify in first dimension, 
            #number of joints*3 + 3 in second dimension relating to skeleton position and orientation
            inputData  = newInput.reshape( ( 1, dsShape[0], dsShape[1] ) )
            print("Classifying frames {} to {} of {} total...".format(offset, offset + pageSize, math.ceil(len( data_set)/pageSize)))
            answer = model.predict( inputData )
            answers_distribution.append(answer)
            if np.max( answer ) > threshold:
                index = [argmax( value ) for value in answer][0]
                class_prediction = labels[index]
                possible_answers[class_prediction] += 1
    
    class_name = next(filter(lambda x: possible_answers[x] == max(possible_answers.values()), possible_answers.keys()))
    s = sum( possible_answers.values() )
    return (class_name, possible_answers, answers_distribution) if s > 0 else ("Class could not be predicted", [], [])

        
        
# =============================================================================================================================
if __name__ == '__main__':
    configure_gpus()
    Classify( csv_file_loc = r"Data\Input\temp.csv")
    
    
    