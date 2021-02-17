# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:17:27 2019

@author: 13383861
"""

import sys
import os
sys.path.append(r".")
#print("Path: ", sys.path)
from flask import Flask, request, abort

from neuralNet.bvh import is_bvh_string_in_CMU_skeleton_format
from neuralNet.classifier import Classify
from neuralNet.bvhToCSV import bvh_to_csv
from flask import jsonify 
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InternalError
#%%


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



'''
An api to classify bvh MoCap data via an LSM.
'''


app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    if not request.json or not 'bvh' in request.json:
        print("Request: ", request.json)
#        abort(400)
    else:
        #print("Request: ", request.json)
        bvh_string = request.json['bvh']
        if not is_bvh_string_in_CMU_skeleton_format(bvh_string):
                return jsonify({"Error: Data does not conform to CMU skeleton, cannot classify"}), 201
        else:
            #first create a csv representation of the data for input to LSTM, as outlined in Rogerio's paper
            temp_csv_loc = r"./Data/Input/temp.csv"
            bvh_to_csv(bvh_string = bvh_string, output_file = temp_csv_loc)
            try:
                class_prediction = Classify( "./neuralNet/config/config.ini", temp_csv_loc)
                #remove the csv file
                os.remove(temp_csv_loc)
                return jsonify({"predicted_class": class_prediction[0], "predictions": class_prediction[1]}), 201

            except InternalError:
                return jsonify({"predicted_class": "Insufficient GPU memory to predict"}), 500
            
            
@app.route('/ping', methods=['GET'])
def ping(): 
    return "Pinged"

#@app.route('/get_skeleton_model_params', methods=['GET'])
#def get_skeleton_model_params(): 
#    return "Pinged"

if __name__ == '__main__':
    #need to configure GPUs before everything else or get runtime errors
    configure_gpus()
    app.run(debug = False, threaded=False)