# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:59:52 2019

@author: admin
"""

import requests

import os
import sys
sys.path.append("..")

def get_class_from_bvh_file(bvh_file_path):
    assert os.path.exists(bvh_file_path), f"{bvh_file_path} is not a valid file path"
    assert os.path.splitext(bvh_file_path)[1] == ".bvh", f"{bvh_file_path} does not have a .bvh extension"
    bvh_string = open(bvh_file_path, 'r').read()
    #send the string to the server
    result = requests.post("http://localhost:5000/classify", json={"bvh": bvh_string})
    return result


def main():
    #print("cwd: ", os.getcwd())

    #read the bvh file as a string

    bvh_files = [f for f in os.listdir("./SampleData") if os.path.isfile(os.path.join("./SampleData", f))]
    for bvh_file in bvh_files:

        result = get_class_from_bvh_file(os.path.join("./SampleData", bvh_file))

        print("Results for " + bvh_file + ":")
        print("--------- Predictions  Summary -----------")
        print("Predictions summary: ", result.json())
        print("--------- Predictions  Summary -----------\n")

        print("--------- Predicted Class -----------")
        print(result.json()["predicted_class"])
        print("--------- Predicted Class -----------")

        print("\n\n")

if __name__ == "__main__":
    main()
