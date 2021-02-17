#Use this to convert a bvh to a csv as input to the LSTM
from os import listdir
from os.path import isfile, join
from neuralNet.bvhModelNoUI import bvhModelNoUI

from neuralNet.bvh import is_bvh_instance_in_CMU_skeleton_format


def bvh_to_csv(bvh_file: "full path to the bvh_file" = None, bvh_string = None, output_file=''):
    '''
    Given a bvh file path or bvh string, converts to a csv for input to the LSTM for classification, which is saved in output_file
    If the output file is not provided, the csv is saved in the same directory as the bvh with file names corresponding with the exception of the extension.
    Function is file path greedy
    '''
    bvh_model = bvhModelNoUI()

    print("loading bvh...")
    bvh_model.Load(bvh_file, bvh_string)
    print("loaded bvh".format(bvh_file))
    if not bvh_model.is_in_CMU_format():
        print("Warning: bvh model is not in CMU format")
    bvh_model.AsInputData("standardized", save_as_csv = True, csv_path = output_file)
    print("Saving csv to {} for LSTM classification".format(output_file))
    
    
    
    
def bvh_folder_to_csv(bvh_folder, output_folder = ''):
    '''
    Converts all bvh files in a given folder to their corresponding csv for input to the LSTM.
    If the output folder is not provided, the csv is saved in a subdirectory of the bvh_folder with file names corresponding with the exception of the extension.
    '''
    bvh_to_convert = onlyfiles = [f for f in listdir(bvh_folder) if isfile(join(bvh_folder, f))]
    print("Converting the following bvh files: {}".format(bvh_to_convert))
    if input("continue?")!='y':
        return
    bvh_list_to_csv(bvh_to_convert, output_folder)
    
    
def bvh_list_to_csv(list_of_bvh_file_names, output_folder=''):
    '''
    Converts all bvh files in a given list to their corresponding csv for input to the LSTM.
    If the output folder is not provided, the csv is saved in the current working directory with file names corresponding with the exception of the extension.
    '''
    for bvh_file in list_of_bvh_file_names:
        bvh_to_csv(bvh_file, output_folder + bvh_file[bvh_file.rfind( "\\" ) + 1:-3] + 'csv')
    
    
if __name__ == '__main__':
    pass
    