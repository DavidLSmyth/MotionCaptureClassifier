# LSTM for classifying MoCap data
For details for how this project was implemented, please refer to the paper: https://v-sense.scss.tcd.ie/research/using-lstm-for-automatic-classification-of-human-motion-capture-data/

## Necessary Hardware/Software
* A machine with a GPU with at least 4Gb of GPU memory. 
* Python 3.x

## Dependencies
A core set of dependencies for use with the Flask api are listed in requirements.txt. An extended set which can run the GUI are listed in extended_requirements.txt. 

## Input
Input is a bvh file. The file must have 31 joints conforming to the CMU mocap dataset: http://mocap.cs.cmu.edu/

## Output
The output is a vector encoding of 4 motions: run, jump, walk, bend down.

## Running the Program
* We recommend creating a new virtual environment
* Install the dependencies in requirements.txt: `pip install -r requirements.txt`
* Update neuralNet\config\config.ini if necessary with the correct file paths. The defaults are relative and can be left for testing purposes.
* With the virtual environment activated, `run start_server.bat` or `start_server.sh`
* Run `run_client_example.bat` to call the classification server with sample animations. Optionally can use postman, curl or any other service to run the classification api at IP:5000


