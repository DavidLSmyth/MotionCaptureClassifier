import xml.etree.ElementTree as ET
import numpy as np
from numpy import argmax
import os
from bvhModelNoUI import bvhModelNoUI
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from random import shuffle
import configparser
from bvhToCSV import bvh_to_csv
from pandas import read_csv

class NeuralNetNoUI:
	'''
	This class contains methods to configure an LSTM according to Rogerio's paper. The LSTM classifies csv files generated from 
	bvh files to return an action in one of the classes
	'''
	def __init__(self, config_file_loc):
		self.config_parser = configparser.ConfigParser()
		self.config_parser.read(config_file_loc)
		self.training_params = self.config_parser["TRAINING"]
		self.classification_params = self.config_parser["CLASSIFICATION"]
		
		#Set the training params
		self.noEpochs = int(self.training_params['noEpochs'])
		self.XML_file_loc = self.training_params["XML_file_loc"]
		self.pageSize = int(self.training_params["pageSize"])
		self.h5_file_loc = self.training_params["h5file_loc"]
		self.nNeurons = int(self.training_params["nNeurons"])
		self.validation_split = float(self.training_params["validation_split"])
		self.batch_size = int(self.training_params["batch_size"])
		self.loss = self.training_params["loss"]
		
		#set the classification params
		self.trained_h5file_loc = self.classification_params["trained_h5file_loc"]
		
		#not sure why Rogerio did this
		self.SetDataStandardized()
		#Once class has been intialised, load the XML file containing the locations of the training data
		self.LoadXML()
		
	def LoadXML( self ):
		'''
		The dataset is stored in an xml file with the following tags:
		<training length="<>" max="<>" min="<>" path="<>" size="<>"> : The training tag encloses sub-tags which define the training files. 
		length could be total number of frames
		max could be max. number of frames.
		min could be min. number of frames.
		path is the path to the folder containing the bvh files
		size is the number is input files in the dataset
		'''
		#parse the XML tree which defines the dataset
		self.dataset = ET.parse( self.XML_file_loc )
		#set the root node for the "training" and "working" datasets
		self.trainingRootNode = self.dataset.getroot().find( "training" )
		self.workingRootNode  = self.dataset.getroot().find( "working" )
		
		#number of available lables is the length of the 
		self.numLabels = len( self.trainingRootNode )
		#the size of the dataset is defined in the XML metadata
		self.datasetSize = int( self.trainingRootNode.attrib["size"] )
		self.datasetMaxLength = int( self.trainingRootNode.attrib["max"] )
		#the length of the training dataset is defined by the XML metadata
		self.trainingLength = int( self.trainingRootNode.attrib["length"] )
		
		#strange construct - finds the number of joints of the first data point
		for label in self.trainingRootNode:
			for entry in label:
				self.nJoints = int( entry.attrib["njoints"] )
				break
			break
		#David:dsShape seems to be dataset shape
		self.datasetShape = ( self.pageSize, 3 + self.nJoints * 3 )
		
		
	# Set data as NORMALIZED.
	# Normalized data means all values will be rescaled to the interval [0..1]
	def SetDataNormalized( self ):
		self.dataType = "normalized"

	# Set data as RESCALED.
	# Rescaled data means all values will be squashed to the interval [-1..1]
	def SetDataRescaled( self ):
		self.dataType = "rescaled"

	# Set data as STANDARDIZED.
	# Standardized data means all values will be use without any preprocessing
	def SetDataStandardized( self ):
		self.dataType = "standardized"
		
		
	# encode a class name into a one-hot list
	def Encode( self, strCode ):
		encoding = list()
		for char in strCode:
			if char == "0":
				encoding.append( 0 )
			else:
				encoding.append( 1 )
		return encoding
		
		
		
	# Performs the fitting of the LSTM model
	def Fit( self ):	
		pathName = self.trainingRootNode.attrib["path"]
		if not os.path.isdir( "data_"+self.dataType ):
			os.mkdir( "data_"+self.dataType )
		listOfFiles = []
		for labelNode in self.trainingRootNode:
			labelCat = self.Encode( labelNode.attrib[ "class" ] ) 
			#labelCat = labelNode.attrib[ "class" ] 
			for entryNode in labelNode.iter( "entry" ):
				listOfFiles.append( [pathName, entryNode.find( "input" ).text, labelCat] )
		shuffle( listOfFiles )
		inputData  = np.array([])
		outputData = np.array([])
		data_set   = []
		nSamples   = 0
		
		for counter, entry in enumerate( listOfFiles, start=1 ):
			try:
				#self.stGeneralLabel.SetLabel( "Loading Training file #" + str( counter ) + " of " + str( len( listOfFiles ) ) )
				
				bvh = bvhModel()
				bvh.Load( entry[0], entry[1] )
				data_set = bvh.AsInputData( self.dataType, self.GeneralGauge, self.stLocalLabel, self.LocalGauge )
				
			finally:
				newOutput = np.array( entry[2] ).reshape( 1, self.numLabels )
				
				for offset in range( 0, len( data_set ), self.pageSize ):
					newInput  = np.array( data_set )[ offset : offset + self.pageSize ]
					if newInput.shape[0] == self.pageSize:
						newInput  = newInput.reshape( ( 1, self.datasetShape[0], self.datasetShape[1] ) )
					
						if nSamples == 0:
							inputData  = newInput
							outputData = newOutput
						else:
							inputData  = np.vstack( ( inputData, newInput ) )
							outputData = np.vstack( ( outputData, newOutput ) )
						nSamples += 1
		self.model.fit( inputData, outputData, validation_split = 0.1, batch_size = 10, epochs = self.nEpochs, verbose = 2 )
		self.model.save( self.h5_file_loc )
		print( "=== TRAINING COMPLETE ===\nSaving model to : " + self.h5_file_loc )
		
	# Creates a new LSTM model
	def Compile( self ):
		#self.stGeneralLabel.SetLabel( "Creating Neural net ..." )
		print("Compiling the LSTM...")
		print("Using the parameters: ", self.training_params)
		self.LoadXML()
		
		# creating the NN model
		self.model = Sequential()
		self.model.add( LSTM( nNeurons, return_sequences = True, input_shape = self.datasetShape ) )
		self.model.add( LSTM( nNeurons ) )
		self.model.add( Dense( self.numLabels, activation = 'softmax' ) )
		self.model.compile( loss = self.loss, optimizer = Adam( lr = 0.001 ), metrics = ['accuracy'] )
		self.model.summary()
		
	# Loads a previous LSTM model or creates a new one if none exists
	def LoadLSTM( self, action ):	
		result = True
		if action:
			if os.path.isfile( self.trained_h5file_loc ):
				#self.stGeneralLabel.SetLabel( "Training Neural net ..." )
				#self.GeneralGauge.SetRange( 10 )
				#self.GeneralGauge.SetValue( 10 )
				
				#self.stLocalLabel.SetLabel( "Loading previously trained LSTM ..." )
				#self.LocalGauge.SetRange( 10 )
				#self.LocalGauge.SetValue( 10 )
				
				self.model = load_model( self.trained_h5file_loc )
			else:
				#wx.MessageBox( 'No previously trained neural net found','Error', wx.OK | wx.ICON_ERROR )
				print("Could not find previously trained neural net, please train and provide a valid path to a weight file in the config file")
				result = False
		else:
			self.Compile()
			self.Fit()
			
		#self.Hide()
		return result
		
	def PredictBVH(self, bvh_file):
		self.LoadLSTM(True)
		#training data needs to be loaded to provide the classes to the LSTM
		if self.dataset is not None:
			#self.Show()

			#pathName = bvh_file
			#self.workingRootNode.attrib["path"]
			
			#self.stGeneralLabel.SetLabel( "Classifying dataset ..." )
			#self.GeneralGauge.SetRange( int( self.workingRootNode.attrib["size"] ) )
			#self.GeneralGauge.SetValue( 0 )
			
			Labels = []
			for entry in self.trainingRootNode:
				Labels.append( entry.tag )
			testData  = []
			Threshold = 0
			
			for counter, entry in enumerate( self.workingRootNode.iter( "entry" ) ):
				try:
					className = "unknown"
					bvh = bvhModelNoUI()
					
					#sets the skeleton
					bvh.Load(bvh_file)
					
					data_set = bvh.AsInputData(self.dataType)
					#LSTM returns a number of possible answers, not immediately clear why this is.
					print("There are {} possible answers".format(len(Labels)))
					possibleAnswers = [0] * len( Labels )
					for offset in range( 0, len( data_set ), self.pageSize ):
						newInput  = np.array( data_set )[ offset : offset + self.pageSize ]
						if newInput.shape[0] == self.pageSize:
							inputData  = newInput.reshape( ( 1, self.datasetShape[0], self.datasetShape[1] ) )
							answer = self.model.predict( inputData )
							if np.max( answer ) > Threshold:
								index = [argmax( value ) for value in answer][0]
								possibleAnswers[index] += 1
					
					print( "possibleAnswers", possibleAnswers )
					index = [argmax( value ) for value in [possibleAnswers]][0]
					if index < len( Labels ):
						className = Labels[ index ]
						
				except Exception as e:
					raise e
					
				finally:
					
					sum = np.sum( possibleAnswers )
					for col, value in enumerate( possibleAnswers ):
						if sum > 0:
							#grid.SetCellValue( counter, 2 + col, str( value ) + " (" + str( round( value / sum * 100, ndigits = 2 ) ) + "%)" )
							if col == index:
								pass
								#grid.SetCellBackgroundColour( counter, 2 + col, wx.GREEN )
							entry.set( "class", className )
							#grid.SetCellValue( counter, 1, className )
						else:
							#grid.SetCellValue( counter, 2 + col, "0" )
							#grid.SetCellBackgroundColour( counter, 2 + col, wx.RED )
							#entry.set( "class", "unknown" )
							#grid.SetCellValue( counter, 1, "unknown" )
							pass

					counter += 1
					#self.GeneralGauge.SetValue( counter )
							
			#grid.AutoSizeColumns()
			print("Predicted class: {}".format(className))
			#output = ET.ElementTree( self.dataset.getroot() )
			#output.write( self.XMLname )
	
	
	
	
	
	
	
	
	
	def PredictCSV(self, csv_file):
		self.LoadLSTM(True)
		#training data needs to be loaded to provide the classes to the LSTM
		if self.dataset is not None:
			#self.Show()

			#pathName = bvh_file
			#self.workingRootNode.attrib["path"]
			
			#self.stGeneralLabel.SetLabel( "Classifying dataset ..." )
			#self.GeneralGauge.SetRange( int( self.workingRootNode.attrib["size"] ) )
			#self.GeneralGauge.SetValue( 0 )
			
			Labels = []
			for entry in self.trainingRootNode:
				Labels.append( entry.tag )
			testData  = []
			Threshold = 0
			
			for counter, entry in enumerate( self.workingRootNode.iter( "entry" ) ):
				try:
					className = "unknown"
					#bvh = bvhModelNoUI()
					
					#sets the skeleton
					#bvh.Load(bvh_file)
					
					data_set = np.array( read_csv( csv_file ) ).tolist()
					print("page size: {}".format(self.pageSize))
					print("dataset size: ", self.datasetSize)
					print("dataset shape: ", self.datasetShape)
					#LSTM returns a number of possible answers, not immediately clear why this is.
					print("There are {} possible answers: {}".format(len(Labels), Labels))
					possibleAnswers = [0] * len( Labels )
					for offset in range( 0, len( data_set ), self.pageSize ):
						newInput  = np.array( data_set )[ offset : offset + self.pageSize ]
						if newInput.shape[0] == self.pageSize:
							print("New input: ", newInput)
							inputData  = newInput.reshape( ( 1, self.datasetShape[0], self.datasetShape[1] ) )
							print("inputData: \n",inputData)
							answer = self.model.predict( inputData )
							if np.max( answer ) > Threshold:
								index = [argmax( value ) for value in answer][0]
								possibleAnswers[index] += 1
					
					print( "possibleAnswers", possibleAnswers )
					index = [argmax( value ) for value in [possibleAnswers]][0]
					if index < len( Labels ):
						className = Labels[ index ]
						
				except Exception as e:
					raise e
					
				finally:
					
					sum = np.sum( possibleAnswers )
					for col, value in enumerate( possibleAnswers ):
						if sum > 0:
							#grid.SetCellValue( counter, 2 + col, str( value ) + " (" + str( round( value / sum * 100, ndigits = 2 ) ) + "%)" )
							if col == index:
								pass
								#grid.SetCellBackgroundColour( counter, 2 + col, wx.GREEN )
							entry.set( "class", className )
							#grid.SetCellValue( counter, 1, className )
						else:
							#grid.SetCellValue( counter, 2 + col, "0" )
							#grid.SetCellBackgroundColour( counter, 2 + col, wx.RED )
							#entry.set( "class", "unknown" )
							#grid.SetCellValue( counter, 1, "unknown" )
							pass

					counter += 1
					#self.GeneralGauge.SetValue( counter )
							
			#grid.AutoSizeColumns()
			print("Predicted class: {}".format(className))
			#output = ET.ElementTree( self.dataset.getroot() )
			#output.write( self.XMLname )
	
	
	
	
	
	
	

	# Make predictions on a trained model using the working dataset
	def Predict(self):
		#self.pageSize = pageSize
		
		self.LoadXML()
		
		if self.dataset is not None:
			#self.Show()

			pathName = self.workingRootNode.attrib["path"]
			
			#self.stGeneralLabel.SetLabel( "Classifying dataset ..." )
			#self.GeneralGauge.SetRange( int( self.workingRootNode.attrib["size"] ) )
			#self.GeneralGauge.SetValue( 0 )
			
			Labels = []
			for entry in self.trainingRootNode:
				Labels.append( entry.tag )
				
			testData  = []
			Threshold = 0
			
			for counter, entry in enumerate( self.workingRootNode.iter( "entry" ) ):
				try:
					className = "unknown"
					bvh	   = bvhModel()
					bvh.Load( pathName, entry.find( "input" ).text )
					
					data_set = bvh.AsInputData(self.dataType)
					#LSTM returns a number of possible answers, not immediately clear why this is.
					print("There are {} possible answers".format(len(Labels)))
					possibleAnswers = [0] * len( Labels )
					for offset in range( 0, len( data_set ), self.pageSize ):
						newInput  = np.array( data_set )[ offset : offset + self.pageSize ]
						if newInput.shape[0] == self.pageSize:
							inputData  = newInput.reshape( ( 1, self.datasetShape[0], self.datasetShape[1] ) )
							answer	 = self.model.predict( inputData )
							if np.max( answer ) > Threshold:
								index = [argmax( value ) for value in answer][0]
								possibleAnswers[index] += 1
					
					#print( possibleAnswers )
					index = [argmax( value ) for value in [possibleAnswers]][0]
					if index < len( Labels ):
						className = Labels[ index ]
					
				finally:
					
					sum = np.sum( possibleAnswers )
					for col, value in enumerate( possibleAnswers ):
						if sum > 0:
							#grid.SetCellValue( counter, 2 + col, str( value ) + " (" + str( round( value / sum * 100, ndigits = 2 ) ) + "%)" )
							if col == index:
								#grid.SetCellBackgroundColour( counter, 2 + col, wx.GREEN )
								pass
							#entry.set( "class", className )
							#grid.SetCellValue( counter, 1, className )
						else:
							#grid.SetCellValue( counter, 2 + col, "0" )
							#grid.SetCellBackgroundColour( counter, 2 + col, wx.RED )
							#entry.set( "class", "unknown" )
							#grid.SetCellValue( counter, 1, "unknown" )
							pass

					counter += 1
					#self.GeneralGauge.SetValue( counter )
							
			#grid.AutoSizeColumns()
			
			output = ET.ElementTree( self.dataset.getroot() )
			output.write( self.XMLname )
		
		

		
		
		
if __name__ == '__main__':
	test_classification = NeuralNetNoUI("D:\\SAUCEFiles\\RogerioLSTMClassifier\\keras\\config\\config.ini")
	test_classification.PredictCSV("D:\\SAUCEFiles\\ProvidingMoCapToFBX\\AnimatedFBX\\ConvertedCSV\\test_push.csv")