import wx
import xml.etree.ElementTree as ET
import numpy as np
from numpy import argmax
import os
from bvhmodel import bvhModel
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from random import shuffle

class NeuralNet( wx.Dialog ):
	def __init__( self, XML, pageSize = 5, nEpochs = 15 ):
		 
		#wx.Dialog.__init__( self, None, -1, "LSTM", size=( 400, 100 ) )
		super(NeuralNet, self).__init__(None, -1, "LSTM", size=( 400, 100 ) )
		
		self.LSTM_MODEL	  = "sauce_lstm_model.h5"
		self.model		   = None
		self.XMLname		 = XML
		self.Dataset		 = None
		self.workingRootNode = None
		self.pageSize		= pageSize
		self.nEpochs		 = nEpochs
		
		self.SetDataStandardized()
		
		self.Bind( wx.EVT_CLOSE, self.OnTerminate )

		self.vbox = wx.BoxSizer( wx.VERTICAL )
		
		self.stGeneralLabel = wx.StaticText( self, wx.ID_ANY )
		self.vbox.Add( self.stGeneralLabel, 1, wx.EXPAND )
		
		self.GeneralGauge = wx.Gauge( self, -1, style = wx.HORIZONTAL )
		self.vbox.Add( self.GeneralGauge, 1, wx.EXPAND )
		
		self.stLocalLabel = wx.StaticText( self, wx.ID_ANY, style = wx.ST_ELLIPSIZE_START )
		self.vbox.Add( self.stLocalLabel, 1, wx.EXPAND )
		
		self.LocalGauge = wx.Gauge( self, -1, style = wx.HORIZONTAL )
		self.vbox.Add( self.LocalGauge, 1, wx.EXPAND )

		self.SetSizer( self.vbox )
		self.Centre()
		self.Show()
		print("Dialog initialised")

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
		
	## Terminates the editor canceling last actions
	# \param event event object to be processed
	def OnTerminate( self, event ):
		self.Destroy()
		
	# Get current model filename
	def GetFileName( self ):
		return self.LSTM_MODEL
		
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

		self.GeneralGauge.SetRange( self.trainingLength*2 )
		self.GeneralGauge.SetValue( 0 )
		#self.stGeneralLabel.SetLabel( "" )
		
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
				self.stGeneralLabel.SetLabel( "Loading Training file #" + str( counter ) + " of " + str( len( listOfFiles ) ) )
				
				bvh = bvhModel()
				bvh.Load( entry[0], entry[1] )
				data_set = bvh.AsInputData( self.dataType, self.GeneralGauge, self.stLocalLabel, self.LocalGauge )
				
			finally:
				newOutput = np.array( entry[2] ).reshape( 1, self.numLabels )
				
				for offset in range( 0, len( data_set ), self.pageSize ):
					newInput  = np.array( data_set )[ offset : offset + self.pageSize ]
					if newInput.shape[0] == self.pageSize:
						newInput  = newInput.reshape( ( 1, self.dsShape[0], self.dsShape[1] ) )
					
						if nSamples == 0:
							inputData  = newInput
							outputData = newOutput
						else:
							inputData  = np.vstack( ( inputData, newInput ) )
							outputData = np.vstack( ( outputData, newOutput ) )
				
						nSamples += 1
			
		self.stGeneralLabel.SetLabel( "Training Neural net ..." )
		self.GeneralGauge.SetValue( self.GeneralGauge.GetRange() )
		self.stLocalLabel.SetLabel( "" )
		self.LocalGauge.SetValue( self.LocalGauge.GetRange() )
				
		self.model.fit( inputData, outputData, validation_split = 0.1, batch_size = 10, epochs = self.nEpochs, verbose = 2 )		
		self.model.save( self.LSTM_MODEL )
		print( "=== TRAINING COMPLETE ===\nSaving model: " + self.LSTM_MODEL )

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
		self.Dataset = ET.parse( self.XMLname )
		
		self.trainingRootNode = self.Dataset.getroot().find( "training" )
		self.workingRootNode  = self.Dataset.getroot().find( "working" )
		
		self.numLabels	  = len( self.trainingRootNode )
		self.dsSize		 = int( self.trainingRootNode.attrib["size"] )
		self.dsMaxLength	= int( self.trainingRootNode.attrib["max"] )
		self.trainingLength = int( self.trainingRootNode.attrib["length"] )
		
		for label in self.trainingRootNode:
			for entry in label:
				self.nJoints = int( entry.attrib["njoints"] )
				break
			break
		#David:dsShape seems to be dataset shape
		self.dsShape = ( self.pageSize, 3 + self.nJoints * 3 )
		#self.dsShape = ( self.dsMaxLength, 3 + self.nJoints * 3 )
	
	# Creates a new LSTM model
	def Compile( self ):
		self.stGeneralLabel.SetLabel( "Creating Neural net ..." )
		
		self.LoadXML()
		
		nNeurons = 15
		
		# creating the NN model
		self.model = Sequential()
		self.model.add( LSTM( nNeurons, return_sequences = True, input_shape = self.dsShape ) )
		self.model.add( LSTM( nNeurons ) )
		self.model.add( Dense( self.numLabels, activation = 'softmax' ) )
		self.model.compile( loss = 'mse', optimizer = Adam( lr = 0.001 ), metrics = ['accuracy'] )
		self.model.summary()
		
	# Loads a previous LSTM model or creates a new one if none exists
	def LoadLSTM( self, action ):	
		result = True
		if action:
			if os.path.isfile( self.LSTM_MODEL ):
				self.stGeneralLabel.SetLabel( "Training Neural net ..." )
				self.GeneralGauge.SetRange( 10 )
				self.GeneralGauge.SetValue( 10 )
				
				self.stLocalLabel.SetLabel( "Loading previously trained LSTM ..." )
				self.LocalGauge.SetRange( 10 )
				self.LocalGauge.SetValue( 10 )
				
				self.model = load_model( self.LSTM_MODEL )
			else:
				wx.MessageBox( 'No previously trained neural net found','Error', wx.OK | wx.ICON_ERROR )
				result = False
		else:
			self.Compile()
			self.Fit()
			
		self.Hide()
		return result

	# Make predictions on a trained model using the working dataset
	def Predict( self, grid, pageSize ):
		self.pageSize = pageSize
		
		self.LoadXML()
		
		if self.Dataset is not None:
			self.Show()

			pathName = self.workingRootNode.attrib["path"]
			
			self.stGeneralLabel.SetLabel( "Classifying dataset ..." )
			self.GeneralGauge.SetRange( int( self.workingRootNode.attrib["size"] ) )
			self.GeneralGauge.SetValue( 0 )
			
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
					
					data_set = bvh.AsInputData( self.dataType, self.GeneralGauge, self.stLocalLabel, self.LocalGauge )
					#LSTM returns a number of possible answers, not immediately clear why this is.
					print("There are {} possible answers".format(len(Labels)))
					possibleAnswers = [0] * len( Labels )
											
					for offset in range( 0, len( data_set ), self.pageSize ):
						newInput  = np.array( data_set )[ offset : offset + self.pageSize ]
						if newInput.shape[0] == self.pageSize:
							inputData  = newInput.reshape( ( 1, self.dsShape[0], self.dsShape[1] ) )
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
							grid.SetCellValue( counter, 2 + col, str( value ) + " (" + str( round( value / sum * 100, ndigits = 2 ) ) + "%)" )
							if col == index:
								grid.SetCellBackgroundColour( counter, 2 + col, wx.GREEN )
							entry.set( "class", className )
							grid.SetCellValue( counter, 1, className )
						else:
							grid.SetCellValue( counter, 2 + col, "0" )
							grid.SetCellBackgroundColour( counter, 2 + col, wx.RED )
							entry.set( "class", "unknown" )
							grid.SetCellValue( counter, 1, "unknown" )

					counter += 1
					self.GeneralGauge.SetValue( counter )
							
			grid.AutoSizeColumns()
			
			output = ET.ElementTree( self.Dataset.getroot() )
			output.write( self.XMLname )
		
		self.Hide()
			
