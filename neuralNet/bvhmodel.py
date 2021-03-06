from bvh import Bvh, BvhNode
import numpy as np
#import wx
import os
from pandas import read_csv

class bvhModel():
	def __init__( self ):
		self.Skeleton = []
		self.mocapdata = None
		self.FileName = ""
		
	def Load( self, pathname, filename ):
		self.PathName = pathname
		self.FileName = filename
		tempFile = pathname + filename
		print(tempFile)
		
		with open( tempFile ) as f:
			self.mocapdata = Bvh( f.read() )
		self.Skeleton = self.GetSkeleton()
		
		return tempFile

	def GetLength( self ):
		if self.mocapdata:
			return self.mocapdata.nframes
		return 0
		
	def GetSkeleton( self ):
		self.Skeleton = []
	
		def iterate_joints( joint ):
			self.Skeleton.append( str( joint ))
			for child in joint.filter( 'JOINT' ):
				iterate_joints( child )
		
		iterate_joints( next( self.mocapdata.root.filter( 'ROOT' )))
		
		return self.Skeleton
	
	def GetData( self ):
		return self.mocapdata
				
	def AsInputData( self, strType, globalGaugeRef, labelRef, gaugeRef ):
		'''
		This seems to convert a bvh to the corresponding CSV in rogerio's paper, which can be used as input to his classifier.
		'''
		data = []
		nFrames = self.GetLength()
		
		pos = self.FileName.rfind( "\\" ) + 1
		outputFile = "data_" + strType.lower() + "\\" + self.FileName[pos:-3] + "csv"
		
        #If an output file is not provided, process the bvh to get corresponding csv input data.
        #otherwise, load a previous version of the file that has been created and return
		if not os.path.isfile( outputFile ): 
			gaugeRef.SetValue( 0 )
			gaugeOffset = 0
			
			if strType.lower() != "standardized":
				labelRef.SetLabel( "Analizing: " + self.FileName )
				gaugeRef.SetRange( 2 * nFrames )
				gaugeOffset = nFrames
				
				print( "File: " + self.FileName + " ... analyzing", end = ' ' )

				# analyzing data to determine its minimum and maximum values
				minimumR = minimumP = [99999, 99999, 99999]
				maximumR = maximumP = [-99999, -99999, -99999]
				
				origin = [0,0,0]
				
				for frame in range( 1, nFrames ):
					globalGaugeRef.SetValue( globalGaugeRef.GetValue() + 1 )
					for joint in self.Skeleton:	
						jointName = joint.split(" ")[1]
						i = 0
						for channel in self.mocapdata.joint_channels( jointName ):
							channelValue = self.mocapdata.frame_joint_channel( frame, jointName, channel )
							
							if channel[1:] != "position":
								if minimumR[i] > channelValue:
									minimumR[i] = channelValue
								if maximumR[i] < channelValue:
									maximumR[i] = channelValue
							else:
								if frame == 1:
									origin[i] = channelValue
								if minimumP[i] > channelValue:
									minimumP[i] = channelValue
								if maximumP[i] < channelValue:
									maximumP[i] = channelValue
							i = (i+1) % 3
								
					gaugeRef.SetValue( frame + 1 )
				labelRef.SetLabel( "Normalizing: " + self.FileName )
				print( "normalizing" )
			else:
				gaugeRef.SetRange( nFrames )
				labelRef.SetLabel( "Loading: " + self.FileName )
				print( "File: " + self.FileName + " ... loading" )
			
			header = ""
			structure = ""
			
			for frame in range( 1, nFrames ):
				globalGaugeRef.SetValue( globalGaugeRef.GetValue() + 1 )
				frameData = []
				for joint in self.Skeleton:	
					jointName = joint.split(" ")[1]
					i = 0
					for channel in self.mocapdata.joint_channels( jointName ):
						if frame == 1:
							if header == "":
								header = jointName + "_" + channel
							else:
								header += "," + jointName + "_" + channel
							structure += jointName + "_" + channel + "\n"
								
						channelValue = self.mocapdata.frame_joint_channel( frame, jointName, channel )
						
						if channel[1:] != "position":
							if strType.lower() == "normalized":
								channelValue = ( channelValue - minimumR[i] ) / ( maximumR[i] - minimumR[i] )
							elif strType.lower() == "rescaled":
								channelValue = 2 * (( channelValue - minimumR[i] ) / ( maximumR[i] - minimumR[i] )) - 1
						else:
							if strType.lower() != "standardized":
								channelValue -= origin[i]
								
								if strType.lower() == "normalized":
									if i == 1:
										channelValue = ( channelValue - minimumP[i] ) / ( maximumP[i] - minimumP[i] )
									else:
										channelValue = 0
								elif strType.lower() == "rescaled":
									channelValue = 2 * (( channelValue - minimumP[i] ) / ( maximumP[i] - minimumP[i] )) - 1
								
						frameData.append( channelValue )
						i = (i+1) % 3
					
				data.append( frameData )
				gaugeRef.SetValue( frame + gaugeOffset + 1 )

			self.SaveAsCSV( data, header, outputFile )
			
			fileNode = open( "structure.txt","w" )
			fileNode.write( structure )
			fileNode.close()
			
		else:
			labelRef.SetLabel( "Loading previous version of file: " + self.FileName )
			print( "Loading previous version of file: " + self.FileName )
			gaugeRef.SetRange( 1 )
			gaugeRef.SetValue( 1 )
			globalGaugeRef.SetValue( globalGaugeRef.GetValue() + nFrames * 2 )
			
			data = np.array( read_csv( outputFile ) ).tolist()
			#data = np.array( read_csv( outputFile, header = None ) ).tolist()
			
		return np.array( data )
		
	def SaveAsCSV( self, data, header, outputFile ):
		fileNode = open( outputFile,"w" )
		
		fileNode.write( header + "\n" )
		for entry in data:
			entryStr = ""
			for value in entry:
				if entryStr != "":
					entryStr += ","
				entryStr += str( value )
			fileNode.write( entryStr + "\n" )
		fileNode.close()
        
    def saveAsBVH(self, outputFilePath):
        '''
        Saves the current bvhmodel to the given outputFilePath
        '''
        
        

			