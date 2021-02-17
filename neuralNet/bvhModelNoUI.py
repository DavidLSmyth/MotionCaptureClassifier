


from neuralNet.bvh import Bvh, is_bvh_instance_in_CMU_skeleton_format
import numpy as np

import os
from pandas import read_csv

class bvhModelNoUI():
	def __init__( self ):
		self.Skeleton = []
		self.mocapdata = None
		self.FileName = ""
		
	def Load( self, bvh_file = None, bvh_str: "option to initialise from file string" = None):
		#self.PathName = pathname
		#self.FileName = filename
		self.bvh_file = bvh_file
		#tempFile = pathname + filename
		#print(tempFile)
		if bvh_str:
			print("\n Reading bvh string \n")
			self.mocapdata = Bvh(bvh_str)
		elif bvh_file:
			with open( self.bvh_file ) as f:
				self.mocapdata = Bvh( f.read() )
		else:
			raise Exception("Provide either a bvh file or a string representing a bvh file")
		self.Skeleton = self.GetSkeleton()
		return bvh_file
		
	def is_in_CMU_format(self):
		return is_bvh_instance_in_CMU_skeleton_format(self.mocapdata)

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
		
		
				
	def AsInputData( self, strType, save_as_csv = False, csv_path = ''):
		'''
		This seems to convert a bvh to the corresponding CSV in rogerio's paper, which can be used as input to his classifier.
		'''
		data = []
		nFrames = self.GetLength()
		
		#pos = self.FileName.rfind( "\\" ) + 1
		#outputFile = "data_" + strType.lower() + "\\" + self.FileName[pos:-3] + "csv"
		
		#If an output file is not provided, process the bvh to get corresponding csv input data.
		#otherwise, load a previous version of the file that has been created and return
		#if not os.path.isfile( csv_path ):
			#gaugeRef.SetValue( 0 )
			#gaugeOffset = 0
			
		if strType.lower() != "standardized":
			#labelRef.SetLabel( "Analizing: " + self.FileName )
			#gaugeRef.SetRange( 2 * nFrames )
			#gaugeOffset = nFrames
			
			print( "File: " + self.FileName + " ... analyzing", end = ' ' )

			# analyzing data to determine its minimum and maximum values
			minimumR = minimumP = [99999, 99999, 99999]
			maximumR = maximumP = [-99999, -99999, -99999]
			
			origin = [0,0,0]
			
			for frame in range( 1, nFrames ):
				#globalGaugeRef.SetValue( globalGaugeRef.GetValue() + 1 )
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
							
				#gaugeRef.SetValue( frame + 1 )
			#labelRef.SetLabel( "Normalizing: " + self.FileName )
			print( "normalizing" )
		else:
			#gaugeRef.SetRange( nFrames )
			#labelRef.SetLabel( "Loading: " + self.FileName )
			#print( "File: " + self.FileName + " ... loading" )
			pass
		
		header = ""
		structure = ""
		
		for frame in range( 1, nFrames ):
			#globalGaugeRef.SetValue( globalGaugeRef.GetValue() + 1 )
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
			#gaugeRef.SetValue( frame + gaugeOffset + 1 )
		if save_as_csv:
			path_separator = "\\" if "\\" in csv_path else "/" if "/" in csv_path else "/"
			while not os.path.isdir(csv_path[:csv_path.rfind( path_separator )]):
				csv_path = input("{} is not a valid path to save the csv.\nPlease provide a valid csv path: ".format(csv_path[:csv_path.rfind( path_separator )]))
				path_separator = "\\" if "\\" in csv_path else "/" if "/" in csv_path else "/"
				print("using path separator ", path_separator)

			self.SaveAsCSV( data, header, csv_path )
			
		print("Writing the structure of the data to ./structure.txt")
		fileNode = open( "structure.txt","w" )
		fileNode.write( structure )
		fileNode.close()
			
		#else:
			#labelRef.SetLabel( "Loading previous version of file: " + self.FileName )
			#print( "Loading previous version of file: " + self.FileName )
			#gaugeRef.SetRange( 1 )
			#gaugeRef.SetValue( 1 )
			#globalGaugeRef.SetValue( globalGaugeRef.GetValue() + nFrames * 2 )
			
			#data = np.array( read_csv( csv_path ) ).tolist()
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
			