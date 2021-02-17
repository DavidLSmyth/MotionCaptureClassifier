#
# LOADER.PY
# It is responsible for loading all files in a given directory 
#
# Developed by Rogerio E. da Silva, April/2018
#

import wx
from wx.grid import Grid
import os
from bvhmodel import bvhModel
import xml.etree.ElementTree as ET

class Loader( wx.Dialog ):
	def __init__( self, parent ):
		'''
		Initialises a dialog box, allowing the user to select a file
		'''
		wx.Dialog.__init__( self, parent, -1, "Loading Data", size=( 250, 70 ) )
		
		self.vbox = wx.BoxSizer( wx.VERTICAL )

		self.gauge = wx.Gauge( self, -1, style = wx.HORIZONTAL )
		self.vbox.Add( self.gauge, 1, wx.EXPAND )
		
		self.SetSizer( self.vbox )
		self.Centre()
		self.Show()

	def LoadWorkingData( self, outputFile: "", fromDir: "", gridFiles: ""):
		'''
		Loads working data. Not sure exactly what this is doing, further documentation pending.
		'''
		xml = ET.parse( outputFile )
		working = xml.getroot().find( "working" )
		if not( working is None ):
			xml.getroot().remove( working )
		working = ET.SubElement( xml.getroot(), "working", path = fromDir )		
		
		if gridFiles.GetNumberRows() > 1:
			gridFiles.DeleteRows( 0, gridFiles.GetNumberRows() )

		filesCounter = 0
		for root, _, files in os.walk( fromDir ):
			if len( files ) > 1:					
				#labelName = ( root.split( "\\" )[-1] ).replace( " ","_" )
				
				for file in files:
					if file[-3:] == "bvh":
						fileName = ( root + "\\" + file ).replace( fromDir, "" )
						
						gridFiles.AppendRows( 1 )
						gridFiles.SetCellValue( filesCounter, 0, fileName )
						gridFiles.SetCellValue( filesCounter, 1, "unknown" )
							
						filesCounter += 1						
						self.gauge.Pulse()
						
						entry = ET.SubElement( working, "entry" )
						entry.set( "class", "unknown" )
						ET.SubElement( entry, "input" ).text = fileName
						
		working.set( 'size', str( filesCounter ))
		gridFiles.AutoSizeColumns()
		
		output = ET.ElementTree( xml.getroot() )
		output.write( outputFile )
		
		self.Destroy()
				
	def LoadTrainingData( self, outputFile, fromDir, gridFiles, gridLabels ):
		xml = ET.Element( "dataset" )
		training = ET.SubElement( xml, "training", path = fromDir )		
		
		if gridFiles.GetNumberRows() > 1:
			gridFiles.DeleteRows( 0, gridFiles.GetNumberRows() )
		if gridLabels.GetNumberRows() > 1:
			gridLabels.DeleteRows( 0, gridLabels.GetNumberRows() )
		
		filesCounter = frameCounter = 0
		listOfLabels = []
		
		minimumLength = maximumLength = 0
		minimumIndex = maximumIndex = 0
		
		for root, _, files in os.walk( fromDir ):
			if len( files ) > 1:					
				labelName = ( root.split( "\\" )[-1] ).replace( " ","_" )
				newLabelInfo = [ labelName, len( files ), 0 ] 
				
				label = ET.SubElement( training, labelName, size = str( len( files ) ) )
				
				labelLength = 0
				
				for file in files:
					if file[-3:] == "bvh":
						bvh = bvhModel()
						fileName = bvh.Load( root, file ).replace( fromDir, "" )
						
						gridFiles.AppendRows( 1 )
						gridFiles.SetCellValue( filesCounter, 0, fileName )
						gridFiles.SetCellValue( filesCounter, 1, str( len( bvh.GetSkeleton() ) ) )
						gridFiles.SetCellValue( filesCounter, 2, str( bvh.GetLength() ) )
						
						frameCounter += bvh.GetLength()
						labelLength += bvh.GetLength()
						
						if filesCounter == 0 or minimumLength > bvh.GetLength():
							minimumLength = bvh.GetLength()
							minimumIndex = filesCounter
						if filesCounter == 0 or maximumLength < bvh.GetLength():
							maximumLength = bvh.GetLength()
							maximumIndex = filesCounter
							
						filesCounter += 1						
						self.gauge.Pulse()
						
						entryNode = ET.SubElement( label, "entry", njoints = str( len( bvh.GetSkeleton() ) ), length = str( bvh.GetLength() ) )
						ET.SubElement( entryNode, "input" ).text = fileName
				label.set( 'length', str( labelLength ))
				newLabelInfo[2] = labelLength
				listOfLabels.append( newLabelInfo )
				
		training.set( 'size', str( filesCounter ))
		training.set( 'min', str( minimumLength ))
		training.set( 'max', str( maximumLength ))
		training.set( 'length', str( frameCounter ))
		
		gridFiles.AutoSizeColumns()
		gridFiles.SetCellBackgroundColour( minimumIndex, 2, wx.GREEN )		
		gridFiles.SetCellBackgroundColour( maximumIndex, 2, wx.RED )
		
		# updating label class IDs
		nLabels = len( listOfLabels )
		nID = 0
		gridLabels.AppendRows( nLabels )
		
		for label, size, length in listOfLabels:
			classLabel = list( "0" * nLabels )
			classLabel[nID] = '1'
			classLabelID = "".join( classLabel )
			
			gridLabels.SetCellValue( nID, 0, label )
			gridLabels.SetCellValue( nID, 1, classLabelID )
			gridLabels.SetCellValue( nID, 2, str( size ) )
			gridLabels.SetCellValue( nID, 3, str( length ) )
			nID += 1
			
			training.find( label ).set( "class", classLabelID )
			
		gridLabels.AutoSizeColumns()

		output = ET.ElementTree( xml )
		output.write( outputFile )
		
		self.Destroy()
		
		
if __name__ == '__main__':
	app = wx.App()
	dialog = Loader(None)
	import time
	time.sleep(5)
	dialog.LoadWorkingData("D:\\SAUCEFiles\\RogerioLSTMClassifier\\Test\\outputFolder\\datasets.xml", "D:\\SAUCEFiles\\RogerioLSTMClassifier\\mocap", "grid" )
					