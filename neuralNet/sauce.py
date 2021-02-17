import time
import wx
from NN import NeuralNet

XML = "datasets.xml"
def run_LSTM_prediction():
    app = wx.App()
    lstmNN = NeuralNet( XML )
    app.MainLoop()
    #if the LSTM has a valid h5 weight file, then load and use it to classify
    if lstmNN.LoadLSTM():
        lstmNN.Predict()

def run_sample_wx():
    import wx

    app = wx.App()  # Create a new app, don't redirect stdout/stderr to a window.
    app.MainLoop()
    frame = wx.Frame(None, wx.ID_ANY, "Hello World") # A Frame is a top-level window.
    frame.Show(True)     # Show the frame.



if __name__ == '__main__':
    run_LSTM_prediction()
