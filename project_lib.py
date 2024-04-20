###########################################################################################################################################
#//////////////////////////////////////////   IMPORTS    //////////////////////////////////////////////////////////////////////////////////
# Standard library imports
import os
import sys
import warnings

# Third-party library imports
import numpy as np
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.metrics import confusion_matrix, classification_report # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore
import librosa # type: ignore
import librosa.display # type: ignore
from IPython.display import Audio # type: ignore

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


###########################################################################################################################################
#//////////////////////////////////////////   CONSTANTS    ////////////////////////////////////////////////////////////////////////////////

labels = ['natural', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']


###########################################################################################################################################
#//////////////////////////////////////////   FUNCTIONS    ////////////////////////////////////////////////////////////////////////////////

# Function to print confusion matrix and classification report
def print_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("confusion matrix")
    plt.show()
    return cm

# Produce model report
def model_report(y_test, y_pred): 
    print_confusion_matrix(y_test, y_pred, labels)
    print(classification_report(y_test, y_pred, target_names=labels))
    return None


###########################################################################################################################################
#//////////////////////////////////////////   FUNCTIONS    ////////////////////////////////////////////////////////////////////////////////