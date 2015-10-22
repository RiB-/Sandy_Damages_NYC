# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:49:32 2015

@author: rfoti
"""

#******************************************************************************
# Importing standard packages
#******************************************************************************
import numpy as np  #library for matrix-array analysis
import matplotlib.pyplot as plt #library to plot graphs
#******************************************************************************

#******************************************************************************
# Importing additional user-specified packages
#******************************************************************************
from PlottingRoutines import Plot_confusion_matrix as PCM   #importing the module to plot the confusion matrix
from PlottingRoutines import Plot_Pred_Labels as PPL #importing the module to plot the spatial maps of classified data
#******************************************************************************

#******************************************************************************
# Setting the working directory paths
#******************************************************************************
Path_01 = 'T:/Philadelphia/Drexel - Research/TPL/Numeric Analyses/Machine Learning Files/'

#******************************************************************************
# Loading Data structures
#******************************************************************************
import pickle #import the module to load/save as pickle
with open(Path_01+'ML_Struct.pkl','rb') as handle:
    ML_Structure = pickle.load(handle)
#end
#******************************************************************************
Name_DF = 'CAP Damage'
Name_CL = 'Random Forest'
#******************************************************************************
# Plotting results
#******************************************************************************

#--------------------------------------------------
# Plotting confusion matrix
#--------------------------------------------------
fc=1
DF = 'CAP Damage' #Assign the Dataframe 
CL = 'Random Forest' #Assign the Classifier
N_F = 6 #Assign the number of features
Labels = ['0','1','2','3','4','5']
CM = ML_Structure[DF][CL]['Confusion_Matrix'][:,:,N_F-1] #Extract the confusion matrix corresponding to the selected Dataframe/Classifier/Number of features
Title = DF+' '+CL+' '+str(N_F)+' Features Confusion Matrix' #Assign the title to be passed to the confusion matrix routine
PCM(CM,Labels, Cmap=plt.cm.Blues, Norm='True', Fig_counter=fc, Title=Title) #Call the routine to plot the confusion matrix
#--------------------------------------------------

#--------------------------------------------------
# Plotting predictions and test data on spatial matrix
#--------------------------------------------------

Path_02 = 'T:/Philadelphia/Drexel - Research/TPL/ASCII Datasets/25x25/Jamaica Bay/'
JamBayMask = np.genfromtxt(Path_02+'JamaicaBay_25x25_Mask.txt',dtype='bool') #import study mask

Rows = ML_Structure[Name_DF]['X_Test'][:,0].astype(int)
Cols = ML_Structure[Name_DF]['X_Test'][:,1].astype(int)

Y_Test = ML_Structure[Name_DF]['Y_Test']
Y_Pred = ML_Structure[Name_DF]['Random Forest']['TestSET'][:,N_F-1]
Subwindows = np.array(([600,720,1720,1850],[790,860,1170,1290]))
Colorlist=['blue','green', 'yellow','orange','orangered','crimson']

fc=fc+1
##Colorlist = ['blue','green', 'yellow',(0.4,0.4,0.4),(1,1,1),(0,0.3,0.4)] #['blue','green', 'yellow','orange','orangered','crimson']
#
#n_labels = len(Colorlist) #extract the desired number of labels
#min_label = np.min(np.unique(Y_Test)) #extract the smalles label from the test vector
#max_label = np.max(np.unique(Y_Test)) #extract the greater label from the test vector

Mask= JamBayMask

PPL(Rows, Cols, Y_Test, Y_Pred, JamBayMask, Subwindows, Fig_counter=fc, Background = JamBayMask.astype('int'))

#[600,720,1600,1720],