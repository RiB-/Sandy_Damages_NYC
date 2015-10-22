# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:26:34 2015

@author: Romano
"""

#******************************************************************************
# Importing standard packages
#******************************************************************************
import numpy as np  #library for matrix-array analysis
import pandas as pd  #library for advanced data analysis
from sklearn.cross_validation import train_test_split #import package to create the train and test dataset
from sklearn.preprocessing import StandardScaler as SC #import the standard scaler package to standardize datasets
from sklearn.metrics import confusion_matrix as CM #import he confusion matrix package to evaluate classification performance
#******************************************************************************
#******************************************************************************
# Importing additional user-specified packages
#******************************************************************************
import MLearn as ML #import user defined package containig some machine learning algorithms
#******************************************************************************

#******************************************************************************
# Setting the working directory paths
#******************************************************************************
Path_01 = 'T:/Philadelphia/Drexel - Research/TPL/ASCII Datasets/25x25/Jamaica Bay/Dataframes/'
Path_02 = 'T:/Philadelphia/Drexel - Research/TPL/Numeric Analyses/Machine Learning Files/'
Cell_size_in = 25 #setting up the spatial resolution of the input matrices
#******************************************************************************

#******************************************************************************
# Loading the input dataframes
#******************************************************************************
CAP_DF_P = pd.load(Path_01+'CAP_Dataframe_Polished.pkl') #Loads the CAP damage dataframe
TREE_DF_P = pd.load(Path_01+'TREE_Dataframe_Polished.pkl') #Loads the street treee damage dataframe
TS_DF_P = pd.load(Path_01+'TS_Dataframe_Polished.pkl') #Loads the traffic signals damage dataframe
#******************************************************************************

#******************************************************************************
# Classify tree damages into Minor (0) and Major (1)
#******************************************************************************
TREE_DF_P.TREE_DMG[TREE_DF_P.TREE_DMG<4] = 0
TREE_DF_P.TREE_DMG[TREE_DF_P.TREE_DMG>3] = 1
#******************************************************************************


#******************************************************************************
# Drop unnecessary features
#******************************************************************************
TREE_DF_P.drop(['CAP_DMG_EST'], axis=1) #drop estimated CAP damage feature
TS_DF_P.drop(['CAP_DMG_EST'], axis=1) #drop estimated CAP damage feature
#******************************************************************************

#******************************************************************************
# Ininzializing parameters and algorithms
#******************************************************************************

#----------------------------------------------------------------
# Initialize USER DEFINED parameters to apply to classification
#----------------------------------------------------------------
Standardize = 'True' #Define whether to standardize the feature fields
Min_N_Feat = np.array([1,1]) #Define the minimum number of features to be analyzed for: CAP DATA, TREE DATA
Max_N_Feat = np.array([10,8]) #Define the maximum number of features to be analyzed for: CAP DATA, TREE DATA
weights='auto' #Weights for Logistic Regression algorithm
Estimators=100 #Number of estimators for Random Forest algorithm
Max_Depth=5 #Maximum depth parameter for Gradient Boosting algorithm
CAP_Features_priority = ['SURGE','ELEV','DIST_SHORE','TREE_DENSITY','DIST_PARK','DIST_WETS','SOIL_PERM','SOIL_FILL','ROW','COL'] #Define the features priority in CAP damage classification
TREE_Features_priority = ['SURGE','ELEV','TREE_DENSITY','DIST_SH','SOIL_PERM','SOIL_FILL','ROW','COL'] #Define the features priority in TREE damage classification
#----------------------------------------------------------------

#----------------------------------------------------------------
# Build the feature masks for dataframe to be classified
#----------------------------------------------------------------   
#CAP_Feat_Mask = np.zeros((Max_N_Feat[0],Max_N_Feat[0]-Min_N_Feat[0]+1),dtype='bool') #Inizialize the mask
#TREE_Feat_Mask = np.zeros((Max_N_Feat[1],Max_N_Feat[1]-Min_N_Feat[1]+1),dtype='bool') #Inizialize the mask

CAP_Feat_Mask = np.zeros(CAP_DF_P.shape,dtype='bool') #Inizialize the mask
TREE_Feat_Mask = np.zeros(TREE_DF_P.shape,dtype='bool') #Inizialize the mask

for ift in range(Min_N_Feat[0]-1, Max_N_Feat[0]):
    Col_location = CAP_DF_P.columns.get_loc(CAP_Features_priority[ift]) #Retrieve the location on dataframe corresopnding to the given priority feature
    CAP_Feat_Mask[ift:,Col_location] = 1 #building the mask including the above selected feature
#end
for ift in range(Min_N_Feat[1]-1, Max_N_Feat[1]):
    Col_location = TREE_DF_P.columns.get_loc(TREE_Features_priority[ift]) #Retrieve the location on dataframe corresopnding to the given priority feature
    TREE_Feat_Mask[ift:,Col_location] = 1 #building the mask including the above selected feature
#end
#----------------------------------------------------------------
    
#******************************************************************************
# Defining a set of dictionaries storing all the datasets and classifiers
#******************************************************************************

#------------------------------------------------
# Dataframes Names and corresponding Dataframes
#------------------------------------------------
Dataframes = {
              'CAP Damage': CAP_DF_P,
              'TREE Damage': TREE_DF_P,
              }
#------------------------------------------------
              
#------------------------------------------------
# Dataframes Names and corresponding field to be predicted
#------------------------------------------------
Drop_Feature = {
                'CAP Damage': 'COMBO',
                'TREE Damage': 'TREE_DMG',
                }
#------------------------------------------------
                
#------------------------------------------------
# Dataframes Names and corresponding min and max number
# of features to consider in classification
#------------------------------------------------               
DF_Feature_N = {
                'CAP Damage': {'Min_N_Feat': Min_N_Feat[0], 'Max_N_Feat': Max_N_Feat[0]},
                'TREE Damage': {'Min_N_Feat': Min_N_Feat[1], 'Max_N_Feat': Max_N_Feat[1]}
                }
#------------------------------------------------
                        
#------------------------------------------------
# Dataframes/Classifier combinations and
# corresponding feature mask and specific parameters
#------------------------------------------------               
Param_list = {
              'CAP Damage': {
                             'Logistic Regression': {'mask': CAP_Feat_Mask, 'weights': weights},
                             'Random Forest': {'mask': CAP_Feat_Mask, 'Estimators': Estimators},
                             'Gradient Boosting': {'mask': CAP_Feat_Mask, 'Max_Depth': Max_Depth}
                            },
              'TREE Damage': {
                             'Logistic Regression': {'mask': TREE_Feat_Mask, 'weights': weights},
                             'Random Forest': {'mask': TREE_Feat_Mask, 'Estimators': Estimators},
                             'Gradient Boosting': {'mask': TREE_Feat_Mask, 'Max_Depth': Max_Depth}
                             }
             }
#------------------------------------------------
                            
#------------------------------------------------
# Classifiers name and corresponding user-defined function to call
#------------------------------------------------
Classifiers = {
               'Logistic Regression': ML.LogReg,
               'Random Forest': ML.RanForest,
               'Gradient Boosting': ML.GradBoost
               }
#------------------------------------------------
               
#******************************************************************************
# Performing the classification
#******************************************************************************
               
Out_Struct = {} #initialize the output variable as a structure
        
for iDF, (Name_DF, DF) in enumerate (Dataframes.items()): #loop for all the elements in the Dataframe dictionary

    Out_Struct[Name_DF] = {} #initialize a substructure within the output variable
    Drop = Drop_Feature[Name_DF] #extract the name of the dataframe column that represents the classes that need to be predicted (it will be dropped from the dataset of predictors)
    
    if Standardize == 'True': #if standardization of the predictor features is required
        Scaler = SC()      
        X_DS = Scaler.fit_transform(DF.drop([Drop], axis=1).values) #extract the features dataset and standardize it
    else: #if no standardization is required
        X_DS = DF.drop([Drop], axis=1).values #extract the features dataset without standardization
    #end
    Y_DS = DF[Drop].values #Extracting the CAP damages for train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X_DS, Y_DS, test_size=0.25) #split the dataset into train and test subsets
            
    for iCL, (Name_CL, Classifier) in enumerate (Classifiers.items()): #loop for all elements in the classifier dictionary (all classifiers that will be used in the analysis)
        
        print 'Dataframe: ',Name_DF,' Classifier: ',Name_CL #print on the console the dataframe analyzed and the classifier used for the analysis
        Out_Struct[Name_DF][Name_CL] = {} #initialize a substructure withing the output variable

        Parameters = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "Min_N_Feat": Min_N_Feat, "Max_N_Feat": Max_N_Feat} #list the parameters to be passed to the classifying routines
        Parameters.update(DF_Feature_N[Name_DF]) #update the parameter list by adding the fields corresponding to the minimum and maximuym number of features to be considered in the classification routine 
        Parameters.update(Param_list[Name_DF][Name_CL]) #update the parameter list by adding the fields corresponding to the feature mask and the classification routine specific parameters
        Train_Out, Test_Out, Mask = Classifier(**Parameters) #call the classification routine with the parameter list
        
        P_Mat = np.zeros((DF_Feature_N[Name_DF]['Max_N_Feat']-DF_Feature_N[Name_DF]['Min_N_Feat']+1,2), dtype='float16') #initialize the performance matrix (it will contain 2 columns (mean and stdev) for each number of feature considered)
        labels = len(np.unique(Y_DS)) #extract the labels from the classification classes
        Conf_M = np.zeros((labels,labels,DF_Feature_N[Name_DF]['Max_N_Feat']-DF_Feature_N[Name_DF]['Min_N_Feat']+1), dtype='int') #initialize the confusion matrix for the classification problem
        
        for ift in range(DF_Feature_N[Name_DF]['Min_N_Feat']-1, DF_Feature_N[Name_DF]['Max_N_Feat']): #loop for the number of features that need to be included in the classification algorithm
        
            Perf = float(Test_Out[:,ift][Test_Out[:,ift]==y_test].size)/Test_Out[:,ift].size #calculate the percentage of items correctly predicted
            Std = np.std(Test_Out[:,ift]-y_test) #calculates the standard deviation of the predictions
            P_Mat[ift,0] = Perf #assign percentage of correct predicitons to the performance matrix
            P_Mat[ift,1] = Std #assign standard deviation of predicitons to the performance matrix
            Conf_M[:,:,ift] = CM(y_test, Test_Out[:,ift],np.unique(Y_DS)) #calls the confusion matrix routine with the test set and prediction set
            print (Perf, Std) #print the performance indicators on the console
            
        #end
            
        Out_Struct[Name_DF][Name_CL]['TrainSET'], Out_Struct[Name_DF][Name_CL]['TestSET'], Out_Struct[Name_DF][Name_CL]['Mask'] = Train_Out, Test_Out, Mask #append the predictions on the train set, test set and mask to the output structure
        Out_Struct[Name_DF][Name_CL]['Performance'] = P_Mat #append the performance matrix to the output structure
        Out_Struct[Name_DF][Name_CL]['Confusion_Matrix'] = Conf_M #append the confusion matrix to the output structure
        
    #end
    
    if Standardize == 'True': #if standardization of the predictor features is required
        X_train = Scaler.inverse_transform(X_train) #revert the standardization
        X_test = Scaler.inverse_transform(X_test) #revert the standardization
    #end
   
    Out_Struct[Name_DF]['X_Train'], Out_Struct[Name_DF]['X_Test'], Out_Struct[Name_DF]['Y_Train'], Out_Struct[Name_DF]['Y_Test'] = X_train, X_test, y_train, y_test #assign the train/test dataset to the output matrix

#end

#******************************************************************************
# Saving Results
#******************************************************************************
import pickle
with open(Path_02+'ML_Struct.pkl','wb') as handle:
    pickle.dump(Out_Struct,handle)
#end
print ('Output file successfully saved')
#******************************************************************************


