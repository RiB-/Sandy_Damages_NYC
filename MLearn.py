# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:17:41 2015

@author: Rfoti
"""

#******************************************************************************
#Importing packages
#******************************************************************************
import numpy as np #library for matrix-array analysis
#******************************************************************************

#******************************************************************************
def RepMissingAVG(a,ax=0): #Substitutes missing data in matrix with average of the corresponding colum (row)
#******************************************************************************
    import  scipy.stats as st #importing statistical package    
    mean = st.nanmean(a,axis=ax) #obtain mean of columns (rows) as needed, nanmean is just convenient
    inds = np.where(np.isnan(a)) #find indices that need to be replaced
    if ax==0:
        ind=1
    else:
        ind=0
    #end
    a[inds]=np.take(mean,inds[ind]) #place column (row) means in the indices, align the arrays using take
#end
#******************************************************************************
    
#******************************************************************************
def LogReg(X_train, X_test, y_train, y_test, Min_N_Feat, Max_N_Feat, mask='None',weights='auto'):
#******************************************************************************

    from sklearn.feature_selection import RFE #import the library to rank features with recursive feature elimination
    from sklearn.linear_model import LogisticRegression as LogR #import the Logistic Regression module
    
    if mask=='None':
        mask = np.zeros((Max_N_Feat-Min_N_Feat+1,int(X_train.shape[1])),dtype='bool') #define the mask to obtain the list of selected features
    #end
    Pred_Train = np.zeros((int(max(y_train.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
    Pred_Test = np.zeros((int(max(y_test.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
    
    print 'Logistic Regression: Training...' #notify the user about the status of the process    
    for ift in range(Min_N_Feat,Max_N_Feat+1): #iterate across the maximum number of features    
        LogReg_obj = LogR(C=1e3, class_weight=weights) #create the logistic regression model
        if mask=='None':
            rfe = RFE(LogReg_obj, ift) #create the RFE model and select the number of attributes
            rfe = rfe.fit(X_train,y_train) #train the RFE (feature selection) model on the train data sets
            mask[ift-Min_N_Feat,:] = rfe.support_ #apply the best feature mask to the output mask
        #end
        LogReg_obj.fit(X_train[:,mask[ift-Min_N_Feat,:]], y_train) #fit the logistic model to the train data sets
        Pred_Train[:,ift-1] = LogReg_obj.predict(X_train[:,mask[ift-Min_N_Feat,:]]) #apply the logistic model to the train dataset
        Pred_Test[:,ift-1] = LogReg_obj.predict(X_test[:,mask[ift-Min_N_Feat,:]]) #apply the logistic model to the test dataset
        print 'Logistic Regression: Predicting...', 100*ift/(Max_N_Feat-Min_N_Feat+1), '%' #notify the user about the status of the process 
    #end
        
    print 'Logistic Regression: Completed!' #notify the user about the status of the process
        
    return Pred_Train, Pred_Test, mask
    
#end
#******************************************************************************
   
#******************************************************************************
def RanForest(X_train, X_test, y_train, y_test, Min_N_Feat, Max_N_Feat, mask, Estimators=100):
#******************************************************************************

    from sklearn.ensemble import RandomForestClassifier as RFC #import library for machine learning analysis
    
    Pred_Train = np.zeros((int(max(y_train.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
    Pred_Test = np.zeros((int(max(y_test.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
    
    print 'Random Forest: Training...' #notify the user about the status of the process    
    for ift in range(Min_N_Feat,Max_N_Feat+1): #iterate across the maximum number of features
        Random_Forest_obj = RFC(n_estimators=Estimators) #call the Random Forest routing built in        
        Random_Forest_obj.fit(X_train[:,mask[ift-Min_N_Feat,:]], y_train) #fit the Random Forest algoritm to selected data
        Pred_Train[:,ift-1] = Random_Forest_obj.predict(X_train[:,mask[ift-Min_N_Feat,:]]).astype(int) #make predictions from trained algoritm using the test data
        Pred_Test[:,ift-1] = Random_Forest_obj.predict(X_test[:,mask[ift-Min_N_Feat,:]]).astype(int) #make predictions from trained algoritm using the test data
        print 'Random Forest: Predicting...', 100*ift/(Max_N_Feat-Min_N_Feat+1), '%' #notify the user about the status of the process
    #end
        
    print 'Random Forest: Completed!' #notify the user about the status of the process
    
    return Pred_Train, Pred_Test, mask

#end
#******************************************************************************

#******************************************************************************
def GradBoost(X_train, X_test, y_train, y_test, Min_N_Feat, Max_N_Feat, mask, Max_Depth=3):
#******************************************************************************

    from sklearn.ensemble import GradientBoostingClassifier as GBC #import library for machine learning analysis
    
    Pred_Train = np.zeros((int(max(y_train.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
    Pred_Test = np.zeros((int(max(y_test.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
    
    print 'Gradient Boosting: Training...' #notify the user about the status of the process    
    for ift in range(Min_N_Feat,Max_N_Feat+1): #iterate across the maximum number of features
        Gradient_Boosting_obj = GBC(max_depth=Max_Depth) #call the Random Forest routing built in        
        Gradient_Boosting_obj.fit(X_train[:,mask[ift-Min_N_Feat,:]], y_train) #fit the Random Forest algoritm to selected data
        Pred_Train[:,ift-1] = Gradient_Boosting_obj.predict(X_train[:,mask[ift-Min_N_Feat,:]]).astype(int) #make predictions from trained algoritm using the test data
        Pred_Test[:,ift-1] = Gradient_Boosting_obj.predict(X_test[:,mask[ift-Min_N_Feat,:]]).astype(int) #make predictions from trained algoritm using the test data
        print 'Gradient Boosting: Predicting...', 100*ift/(Max_N_Feat-Min_N_Feat+1), '%' #notify the user about the status of the process
    #end
        
    print 'Gradient Boosting: Completed!' #notify the user about the status of the process
    
    return Pred_Train, Pred_Test, mask

#end
#******************************************************************************
   
##******************************************************************************
#def RanForest(X_train, X_test, y_train, y_test, Min_N_Feat, Max_N_Feat, Estimators=100):
##******************************************************************************
#    from sklearn.feature_selection import RFE #import the library to rank features with recursive feature elimination  
#    from sklearn.ensemble import RandomForestClassifier as RFC #import library for machine learning analysis\
#    
#    mask=np.zeros((Max_N_Feat-Min_N_Feat+1,int(X_train.shape[1])),dtype=bool) #define the mask to obtain the list of selected features
#    Pred_Train = np.zeros((int(max(y_train.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
#    Pred_Test = np.zeros((int(max(y_test.shape)),Max_N_Feat-Min_N_Feat+1),dtype='int') #define the matrix of outputs (each prediction set is stored in a different column)
#    
#    print 'Random Forest: Training...' #notify the user about the status of the process    
#    for ift in range(Min_N_Feat,Max_N_Feat): #iterate across the maximum number of features  
#        Random_Forest_obj = RFC(n_estimators=Estimators) #call the Random Forest routing built in
#        rfe = RFE(Random_Forest_obj, ift+1) #create the RFE model and select the number of attributes
#        rfe = rfe.fit(X_train,y_train) #train the RFE (feature selection) model on the train data sets
#        mask[ift-Min_N_Feat,:] = rfe.support_ #apply the best feature mask to the output mask        
#        Random_Forest_obj.fit(X_train, y_train) #fit the Random Forest algoritm to selected data
#        Pred_Train[:,ift] = Random_Forest_obj.predict(X_train).astype(int) #make predictions from trained algoritm using the test data
#        Pred_Test[:,ift] = Random_Forest_obj.predict(X_test).astype(int) #make predictions from trained algoritm using the test data
#        print 'Random Forest: Predicting...', 100*(ift+1)/(Max_N_Feat-Min_N_Feat), '%' #notify the user about the status of the process
#    #end
#        
#    print 'Random Forest: Completed!' #notify the user about the status of the process
#    
#    return Pred_Train, Pred_Test, mask
#
##end
##******************************************************************************

