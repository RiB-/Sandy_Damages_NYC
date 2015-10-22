# -*- coding: utf-8 -*-
"""
Created on Fri Apr 03 12:19:01 2015

@author: rfoti
"""

#******************************************************************************
#Importing packages
#******************************************************************************
import numpy as np #library for matrix-array analysis
import matplotlib.pyplot as plt #library to plot graphs
#******************************************************************************

#******************************************************************************
def Plot_confusion_matrix(CM, Labels, Norm='True', Cmap=plt.cm.Blues, Fig_counter=1, Title='Confusion matrix'):
#******************************************************************************
    #Plots the confusion matrix as a chessed graph with colors instead of numbers
    #INPUT: 1) Confusion Matrix
    #       2) Vector of labels
    #       3) Normalization of Confusion Matrix
    #       4) Colormap
    #       5) Title of plot
    #OUTPUT: Confusion matrix plot
#******************************************************************************
    if Norm == 'True':
        CM = CM.astype('float')/CM.sum(axis=0)[np.newaxis,:] #Normalize the matrix along the TRUE label axis
    #end
    plt.figure(Fig_counter)
    plt.imshow(CM, interpolation='nearest', cmap=Cmap) #create the graph and set the interpolation
    plt.title(Title) #adding the title
    plt.colorbar() #additing the colorbar
    if Norm == 'True':
       plt.clim(0,1) #Set the colorbar limits
    #end    
    tick_marks = np.arange(len(Labels)) #defininig the tick marks
    plt.xticks(tick_marks, Labels) #apply the labels to marks
    plt.yticks(tick_marks, Labels) #apply the labels to marks
#    plt.tight_layout() #setting the layout of the plot
    plt.ylabel('True label') #adding the y-axis title
    plt.xlabel('Predicted label') #adding the x-axis title
#end
#******************************************************************************

#******************************************************************************
def Plot_Pred_Labels(Rows, Cols, Y_Test, Y_Pred, Mask, Subwindows, n_labels=6, min_label=0, max_label=5, Colorlist='None', Background='False', Backgroundcolor='aqua', Nullcolor='lightgrey', Fig_counter=1, Title='None'):
#******************************************************************************    
    #Plots 2N sets of spatial images showing labels for test set and predicted set
    #INPUT: 1) Rows: Row coordiates of points
    #       2) Cols: Col coordiates of points
    #       3) Y_Test: labels for the test set
    #       4) Y_Pred: labels for the predicted set
    #       5) Mask: Spatial mask of the area
    #       6) Subwindows: subwindows of subarea to show individually
    #       7) Maximum, minimun and total number of labels
    #       8) List of colors to assign to individual labels
    #       9) Background: background image (land, for instance) as a binary matrix
    #       10) Background color: color to assign to the background
    #       11) Title: List of titles for sub-areas
    #OUTPUT: Plots of predicted and test classified points
#******************************************************************************
    #---------------------------------------
    #Handling dafault argument list
    #---------------------------------------      
    if Colorlist == 'None':
        Colorlist = ['blue','green', 'yellow','orange','orangered','crimson']
    #end
    #---------------------------------------
    rs = Mask.shape[0] #number of rows of the spatial window to be plotted
    cs = Mask.shape[1] #number of columns of the spatial window to be plotted
    Test_MAT = (min_label-1)*np.ones((rs,cs),dtype='int') #initialize the map for the test set (the null values will be assigned a value smaller by one than the smallest label)
    Pred_MAT = (min_label-1)*np.ones((rs,cs),dtype='int') #initialize the map for the prediction set (the null values will be assigned a value smaller by one than the smallest label)
    min_label = min_label-1 #update the minimun label in the dataset (the null label is really a dummy label)
    n_labels = n_labels+1 #update the number of labels in the dataset (the null label is really a dummy label)
    
    if Background != 'False': #if a background image is provided (needs to be a flat image with only one non-zero value for all non-zero cells)
        Background[Background<0] = 0 #adjust the format of the masking background in case it is provided with negative values
        Background[Background>0] = 1 #adjust the format of the masking background to set all the positive values to 1 (non strictly necessary)       
        Test_MAT[Background!=0] = min_label-1 #assign a dummy label to the map cells where background image is non-zero in the test set map (the background label will be assigned a value smaller by one than the null label)
        Pred_MAT[Background!=0] = min_label-1 #assign a dummy label to the map cells where background image is non-zero in the prediction set map (the background label will be assigned a value smaller by one than the null label)
        n_labels = n_labels+1 #updates the number of labels
        min_label = min_label-1 #update the minimum label
    #end
        
    for el in xrange(Y_Pred.shape[0]): #loop for all the elements in the prediction matrix
        Test_MAT[Rows[el],Cols[el]] = Y_Test[el] #assign the test set label to the given coordinate in the test map
        Pred_MAT[Rows[el],Cols[el]] = Y_Pred[el] #assign the prediction set label to the given coordinate in the prediction map
    #end   

    #***************************************
    # Plotting the test and prediction maps
    #***************************************
    import matplotlib as mpl #imports the plotting library    
    
    #---------------------------------------
    #Define the colormap
    #---------------------------------------    
    lscm = mpl.colors.LinearSegmentedColormap #assign a segmented colormap to a variable to produce discrete colors for each label    
    bounds = np.linspace(min_label-0.5,max_label+0.5,n_labels+1) # define the labels bins and normalize according to the number of labels
    if Background != 'False': #if a background image is provided
        Colorlist.reverse() #reverse the colorlist     
        Colorlist.extend([Backgroundcolor,Nullcolor]) #extend the colorlist to incorporate teh colors for the null color and for the background
        Colorlist.reverse() #reverse the colorlist 
    else:       
        Colorlist.reverse() #reverse the colorlist 
        Colorlist.extend([Nullcolor]) ##extend the colorlist to incorporate only the null color
        Colorlist.reverse() #reverse the colorlist 
    #end
    Colorlist = [mpl.colors.colorConverter.to_rgba(c) for c in Colorlist]
    mycm = lscm.from_list('mycm',Colorlist) #define a cololmap from the colorlist provided       
    cmaplist = [mycm(i) for i in range(mycm.N)] # extract all colors from the  map
    mycm = mycm.from_list('Custom cmap', cmaplist, mycm.N) # create the new map
    norm = mpl.colors.BoundaryNorm(bounds, mycm.N) #maps the colormap to individual labels (numbers)
    mycm.set_bad(Nullcolor) #set "bad data" as the nullcolor
    
    fc=Fig_counter #initialize the figure counter
    for nw in xrange(Subwindows.shape[0]): #loops across the number of windows that need to be plotted
        plt.figure(fc) #initialize figure for test set plot
        if Title!='None': #if title list is provided
            Title_fig = Title[nw]+' Test' #assign the title to the test set plot
            plt.title(Title_fig) #insert the title into the subplot
        else:
            Title_fig = 'Test'
        #end
        plt.title(Title_fig) #insert the title into the subplot
        TM = Test_MAT[Subwindows[nw,0]:Subwindows[nw,1]+1,Subwindows[nw,2]:Subwindows[nw,3]+1] #extracts the test set window
        imgplot = plt.imshow(TM, cmap=mycm, norm=norm) #plots the figure and applies the colormap and normalization
        imgplot.set_clim(min_label-0.5,max_label+0.5)
        plt.colorbar().set_ticks(np.linspace(min_label,max_label,n_labels)) #assign the colorbar ticks       
        fc = fc+1 #updates the figure counter
        
        plt.figure(fc) #initialize figure for prediction set plot
        if Title!='None': #if title list is provided
            Title_fig = Title[nw]+' Predict' #assign the title to the test set plot
            plt.title(Title_fig) #insert the title into the subplot
        else:
            Title_fig = 'Predict'
        #end
        plt.title(Title_fig) #insert the title into the subplot
        PM = Pred_MAT[Subwindows[nw,0]:Subwindows[nw,1]+1,Subwindows[nw,2]:Subwindows[nw,3]+1] #extracts the prediction set window
        imgplot = plt.imshow(PM, cmap=mycm, norm=norm) #plots the figure and applies the colormap and normalization
        imgplot.set_clim(min_label-0.5,max_label+0.5)
        plt.colorbar().set_ticks(np.linspace(min_label,max_label,n_labels)) #assign the colorbar ticks 
        fc = fc+1  #updates the figure counter
    #end
        
#end

#******************************************************************************
def Plot_Predicted_Field(Matrix, Mask, n_labels=6, min_label=0, max_label=5, Colorlist='None', Background='False', Backgroundcolor='aqua', Nullcolor='lightgrey', Fig_counter=1, Title='None'):
#******************************************************************************    
    #Plots 2N sets of spatial images showing labels for test set and predicted set
    #INPUT: 1) Matrix: matrix of labels
    #       2) Mask: Spatial mask of the area
    #       3) Subwindows: subwindows of subarea to show individually
    #       4) Maximum, minimun and total number of labels
    #       5) List of colors to assign to individual labels
    #       6) Background: background image (land, for instance) as a binary matrix
    #       7) Background color: color to assign to the background
    #       8) Title of Figure
    #OUTPUT: Plot of predicted field
#******************************************************************************

    #---------------------------------------
    #Handling dafault argument list
    #---------------------------------------      
    if Colorlist == 'None':
        Colorlist = ['blue','green', 'yellow','orange','orangered','crimson']
    #end
    #---------------------------------------
    min_label = min_label-1 #update the minimun label in the dataset (the null label is really a dummy label)
    n_labels = n_labels+1 #update the number of labels in the dataset (the null label is really a dummy label)
    
    if Background != 'False': #if a background image is provided (needs to be a flat image with only one non-zero value for all non-zero cells)
        Background[Background<0] = 0 #adjust the format of the masking background in case it is provided with negative values
        Background[Background>0] = 1 #adjust the format of the masking background to set all the positive values to 1 (non strictly necessary)       
        Matrix[Background!=0] = min_label-1 #assign a dummy label to the map cells where background image is non-zero in the test set map (the background label will be assigned a value smaller by one than the null label)
        n_labels = n_labels+1 #updates the number of labels
        min_label = min_label-1 #update the minimum label
    #end

    #***************************************
    # Plotting the test and prediction maps
    #***************************************
    import matplotlib as mpl #imports the plotting library    
    
    #---------------------------------------
    #Define the colormap
    #---------------------------------------    
    lscm = mpl.colors.LinearSegmentedColormap #assign a segmented colormap to a variable to produce discrete colors for each label    
    bounds = np.linspace(min_label-0.5,max_label+0.5,n_labels+1) # define the labels bins and normalize according to the number of labels
    if Background != 'False': #if a background image is provided
        Colorlist.reverse() #reverse the colorlist     
        Colorlist.extend([Backgroundcolor,Nullcolor]) #extend the colorlist to incorporate teh colors for the null color and for the background
        Colorlist.reverse() #reverse the colorlist 
    else:       
        Colorlist.reverse() #reverse the colorlist 
        Colorlist.extend([Nullcolor]) ##extend the colorlist to incorporate only the null color
        Colorlist.reverse() #reverse the colorlist 
    #end
    Colorlist = [mpl.colors.colorConverter.to_rgba(c) for c in Colorlist]
    mycm = lscm.from_list('mycm',Colorlist) #define a cololmap from the colorlist provided       
    cmaplist = [mycm(i) for i in range(mycm.N)] # extract all colors from the  map
    mycm = mycm.from_list('Custom cmap', cmaplist, mycm.N) # create the new map
    norm = mpl.colors.BoundaryNorm(bounds, mycm.N) #maps the colormap to individual labels (numbers)
    mycm.set_bad(Nullcolor) #set "bad data" as the nullcolor
    
    fc=Fig_counter #initialize the figure counter
    plt.figure(fc) #initialize figure for test set plot
    if Title!='None': #if title list is provided
        Title_fig = Title #assign the title to the test set plot
        plt.title(Title_fig) #insert the title into the subplot
    else:
        Title_fig = 'Predicted Sandy Damage'
    #end
    plt.title(Title_fig) #insert the title into the subplot
    imgplot = plt.imshow(Matrix, cmap=mycm, norm=norm, interpolation='none') #plots the figure and applies the colormap and normalization
    imgplot.set_clim(min_label-0.5,max_label+0.5)
    plt.colorbar().set_ticks(np.linspace(min_label,max_label,n_labels)) #assign the colorbar ticks       
    fc = fc+1 #updates the figure counter
        
#end

