# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 12:00:10 2020

@author: fysikos6
"""
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50

from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
# load model
#model = ResNet50()
# summarize the model
#model.summary()
import os
import numpy as np
import matplotlib.pyplot as plt


folder="food/food/"
def load_images_from_folder(folder): # loads and preprocesses images for inputing
    images = []
    for filename in tqdm(os.listdir(folder)):
        img = load_img(r"food/food/" + filename, target_size= (224, 224)) 
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)
        if img is not None:
            images.append(img)
    return images

def from_triplets_to_pairs(dataframe):
    positive_pairs_df=dataframe[[0,1]].copy()
    negative_pairs_df=dataframe[[0,2]].copy()
    
    y_positives=np.ones(positive_pairs_df.shape[0],dtype=int)
    
    negative_pairs_df=negative_pairs_df.rename(columns={2:1})
    
    y_negatives=np.zeros(negative_pairs_df.shape[0],dtype=int)
    
    y_all=np.concatenate((y_positives,y_negatives))
    
    dataframes=[positive_pairs_df,negative_pairs_df]
    full_dataframe=pd.concat(dataframes,ignore_index=True)
    full_array=full_dataframe.to_numpy()
    return full_array ,y_all
    
#%%
#Import the txt data
folder="food/food/"


triples=pd.read_csv(r'train_triplets.txt',delimiter=" ",dtype=str,header=None)

train_triples, validation_triples=triples.iloc[:53563,:],triples.iloc[53563:,:]

test_triples=pd.read_csv(r'test_triplets.txt',delimiter=" ",dtype=str,header=None)

#make the triples into 'doubles'
train_pairs , y_train_pairs = from_triplets_to_pairs(train_triples)

#import all the images
images=load_images_from_folder("food/food/")
 
#%%
input1 = Input((224, 224, 3)) # the column with the "main"picture of the pairs?
input2 = Input((224, 244, 3)) # the column with the "comparison (related/unrelated, or positive and negative) picture of the pairs?

backbone = ResNet50(pooling='avg').output # what is the backbone?
 
features1 = backbone(input1) # backbone and input?
features2 = backbone(input2)

concat = Concatenate()([features1, features2]) # ??
dense = Dense(1)(concat)
out = Activation('sigmoid')(dense) 

# define a model with a list of two inputs
model = Model(inputs=[input1, input2], outputs=output) #this Model class describes what..?

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics='accuracy') #this I know
#%%
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True) #in the end it should run like this, or?

#%%
#%% Prediction
y_proba=model.predict_classes(X_test[:]) # here I should predict the result, one or zero from the test data
