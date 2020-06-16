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
#from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50

from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Activation
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
        img = img_to_array(img, dtype=int)
        #img = img.reshape((1,) + img.shape)
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


# def images_list_for_x(train_pairs,images):
#     anchorColumn=np.empty(len(train_pairs))
#     comparisonColumn=np.empty(len(train_pairs))
#     for index,row in tqdm(enumerate(train_pairs)):
#         anckor= images[int(train_pairs[index,0])]
#         comparison= images[int(train_pairs[index,1])]
#         anchorColumn=np.append(anchorColumn,anckor)
        
#         comparisonColumn=np.append(comparisonColumn,comparison)
        
#     return anchorColumn, comparisonColumn

def images_list_for_x(train_pairs,images):
    anchorColumn=[]
    comparisonColumn=[]
    for index,row in tqdm(enumerate(train_pairs)):
        anckor= images[int(train_pairs[index,0])]
        comparison= images[int(train_pairs[index,1])]
        anchorColumn.append(anckor)        
        comparisonColumn.append(comparison)      
        
    anchorColumnarray=np.array(anchorColumn)    
    comparisonColumnarray=np.array(comparisonColumn)
    return anchorColumnarray, comparisonColumnarray

def filter_triples(df, maxval):
    k = [df[col].astype(int) < maxval for col in df]
    return df[k[0] & k[1] & k[2]]
   
   
#%%
#Import the txt data
folder="food/food/"
max_img = 1280 #maximum number of images we allow every time

# Load in the data
triples=pd.read_csv(r'train_triplets.txt',delimiter=" ",dtype=str,header=None)
test_triples=pd.read_csv(r'test_triplets.txt',delimiter=" ",dtype=str,header=None)
#train_triples, validation_triples=triples.iloc[:53563,:],triples.iloc[53563:,:]


# Truncate
#keep = (triples[0].astype(int) < max_img) & (triples[1].astype(int) < max_img) & (triples[2].astype(int) < max_img)
#print('Keep {} rows'.format(np.sum(keep)))
#triples = triples.loc[keep,:]
triples = filter_triples(triples, max_img)
test_triples = filter_triples(test_triples, max_img)

# Split into test, validation
train_len = triples.shape[0] // 2 # define the percentage of data to be used for training
train_triples, validation_triples=triples.iloc[:train_len,:],triples.iloc[train_len:,:]



#make the triples into 'doubles'
train_pairs , y_train_pairs = from_triplets_to_pairs(train_triples)
test_pairs, y_test_pairs=from_triplets_to_pairs(test_triples)
validation_pairs, y_validation_pairs= from_triplets_to_pairs(validation_triples)



#make the triples into 'doubles'
train_pairs , y_train_pairs = from_triplets_to_pairs(train_triples)
test_pairs, y_test_pairs=from_triplets_to_pairs(test_triples)
validation_pairs, y_validation_pairs= from_triplets_to_pairs(validation_triples)

#import all the images
images=load_images_from_folder("food/food/")

anchor_list, comparison_list=images_list_for_x(train_pairs,images)
anchor_list_validation, compraison_list_validation=images_list_for_x(validation_pairs,images)

anchor_list_test, compraison_list_test= images_list_for_x(test_pairs,images)
 
#%%
input1 = Input((224, 224, 3), name="BackboneNet_input1") # the column with the "main"picture of the pairs?
input2 = Input((224, 224, 3), name="BackboneNet_input2") # the column with the "comparison (related/unrelated, or positive and negative) picture of the pairs?

# base_model=ResNet50()
backbone = ResNet50(pooling='avg') #.output # what is the backbone?

for layer in backbone.layers[:-2]:
    layer.trainable= False
 
features1 = backbone(input1) # pretrained model input1
features2 = backbone(input2) # pretrained model input2



concat = Concatenate()([features1, features2]) # ??
dense = Dense(1)(concat)
output = Activation('sigmoid')(dense) 

# define a model with a list of two inputs
model = Model(inputs=[input1, input2], outputs=output) #this Model class describes what..?

# model.summary()
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy']) #this I know
model.summary()

# for layer in backbone.layers:
#     print(layer,layer.trainable)
#%%

# initialize the number of epochs and batch size
EPOCHS = 10
BS = 32 #batch size

# def load_image(filename): # loads and preprocesses images for inputing
    
#     img = load_img(r"food/food/" + filename, target_size= (224, 224)) 
#     img = img_to_array(img, dtype=int)
#         #img = img.reshape((1,) + img.shape)
        
#     return img
            
# def images_on_the_fly(train_pairs,y,batch_size):
#     aug = ImageDataGenerator()
#     genX2 = aug.flow(train_pairs[:,0],train_pairs[:,1], batch_size=batch_size,seed=666)
#     genX1 = aug.flow(train_pairs[:,0],y,  batch_size=batch_size,seed=666)

#     while True:
#             X1i = load_image(genX1.next()+".jpg")
#             X2i = load_image(genX2.next()+".jpg")
#             #Assert arrays are equal - this was for peace of mind, but slows down training
#             #np.testing.assert_array_equal(X1i[0],X2i[0])
#             yield [X1i[0], X2i[1]], X1i[1]
    
      
def gen_flow_for_two_inputs(X1, X2, y, batch_size):
    # construct the training image generator for data augmentation
    aug = ImageDataGenerator()
    genX2 = aug.flow(X1,X2, batch_size=batch_size,seed=2)
    genX1 = aug.flow(X1,y,  batch_size=batch_size,seed=2)
    
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]
            
gen_flow = gen_flow_for_two_inputs(anchor_list,comparison_list,y_train_pairs,BS)
validation_flow= gen_flow_for_two_inputs(anchor_list_validation,compraison_list_validation,y_validation_pairs,BS)

H=model.fit_generator(gen_flow, validation_data=validation_flow , steps_per_epoch=len(anchor_list,) // BS, validation_steps= len(anchor_list_validation)//BS,epochs=EPOCHS)

#model.fit_generator([anchor_list,comparison_list], y_train_pairs, batch_size=batch_size, epochs=epochs)#, validation_data=(x_test, y_test), shuffle=True) #in the end it should run like this, or?
#%%
#%% Prediction
y_proba=model.predict_classes(X_test[:]) # here I should predict the result, one or zero from the test data
