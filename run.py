# imports
###########################
import cv2
import time
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from load_keras_model import build

###########################

# helper function to get one hot encoded labels

labels=pd.read_csv("train.csv")
labels_all=[]
for i in list(set(labels.color)):
    for j in list(set(labels.type)):
        labels_all.append(tuple([i,j]))
unique_labels=set(labels_all)        

unique_labels_list=list(unique_labels)
mlb = MultiLabelBinarizer()
mlb.fit(unique_labels_list)

# helper function for normalizing test data

def normalize_data(data):
    return data/127.5-1

# helper function for getting color and type

def get_color_type(pred_classes):
    if pred_classes[0] in color_list:
        color=pred_classes[0]
        type_dress=pred_classes[1]
    else:
        color=pred_classes[1]
        type_dress=pred_classes[0]
    return color,type_dress

color_list=set(labels.color)
dress_type=set(labels.type)

nb_rows=96
nb_cols=96
nb_channel=3



# predict("sample_df.csv","imgs/train/")

def predict(path_to_test_csv,imgs_directory="imgs/test/"):
    '''path_to_test_csv - test csv which will contain the test image uids
       returns - a dataframe containing the test image uid, type and color as 3 different columns
     '''
    
    ### batch predictions
    start_time=time.time()
    test_images=pd.read_csv(path_to_test_csv)
    test_len=len(test_images)
    print("test_images are {} in number".format(test_len))
    test_data = np.zeros((test_len,nb_rows, nb_cols, nb_channel)) 
    idx=0
    uid_list=[]
    color_list=[]
    type_list=[]
    for i in test_images.uid.values:  ## uid column
        image = cv2.imread(imgs_directory+str(i)+".jpg", cv2.IMREAD_COLOR)
        
        if image is None or image.shape==(0,0,0):
            continue
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(image.shape)
            resized = cv2.resize(image, (nb_rows,nb_cols), interpolation = cv2.INTER_AREA)
            resized_norm=normalize_data(resized)
            test_data[idx] = resized_norm
            idx+=1
            uid_list.append(str(i))
            
    #### loading model weights
    model=build(nb_rows,nb_cols,nb_channel,classes=len(mlb.classes_),act="sigmoid")
    model.load_weights("models/model-00001-0.05778-0.97879-0.00000-0.99802.h5")
    model_pred=model.predict(test_data)
    classes=[i[-2:] for i in np.argsort(model_pred)]
    pred_classes=[]
    for i in classes:
        pred_classes.append(mlb.classes_[i])
    
    for i in pred_classes:
        color,type_dress=get_color_type(i)
        color_list.append(color)
        type_list.append(type_dress)
    
    
    all_res=[uid_list,type_list,color_list]
    end_time=time.time()
    
    print("Time taken for {0} test images is {1} seconds".format(test_len,(end_time-start_time)))
    
    print('Predict function ends')
    
    return pd.DataFrame(list(zip(*all_res)),columns=["uid","type","color"])


