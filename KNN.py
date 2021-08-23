import numpy as np 
import cv2 
import os
import pandas as pd  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import seaborn as sn
import scipy
from collections import Counter
import imageio 
import re
import glob
import operator

#Training_Set

#load training data

path=os.fspath(r'E:\NileUnivercity\MACHINE LEARING\lectures\week 8\Assignment 3\Train')




def sorted_( n ):
 
    x = lambda text: int(text) if text.isdigit() else text
    y = lambda key: [x(c) for c in re.split('([0-9]+)', key)]
    return sorted(n, key = y)

projectFiles = glob.glob( os.path.join(path, '*.jpg') )


training_sorted = sorted_([str(x) for x in projectFiles])


training_images=[cv2.imread(os.path.join(path,filename),cv2.IMREAD_GRAYSCALE)/255 for filename in training_sorted]




training_images = np.array(training_images)
flattend_images= training_images.reshape(2400,784)
Label =pd.read_csv('Training Labels.txt',header=None).values




       
    


#distance_matrix
flattend_images_copy=flattend_images

distance=[]
for i in range(2400):
    dd=[]
    for j in flattend_images_copy:
        d=np.linalg.norm(np.array(flattend_images[i])-np.array(j))
        dd.append(d)
    distance.append(dd)
        

distance=np.array(distance) 

distance_index=np.argsort(distance,axis=1)

    



label_values=[]
for i in distance_index:
    f=[]
    for j in i:
        f.append(operator.getitem(Label,j))
    label_values.append(f)



label_values=np.array(label_values)        
label_values=label_values.reshape(2400,2400)




label_values1=[]
for i in label_values:
    a=[]
    for j in i:
        a.append(j)
    label_values1.append(a)



for i in range(2400):
    
    del label_values1[i][0]
    

label_values1=np.array(label_values1)


#LOO_for_K_from_1_to_100

o=[]
for k in range(1,101):
    z=[]
    for i in label_values1[:,:k]:
        q=[]
        for j in i:
            q.append(j)
        w=Counter(q).most_common(1)[0][0]
        z.append(w)
    o.append(z)

o=np.array(o)


Error_num_for_each_K=[]
for k in range(100):   
    error=0    
    for i in range(2400):
        if o[k][i]!=Label[i]:
            error=error+1
    Error_num_for_each_K.append(error)
    
#Best_K    
Best_k=Error_num_for_each_K.index(min(Error_num_for_each_K))+1


# matrix Accurcy
from sklearn.metrics import accuracy_score
print(accuracy_score(Label, o[3])*100)
    
        
        
#plot k with Error   
 
k_value=[]
for i in range(1,101):
    k_value.append(i)
    
    
plt.plot(k_value,Error_num_for_each_K)
plt.xlabel("K_value")
plt.ylabel("Error_Number")
plt.show()



#Testing_Set



#load test data

path_t=os.fspath(r'E:\NileUnivercity\MACHINE LEARING\lectures\week 8\Assignment 3\Test')




def sorted_( n ):
 
    x = lambda text: int(text) if text.isdigit() else text
    y = lambda key: [x(c) for c in re.split('([0-9]+)', key)]
    return sorted(n, key = y)
projectFiles_t = glob.glob( os.path.join(path_t, '*.jpg') )


training_sorted_t = sorted_([str(x) for x in projectFiles_t])


training_images_t=[cv2.imread(os.path.join(path_t,filename),cv2.IMREAD_GRAYSCALE)/255 for filename in training_sorted_t]




training_images_t = np.array(training_images_t)
flattend_images_t= training_images_t.reshape(200,784)
Label_t =pd.read_csv('Test Labels.txt',header=None).values


#distance_matrix



distance_t=[]
for i in range(200):
    dd=[]
    for j in flattend_images:
        d=np.linalg.norm(np.array(flattend_images_t[i])-np.array(j))
        dd.append(d)
    distance_t.append(dd)
        

distance=np.array(distance) 

distance_index_t=np.argsort(distance_t,axis=1)

    



label_values_t=[]
for i in distance_index_t:
    
    f=[]
    for j in i:
        f.append(operator.getitem(Label,j))
    label_values_t.append(f)



label_values_t=np.array(label_values_t)        
label_values_t=label_values_t.reshape(200,2400)


#KNN_for_test

z_t=[]
for i in label_values_t[:,:4]:
    q=[]
    for j in i:
        q.append(j)
    w=Counter(q).most_common(1)[0][0]
    z_t.append(w)
    

P_t=np.array(z_t)


# confution matrix    
from sklearn.metrics import confusion_matrix
conf_array=confusion_matrix(Label_t, P_t)

# matrix Accurcy
from sklearn.metrics import accuracy_score
print(accuracy_score(Label_t, P_t)*100)


# Convert the confusion matrix to an image


cm_img = pd.DataFrame(conf_array)
sn.heatmap(cm_img, annot=True)