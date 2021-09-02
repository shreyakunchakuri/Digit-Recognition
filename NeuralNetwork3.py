import numpy as np
import scipy as sc
import csv
import pandas as pd
import time
import random as r;
import math


start=time.time();



def softmax(x):
    
    exponential_value=np.exp(x)
    value=exponential_value/np.sum(exponential_value,axis=0);
    
    return value;
    
    #Softmax function 
    print("softmax function");
  

def sigmoid(x):
    if len(x)==1:
        return 1.0/(1+np.exp(-x))
    v=[]
    for i in range(len(x)):
        v.append(1.0/(1+np.exp(-x[i])))
        
    v=np.array(v)
    return v

def sigmoid_derivative(x):
    if len(x)==1:
        f=sigmoid(x)
        return  f*(1-f)
    f=sigmoid(x)
    for i in range(len(x)):
        f[i]=f[i]*(1-f[i])
    return f
    
def ANN(bat,weights,biases):
    
    feed_forward={};
    forw={};
    batch=bat[:,:784]
         
    cal_val1=weights["w1"].dot(batch.transpose())+biases["b1"];
         
    feed_forward["hl1"]=np.array(sigmoid(cal_val1));
        
    cal_val2=weights["w2"].dot(feed_forward["hl1"])+biases["b2"];
        
    feed_forward["hl2"]=np.array(sigmoid(cal_val2));
        
    cal_val3=weights["w3"].dot(feed_forward["hl2"])+biases["b3"];
        
    forw["ol"]= np.array(softmax(cal_val3));
    
    return forw;
    
def cross_entropy(real, obs):
    """
    compute loss function
    """
    print(real.shape,(np.log(obs)).shape)
    L_sum = np.sum(np.multiply(real, np.log(obs)))
    m = real.shape[1]
    L = -(1./m) * L_sum

    return L   
    
    
if __name__ == "__main__":


    with open('train_image.csv',newline='\n') as f:
            
         csv_reader=csv.reader(f);
         #train_images=[]
         #train_images=list(csv_reader);
         train_images = [[int(string) for string in inner_list] for inner_list in csv_reader]
    f.close();
    """
    ti=pd.read_csv("train_image.csv",header=None);
    train_images=ti.values.tolist();
    """
    
    with open('train_label.csv',newline='\n') as f:
        
         f_reader=csv.reader(f); 
         train_labels=[int(i[0]) for i in f_reader];
     
    f.close();    
    #tl=pd.read_csv("train_label.csv");
    #train_labels=[int(i) for i in tl]
    
    with open('test_image.csv',newline='\n') as f:
            
         csv_reader=csv.reader(f);
         #train_images=[]
         #train_images=list(csv_reader);
         test_images = [[int(string) for string in inner_list] for inner_list in csv_reader]
     
    f.close();    
    """     
    tei=pd.read_csv("test_image.csv",header=None);
    test_images=tei.values.tolist();
    """
    
    #tel=pd.read_csv("test_label.csv");
    #test_labels=tel.values.tolist();
    with open('test_label.csv',newline='\n') as f:
        
         f_reader=csv.reader(f); 
         test_labels=[int(i[0]) for i in f_reader];
         
    f.close();
     
         
    train_images=np.array(train_images);
    train_labels=np.array(train_labels);
    train_labels=np.reshape(train_labels,(len(train_labels),1))
    test_images=np.array(test_images);
    test_labels=np.array(test_labels);
    test_labels=np.reshape(test_labels,(len(test_labels),1))
    
    train_labels_new=np.zeros((60000,10));
    for i in range(len(train_labels)):
         train_labels_new[i][train_labels[i][0]]=1
         
    test_labels_new=np.zeros((10000,10));    
    for i in range(len(test_labels)):
         test_labels_new[i][test_labels[i][0]]=1
         
    train_data=np.hstack((train_images,train_labels_new));
    test_data=np.hstack((test_images,test_labels_new))
   
    eta=0.01;
    n_batches=50;
    batch_size=300;
    temp1=0
    temp2=300;


        
    c_size=len(train_data[0])-10    
        
    weights={       "w1":np.random.randn(256,c_size)*(1/math.sqrt(c_size)) ,
                    "w2":np.random.randn(128,256)*(1/math.sqrt(256)) ,
                    "w3":np.random.randn(10,128)*(1/math.sqrt(128)) 
            }
    
    biases={        "b1":np.random.randn(256,1)*(1/np.sqrt(c_size)),
                    "b2":np.random.randn(128,1)*(1/np.sqrt(256)),
                    "b3":np.random.randn(10,1)*(1/np.sqrt(128))    
           }
    
    count=0;
    for ep in range(200):
            
            
            temp1=0
            temp2=300;
            print(ep)
            batches=[];
            np.random.shuffle(train_data);
            for i in range(0,n_batches):
                batches.append(train_data[temp1:temp2])
                temp1=temp2
                temp2=temp2+batch_size;
            for bat in batches:
                #------------------------------------------------------ 
                feed_forward={};
                batch=bat[:,:784]
                 
                cal_val1=weights["w1"].dot(batch.transpose())+biases["b1"];
                 
                feed_forward["hl1"]=np.array(sigmoid(cal_val1));
                
                cal_val2=weights["w2"].dot(feed_forward["hl1"])+biases["b2"];
                
                feed_forward["hl2"]=np.array(sigmoid(cal_val2));
                
                cal_val3=weights["w3"].dot(feed_forward["hl2"])+biases["b3"];
                
                feed_forward["ol"]= np.array(softmax(cal_val3));
                #---------------------------------------------------------
                
                target=bat[:,784:];
                #target=np.reshape(target,(len(target),1));
                
                dz3=feed_forward["ol"]-target.transpose();
                
                dw3=(1.0/batch_size)*dz3.dot(feed_forward["hl2"].transpose());
                db3=(1.0/batch_size)*np.sum(dz3,axis=1,keepdims=True);
                
                
                dhl2=weights["w3"].transpose().dot(dz3);
                dz2=dhl2*sigmoid(cal_val2)*(1-sigmoid(cal_val2));
                
                dw2=(1.0/batch_size)*(dz2.dot(feed_forward["hl1"].transpose()))
                db2=(1.0/batch_size)*np.sum(dz2,axis=1,keepdims=True)
                
                
                dhl1=weights["w2"].transpose().dot(dz2);
                dz1=dhl1*sigmoid(cal_val1)*(1-sigmoid(cal_val1))
                
                
                dw1=(1.0/batch_size)*(dz1.dot(batch))
                db1=(1.0/batch_size)*np.sum(dz1,axis=1,keepdims=True)
               
                #---------------------------------------------------------
                
                weights["w1"]=weights["w1"]-eta*dw1;
                weights["w2"]=weights["w2"]-eta*dw2;
                weights["w3"]=weights["w3"]-eta*dw3;
                
                biases["b1"]=biases["b1"]-eta*db1;
                biases["b2"]=biases["b2"]-eta*db2;
                biases["b3"]=biases["b3"]-eta*db3;
    forward={}    
            
    forward = ANN(train_data,weights,biases);
    real_target=train_data[:,784:]
    train_error=cross_entropy(real_target.transpose(),forward["ol"])
    
    forward={}        
    forward=ANN(test_data,weights,biases);
    real_target=test_data[:,784:]
    test_error=cross_entropy(real_target.transpose(),forward["ol"])
    
    
    for test in range(10000):
        
        actual=np.argmax(forward["ol"][:,test]);
        expected=test_labels[test][0]
        
        if (actual==expected):
            count=count+1
            
            
end=time.time();
print(end-start);
            