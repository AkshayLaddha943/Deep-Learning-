# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:17:25 2020

@author: Admin
"""

import numpy as np
import pandas as pd
import xlsxwriter
from sklearn.metrics import r2_score

Dataframe = pd.read_csv("C:\\Users\\Admin\\Desktop\\IP_LIVE_PROJECT_ Artificial_Intelligence_Akshay_Laddha_1254\\Project code\\AI-DataTrain.csv", encoding = 'latin1')
Dataframe = Dataframe.drop('Unnamed: 0' , axis = 1)


#df_train_x =  Dataframe.iloc[0:900, :]
df_y = np.array([])

            
for column in Dataframe:
    b = np.array([Dataframe[column].value_counts(sort=False)[0] / Dataframe.shape[0]])
    for i in b:
        print(i)
        if i>0.1 and i<0.3:
           df_y = np.append(df_y,i)
        elif i>0.4 and i<0.6:
            df_y = np.append(df_y,i)
        elif i>0.7 and i<0.9:
            df_y = np.append(df_y,i)
        else:
            df_y = np.append(df_y,i)
            
Dataframe = Dataframe.T
df_y = np.reshape(df_y, (50,1))

n_h = 20
output_size = 1

class neural_network(): 
    
          def __init__(self): 

              self.W1 = np.random.randn(Dataframe.shape[1], n_h)
              self.W2 = np.random.randn(n_h, output_size) 
              self.lr=0.3
      
          def forward(self,X):
              self.z = np.dot(X, self.W1)
              self.z2 = self.sigmoid(self.z)
              self.z3 = np.dot(self.z2, self.W2)
              o = self.sigmoid(self.z3)
              return o
      
          def sigmoid(self, s):
              return 1/(1+np.exp(-s))

          def sigmoidPrime(self, s):
              return s * (1 - s)
          

          def backward(self, X, y, o):
              self.o_error = y - o
              self.o_delta = self.o_error*self.sigmoidPrime(o)
              self.z2_error = self.o_delta.dot(self.W2.T)
              self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
              self.dW2 = np.dot(self.z2.T, (self.o_delta))
              self.dW1 = np.dot(X.T,  (np.dot(self.o_delta, self.W2.T) * self.sigmoidPrime(self.z2)))            
              self.W1 = self.W1 + self.lr * self.dW1
              self.W2 = self.W2 + self.lr * self.dW2 
          

          def train(self, X, y):
              o = self.forward(X)
              self.backward(X, y, o)          
          
nn = neural_network()
for i in range(3500):
     print("Input\n" + str(Dataframe))
     print("Actual Output: \n" + str(df_y))
     p = str(nn.forward(Dataframe))
     print("Predicted Output: \n" + p)
     print("Loss: \n" + str(np.mean(np.square(df_y - nn.forward(Dataframe)))))
     print((r2_score(nn.forward(Dataframe),df_y)) * 100)
     print("\n")
     nn.train(Dataframe, df_y)
    
a = p.splitlines()

a_list = []
for c in a:
        a_list.append(str(c)[2:-2])
        
        
workbook = xlsxwriter.Workbook('C:\\Users\\Admin\\Desktop\\IP_LIVE_PROJECT_ Artificial_Intelligence_Akshay_Laddha_1254\\Project code\\output.xlsx') 
worksheet = workbook.add_worksheet() 
row = 0
row_2 = 0
column = 0
column_weight = 1
for c,d in Dataframe.iterrows():
    worksheet.write(row, column, c)
    row += 1
  
for pol in a_list:
    worksheet.write(row_2, column_weight, float(pol))
    row_2 += 1         
workbook.close()
   




         

     