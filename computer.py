# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:42:01 2019

@author: Ganesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

computer = pd.read_csv("F:\\R\\files\\Computer_Data.csv")

computer.columns

# convert this catagorical variable cd, multi, premium to dummy variables
computer.cd = pd.get_dummies(computer.cd)
computer.multi =  pd.get_dummies(computer.multi)
computer.premium = pd.get_dummies(computer.premium)

computer = computer.drop(computer[["Unnamed: 0"]],axis = 1)

#normalizing the data
from sklearn import preprocessing

computer.iloc[:,0:5] = preprocessing.normalize(computer.iloc[:,0:5])
computer.iloc[:,8:10] = preprocessing.normalize(computer.iloc[:,8:10])

#to find correlation

computer.corr()
import seaborn as sns
sns.pairplot(computer)
#high collenearity problem between all variables
#model building
import statsmodels.formula.api as sm

model1 = sm.ols("price~ speed + hd + ram + screen + cd + multi + premium + ads + trend", data = computer).fit()

model1.summary()

#to find influence values

import statsmodels.api as s
s.graphics.influence_plot(model1)

#calculate vif values
m1 = sm.ols("speed ~ hd + ram + screen + cd + multi + premium + ads + trend", data = computer).fit().rsquared
vif_speed = 1/(1-m1)
vif_speed #1.73

m2 = sm.ols("hd ~ speed + ram + screen + cd + multi + premium + ads + trend", data = computer).fit().rsquared
vif_hd = 1/(1-m2)
vif_hd #3.87

m3 = sm.ols("ram ~ speed + hd + screen + cd + multi + premium + ads + trend", data = computer).fit().rsquared
vif_ram = 1/(1-m3)
vif_ram #2.48

m4 = sm.ols("screen ~ speed + hd + ram + cd + multi + premium + ads + trend", data = computer).fit().rsquared
vif_screen = 1/(1-m4)
vif_screen #1.56

m5 = sm.ols("ads ~ speed + hd + ram + cd + multi + premium + screen + trend", data = computer).fit().rsquared
vif_ads = 1/(1-m5)
vif_ads #14.95

m6 = sm.ols("trend ~ speed + hd + ram + cd + multi + premium + screen + ads", data = computer).fit().rsquared
vif_trend = 1/(1-m6)
vif_trend #23.58

#draw variable plot

s.graphics.plot_partregress_grid(computer)



#remove trend and build a model

model2 = sm.ols("price~ speed + hd + ram + screen + cd + multi + premium + ads", data = computer).fit()

model2.summary()

pred = model1.predict(computer)
pred.corr(computer.price) #0.97 % is correct
