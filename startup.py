# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:10:23 2019

@author: Ganesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

startup = pd.read_csv("F:\\R\\files\\50_Startups.csv")
#create dummies of states
state = pd.get_dummies(startup.State)
state
state.columns
startup["California"] = state.California
startup["Florida"] = state.Florida
startup["Newyork"] = state["New York"]

startup = startup.drop(["State"], axis = 1)
startup = startup.rename(columns = {"R&D Spend": "RDSpend", "Marketing Spend" : "MSpend"})
#normalise the data
from sklearn import preprocessing

startup.iloc[:,0:4] = preprocessing.normalize(startup.iloc[:,0:4])

#find correlation
startup.corr()

#scatter plot between variables
import seaborn as sns
sns.pairplot(startup)
#collenearity problem is occur maximum between administration and marketing spend

#model building
import statsmodels.formula.api as sks

model1 = sks.ols("Profit~RDSpend +Administration+MSpend+California+Florida+Newyork", data = startup).fit()

model1.summary()

model1.params

#checking influencial measures

import statsmodels.api as sm

sm.graphics.influence_plot(model1)
# maximum influence values are 48 and 49
#
#startup_new = startup.drop(startup.index[[48,49]], axis = 0)
#
#model2 = sks.ols("Profit~RDSpend +Administration+MSpend+California+Florida+Newyork", data = startup_new).fit()
#
#model2.summary()

#calculating vif values

vif1 = sks.ols("RDSpend ~ Administration + MSpend + California + Florida + Newyork", data = startup).fit().rsquared
vif_RDS = 1/(1-vif1)
vif_RDS #1.35
vif2 = sks.ols("Administration ~ RDSpend + MSpend + California + Florida + Newyork", data = startup).fit().rsquared
vif_Adm = 1/(1-vif2)
vif_Adm #8.22
vif3 = sks.ols("MSpend ~ Administration + RDSpend + California + Florida + Newyork", data = startup).fit().rsquared
vif_MS = 1/(1-vif3)
vif_MS #7.83

sm.graphics.plot_partregress_grid(model1) #addedvariable plot

pred = model1.predict(startup)
pred.corr(startup.Profit) #prediction percentage is 78
