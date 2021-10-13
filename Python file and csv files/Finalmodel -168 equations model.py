# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 12:30:35 2019

@author: Case Study 1 Group
"""
#This block is to import important libraries to be used later for data manipulation
import pandas as pd
from pandas import DataFrame
import numpy as np

#This line is to import data from the excel file with the historic values of all dependent and independent variables 
combineddataset = pd.read_csv(r'C:/Users/ASUS/.spyder-py3/Historic Data.csv')
#This line tells python about the columns available in the csv file
combineddataset=DataFrame(combineddataset,columns=['SerialNumber','Year','Hours','Filtered Price','Demand','Hydropower','Wind','Photovoltaics',
                                                   'Import','Export','Coal Price','Neutral Gas Price Index','NetConnect Germany Gas Market',
                                                   'CO2 Certificates','Currency US-EU','Availability'])
print(combineddataset)
#This line is to make python read the values only upto a certain serial number in the excel file
combineddatsetyear=combineddataset[combineddataset.SerialNumber<17833]
print(combineddatsetyear)
#Even though the dataset contains many independent variables, after a few analysis only following variables were used for modelling
useddataset=combineddatsetyear[['SerialNumber','Year','Hours','Filtered Price','Demand','Wind','Photovoltaics',
                                                   'Coal Price',
                                                   'CO2 Certificates','Availability']]
#This line is to create the title row for the output excel file
np_finalsolution=np.array(['intercept','Demand','Wind','Photovoltaics',
                                                   'Coal Price',
                                                   'CO2 Certificates','Availability','Price forecast'])
                                                   
#This 'for' loop is to filter every hour of a week separately. The loop runs separately for each hour of a week and predicts the intercepts and coefficients
for x in range(1,169): 
    useddataset_x=useddataset[useddataset.Hours==x] #This line is to take out only the price data at hour x, hour 1 means price at 1st hour of monday
#This block finds the intercept and regression coefficients
    from sklearn import linear_model
    import statsmodels.api as sm
#Following two lines declares the independent X variables and dependent Y variable to python 
    X = useddataset_x[['Demand','Wind','Photovoltaics',
                     'Coal Price',
                      'CO2 Certificates','Availability']]
    Y = useddataset_x['Filtered Price']
#Following four lines fits a regression line and prints the intercept and coefficient for every hour
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    
#This block is to predict the price using the predicted values of fundamental and exogenous variables taken from the csv file 'futurevalues_targetweek_1.csv'
    exovariables=pd.read_csv(r'C:/Users/ASUS/.spyder-py3/futurevalues_targetweek_1.csv')
    exovariables=DataFrame(exovariables,columns=['Hour','Demand','Wind','Photovoltaics',
                     'Coal Price',
                      'CO2 Certificates','Availability'])
    print (exovariables)
    exovariables_x=exovariables[exovariables.Hour==x]
    predicteddata=exovariables_x[['Demand','Wind','Photovoltaics',
                   'Coal Price',
                      'CO2 Certificates','Availability']]
    print ('Predicted  Price: \n', regr.predict(predicteddata))
    
    #This block is to stack the coefficients,intercepts and predicted prices obtained for each hour into one single array of 168 rows which can later be exported to an excel file
    coefficient_x=regr.coef_
    intercept_x=regr.intercept_
    predictedprice_x=regr.predict(predicteddata)
    np_stackeddata=np.hstack((intercept_x,coefficient_x,predictedprice_x))
    print(np_stackeddata)
    np_finalsolution=np.vstack((np_finalsolution,np_stackeddata))
    # This block use statsmodels to predict the price directly which also prints certain important statistical parameters like p-value, standard error etc.
    X = sm.add_constant(X) # adding a constant
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X) 
    print_model = model.summary()
    print (x)
    print(print_model)
    #the 'for' loop ends here

#This block finally exports the solution set back to a csv file with name 'finalfiletargetweek_1' which is essentially the output
finalsolutioninexcel=pd.DataFrame(np_finalsolution)
finalsolutioninexcel.to_csv('finalfiletargetweek_1.csv',index=True)












