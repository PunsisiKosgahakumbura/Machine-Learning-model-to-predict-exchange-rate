#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("exchangerate.csv")
data.head()


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[7]:


data = data.drop(data.index[[0,0]],
                axis= 0) # remove the 0th row sice there is no any usefull information.


# In[8]:


data.describe(include=['object'])


# In[9]:


data.describe(include=['float64'])


# In[11]:


data["Value"]= data["Value"].astype("float64")#convert value into float


# In[12]:


data['Element Code'] = data['Element Code'].replace(['LCU', 'SLC'], [0, 1]) #convert element codes into numeric


# In[14]:


data['ISO Currency Code'] = data['ISO Currency Code'].replace({'LNR':0, 'LKR':1}) #convert currency code into numeric


# In[15]:


data = data[data['Months Code']!=7021]  #remove months= anuual, from data set


# In[16]:


data=data.drop(columns=['Iso3','StartDate','EndDate','Area Code (M49)','Area','Element',
                            'Currency','Year','Months','Unit','Flag'])#drop the unnecessary columns from data set


# In[17]:


data.describe() #getting descriptive statistics


# In[18]:


data.corr()['Value']   #get the correlation with y variable


# In[20]:


data.shape


# In[21]:


data.drop(['Area Code','Element Code'],
                axis= 1,inplace=True) #drop area code and Element Code as there is no variation                                    


# In[23]:


import sklearn as sl
X = data.drop(['Value'], axis=1)
Y = data['Value']
Y


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42) 
#20% of the data is assigned to the test set, 60% training set and a 20% validation set.


# In[25]:


X_train.shape ,X_test.shape,  X_val.shape


# In[26]:


train_data = X_train.join(Y_train)
train_data


# In[27]:


train_data.hist()


# In[28]:


train_data.corr()


# In[29]:


import matplotlib.pyplot as plt

plt.plot(data['Year Code'], data['Value'])
plt.xlabel('Year Code')
plt.ylabel('Value')
plt.title('Variation of Exchange rate with Year')
plt.xticks(rotation=20)
plt.show()


# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[31]:


def train_and_validate_model(model, X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train)
    Y_pred_val = model.predict(X_val)
    mse_val = mean_squared_error(Y_val, Y_pred_val)
    return mse_val


# In[32]:


linear_model = LinearRegression()
rf_model = RandomForestRegressor()
dt_model = DecisionTreeRegressor()


# In[33]:


#Linear Regression
mse_val = train_and_validate_model(linear_model, X_train, Y_train, X_val, Y_val)
print("Validation MSE:", mse_val)


# In[34]:


# Random Forest Regressor
mse_val_rf = train_and_validate_model(rf_model, X_train, Y_train, X_val, Y_val)
print("Random Forest Regressor Validation MSE:", mse_val_rf)


# In[35]:


# Decision Tree Regressor
mse_val_dt = train_and_validate_model(dt_model, X_train, Y_train, X_val, Y_val)
print("Decision Tree Regressor Validation MSE:", mse_val_dt)


# In[36]:


def scatter_plot_validation_predictions(X_val, Y_val, Y_pred_val, model_name):
    plt.scatter(X_val.iloc[:, 0], Y_val, color='blue', label='Actual')
    plt.scatter(X_val.iloc[:, 0], Y_pred_val, color='red', label='Predicted ' + model_name)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(f'{model_name} - Validation Predictions')
    plt.legend()
    plt.show()


# In[37]:


Y_pred_linear_val = linear_model.predict(X_val)
Y_pred_rf_val = rf_model.predict(X_val)
Y_pred_dt_val = dt_model.predict(X_val)
scatter_plot_validation_predictions(X_val, Y_val,Y_pred_linear_val, 'Linear Regression')
scatter_plot_validation_predictions(X_val, Y_val,Y_pred_rf_val, 'Random Forest Regressor')
scatter_plot_validation_predictions(X_val, Y_val,Y_pred_dt_val, 'Decision Tree Regressor')


# In[38]:


y_pred_rf_test = rf_model.predict(X_test)
mse_rf_test = mean_squared_error(Y_test, y_pred_rf_test)
print("Random Forest Regressor Test MSE:", mse_rf_test)


# In[39]:


y_pred_dt_test = dt_model.predict(X_test)
mse_dt_test = mean_squared_error(Y_test, y_pred_dt_test)
print("Decision Tree Regressor Test MSE:", mse_dt_test)


# In[41]:


def model_acc(model):
    model.fit(X_train, Y_train)
    acc = model.score(X_test, Y_test)
    print(str(model) + '>>>' + str(acc))


# In[42]:


model_acc(linear_model)
model_acc(rf_model)
model_acc(dt_model)


# In[43]:


data.info()


# In[44]:


# Prompt the user for input values
iso_currency_code = input("Enter ISO Currency Code: ")
year_code = input("Enter Year Code: ")
months_code =input("Enter Months Code: ")

# Create a DataFrame with the user input
input_data = pd.DataFrame({'ISO Currency Code': [iso_currency_code],
                           'Year Code': [year_code],
                           'Months Code': [months_code]})

# Make predictions using the loaded model and the user input
predicted_value = dt_model.predict(input_data)

# Print the predicted value
print("Predicted Value:", predicted_value[0])          


# In[45]:


# Prompt the user for input values
iso_currency_code = input("Enter ISO Currency Code: ")
year_code = input("Enter Year Code: ")
months_code =input("Enter Months Code: ")

# Create a DataFrame with the user input
input_data = pd.DataFrame({'ISO Currency Code': [iso_currency_code],
                           'Year Code': [year_code],
                           'Months Code': [months_code]})

# Make predictions using the loaded model and the user input
predicted_value = dt_model.predict(input_data)

# Print the predicted value
print("Predicted Value:", predicted_value[0])     


# In[46]:


# Prompt the user for input values
iso_currency_code = input("Enter ISO Currency Code: ")
year_code = input("Enter Year Code: ")
months_code =input("Enter Months Code: ")

# Create a DataFrame with the user input
input_data = pd.DataFrame({'ISO Currency Code': [iso_currency_code],
                           'Year Code': [year_code],
                           'Months Code': [months_code]})

# Make predictions using the loaded model and the user input
predicted_value = dt_model.predict(input_data)

# Print the predicted value
print("Predicted Value:", predicted_value[0])  #check for a testing data


# In[ ]:




