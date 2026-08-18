# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Start\n
2.Import the required libraries:
3.NumPy
4.Pandas
5.StandardScaler
6.Load the 50_Startups.csv dataset.
7.Select the three input features:
8.R&D Spend
9.Administration
10.Marketing Spend
11.Select Profit as the target variable.
12.Standardize the input features using StandardScaler.
13.Standardize the target variable.
14.Add a bias column of 1s to the input matrix.
15.Initialize the model parameters theta with zeros.
16.Set the learning rate and number of iterations.
17.Repeat the following steps for the specified number of iterations:

18.Calculate predictions:

    predictions=Xθ

19.Calculate errors:

    error=predictions−y

20.Update the parameters:

    θ=θ−α(gradient)
Return the optimized values of theta.
Take new startup data as input.
Standardize the new input using the same scaler.
Add the bias term.

21.Calculate the scaled prediction using:

22.prediction=Xθ
23.Convert the scaled prediction back to the original profit scale using inverse_transform().
24.Display the predicted profit.
25.Stop

## Program:

# Program to implement the linear regression using gradient descent.
# Developed by: Jegan P
# RegisterNumber:  212225240061

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1,num_iters=1000):

  #add bias column
  X=np.c_[np.ones(len(X1)),X1]

  #initialize theta
  theta=np.zeros((X.shape[1],1))

  #Gradient Descent
  for _ in range(num_iters):
    predictions=X.dot(theta)
    errors=predictions-y
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta



data=pd.read_csv(r"/content/drive/MyDrive/50_Startups.csv")

#Features(R&D Spend, Administration, Marketing Spend)
X=data.iloc[:,:-1].values

X=X[:,[0,1,2]]

#Target
y=data.iloc[:,-1].values.reshape(-1,1)

#Scale features
scaler_X=StandardScaler()
X_scaled=scaler_X.fit_transform(X)

#Scale target
scaler_y=StandardScaler()
y_scaled=scaler_y.fit_transform(y)
print(X_scaled)

#Train model
theta = linear_regression(X_scaled, y_scaled)
print("Theta:")
print(theta)


new_data=np.array([[165349.2,136897.8,471784.1]])
new_scaled=scaler_X.transform(new_data)
new_scaled=np.c_[np.ones(1),new_scaled]
prediction_scaled=new_scaled.dot(theta)
prediction=scaler_y.inverse_transform(prediction_scaled)
print("Scaled Prediction:",prediction_scaled)
print("Predicted Profit:",prediction)
```

## Output:

<img width="710" height="718" alt="image" src="https://github.com/user-attachments/assets/2ea5bdbf-cf23-452c-8d75-60d95fd1ebc4" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
