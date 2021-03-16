#Simple Linear Regression Example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
df = pd.read_csv("C:\\Users\\Arindam's PC\\Desktop\\Python Datas\\Reg_data.csv")
print(df)
print(df.columns)
X = df[['EngineSize']]
Y = df[['Co2']]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.40,random_state = 101)
print(X_train)
print(Y_train)
lm = LinearRegression()
lm.fit(X_train,Y_train)
print(lm.intercept_)
print(lm.coef_)
print(X_test)
print(Y_test)
prediction = lm.predict(X_test) 
print(prediction)
plt.scatter(Y_test,prediction)
sns.histplot(Y_test-prediction)
print(metrics.mean_absolute_error(Y_test,prediction))
print(metrics.mean_squared_error(Y_test,prediction))
print(np.sqrt(metrics.mean_squared_error(Y_test,prediction)))
print(metrics.explained_variance_score(Y_test,prediction))
print(metrics.r2_score(Y_test,prediction))
#below is useful for multiple linear regression to determine which independent
#variable is more effective.
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(cdf)
      
                 

                  

