from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


start = date(2012,1,1)
#end =  date(2017,9,28)
end = date.today()

#stock = data.DataReader("INPX","yahoo",start,end)
#stock.head()
#stock.tail()
#stock['Adj Close'].plot()
#plt.show()

#===============#

apple = data.DataReader("AAPL","yahoo",start,end)
#apple['Adj Close'].plot()
#apple["2017"].plot()

index=apple.index
win = 5
N_AdjClose = ['AdjClose-4','AdjClose-3','AdjClose-2',
              'AdjClose-1','AdjClose-0','AdjClose+1']
N_Low = ['Low-4','Low-3','Low-2',
              'Low-1','Low-0']
N_High = ['High-4','High-3','High-2',
              'High-1','High-0']
N_Open = ['Open-4','Open-3','Open-2',
              'Open-1','Open-0']

zeros_cl = np.zeros((len(apple)-(win+1), win+1 ))     #AdjClose
df_cl = pd.DataFrame(zeros_cl, columns=N_AdjClose, index=index[win:len(index)-1])

zeros_lo = np.zeros((len(apple)-(win), win ))  #Low
df_lo = pd.DataFrame(zeros_lo, columns=N_Low, index=index[win:len(index)])   

zeros_hi = np.zeros((len(apple)-(win), win ))   # High
df_hi = pd.DataFrame(zeros_hi, columns=N_High, index=index[win:len(index)])

zeros_op = np.zeros((len(apple)-(win), win ))   #Open
df_op = pd.DataFrame(zeros_op, columns=N_Open, index=index[win:len(index)])

for d in range(win+1, len(index)):
    p_op = apple['Open'][index[d-(win):d]]
    df_op.iloc[d-(win),:] = p_op.values.reshape(-1)
    
    p_lo = apple['Low'][index[d-(win):d]]
    df_lo.iloc[d-(win),:] = p_lo.values.reshape(-1)
    
    p_hi = apple['High'][index[d-(win):d]]
    df_hi.iloc[d-(win),:] = p_hi.values.reshape(-1)
    
    p_cl = apple['Adj Close'][index[d-(win+1):d]]
    df_cl.iloc[d-(win+1),:] = p_cl.values.reshape(-1)

df = pd.concat([df_op, df_hi, df_lo, df_cl])

#Fill missing values
df.fillna(df.mean(), inplace=True)

#Prepare input and target
X = df.drop(['AdjClose+1'], axis=1)
y = df['AdjClose+1']

N_data = len(X)
N_train = int(0.8*N_data)
N_test = N_data - N_train

X_train = X.iloc[0:N_train,:]
X_test = X.iloc[N_train+1:N_data-1,:]
y_train = y.iloc[0:N_train]
y_test = y.iloc[N_train+1:N_data-1]

mlp = MLPRegressor(hidden_layer_sizes=(10,))
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

#Compute R^2 and root mean squared error
print("R^2 = {}".format(mlp.score(X_test, y_test)))
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
print("Root mean square error = {}".format(rmse_mlp))

y_test_pred_mlp = pd.DataFrame({'y_test':y_test,
                               'y_pred_mlp':y_pred_mlp}, index=X_test.index )

y_test_pred_mlp.plot(legend=True)
plt.title('MLP Apple Stock Prediction')







reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

#Compute R^2 and root mean squared error
print("R^2 = {}".format(reg.score(X_test, y_test)))
rmse_reg = np.sqrt(mean_squared_error(y_test, y_pred_reg))
print("Root mean square error = {}".format(rmse_reg))

y_test_pred_reg = pd.DataFrame({'y_test':y_test,
                                'y_pred_reg':y_pred_reg}, index=X_test.index )
y_test_pred_reg.plot(legend=True)
plt.title('Linear Regression Apple Stock Prediction')

#y_test_pred_reg['2017'].plot(legend=True)

plt.show()
