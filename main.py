from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('salary_prediction_data.csv')
data = pd.get_dummies(data, drop_first=True) #Turns text data into one-hot binary data since robots can't read

y = data['Salary']
X = data.drop('Salary', axis=1)

scaler = StandardScaler()

r2Linear = 0
r2RF = 0
r2SVR = 0
r2DT = 0

mseLinear = 0
mseRF = 0
mseSVR = 0
mseDT = 0

for i in range(10): # Test multiple times and take the average to eliminate luck.

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i) #I think this is called a monty carlo?
    Xtrain = scaler.fit_transform(X_train)
    Xtest = scaler.transform(X_test)

    LinearRegressor = LinearRegression()
    RFRegressor = RandomForestRegressor(random_state=i) #not sure if the parameters have to be named or if it's just for clarity
    DTRegressor = DecisionTreeRegressor(random_state=i)
    SVRRegressor = SVR(C=10000) #salary numbers tend to be big so we need a larger penalty

    LinearRegressor.fit(Xtrain, y_train)
    RFRegressor.fit(Xtrain, y_train)
    DTRegressor.fit(Xtrain, y_train)
    SVRRegressor.fit(Xtrain, y_train) #Autocomplete on pycharm is incredible, I didn't have to type 90% of this

    pLinear = LinearRegressor.predict(Xtest)
    pRF = RFRegressor.predict(Xtest)
    pDT = DTRegressor.predict(Xtest)
    pSVR = SVRRegressor.predict(Xtest)

    r2Linear += r2_score(y_test, pLinear)
    r2RF += r2_score(y_test, pRF)
    r2SVR += r2_score(y_test, pSVR)
    r2DT += r2_score(y_test, pDT)

    mseLinear += mean_squared_error(y_test, pLinear)
    mseRF += mean_squared_error(y_test, pRF)
    mseSVR += mean_squared_error(y_test, pSVR)
    mseDT += mean_squared_error(y_test, pDT)


print("Linear model amse: ", mseLinear/10, " ar2: ",r2Linear/10)
print("RF  model amse: ", mseRF/10, " ar2: ",r2RF/10)
print("SVR model amse: ", mseSVR/10, " ar2: ",r2SVR/10)
print("DT model amse: ", mseDT/10, " ar2: ",r2DT/10)

# Results from test:
# Linear model amse:  99674975.13450691  ar2:  0.8728807244612167
# RF  model amse:  131496990.6114753  ar2:  0.8318260546360072
# SVR model amse:  115758747.41165285  ar2:  0.8522392071029163
# DT model amse:  231679145.43609816  ar2:  0.703120654542128
# For this data, it seems the simple linear model outperforms others consistently. This makes me think the data is
# probably forged. This is how the Schoen scandal came out, after all. The second derivative would be close to 0