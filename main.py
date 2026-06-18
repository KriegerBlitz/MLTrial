from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from rich.console import Console
from rich.table import Table

data = pd.read_csv('salary_prediction_data.csv')
data = pd.get_dummies(data, drop_first=True) #Turns text data into one-hot binary data

y = data['Salary']
X = data.drop('Salary', axis=1)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

AvgTrainingTime = {
        "Linear Regressor" : 0,
        "Random Forest Regressor" : 0,
        "Decision Tree Regressor" : 0,
        "Support Vector Regressor" : 0,
        "MLP Regressor" : 0
}
AvgPredictionTime = {
        "Linear Regressor" : 0,
        "Random Forest Regressor" : 0,
        "Decision Tree Regressor" : 0,
        "Support Vector Regressor" : 0,
        "MLP Regressor" : 0
}
AvgError = {
    "Linear Regressor" : [0,0],
    "Random Forest Regressor" : [0,0],
    "Decision Tree Regressor" : [0,0],
    "Support Vector Regressor" : [0,0],
    "MLP Regressor" : [0,0]
}
for i in range(10): # Test multiple times and take the average to eliminate luck.

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i) #I think this is called a monty carlo?
    Xtrain = scaler_X.fit_transform(X_train)
    Xtest = scaler_X.transform(X_test)
    ytrain = scaler_Y.fit_transform(y_train.values.reshape(-1,1)).ravel()
    ytest = scaler_Y.transform(y_test.values.reshape(-1,1)).ravel()

    Models = {
        "Linear Regressor" : LinearRegression(),
        "Random Forest Regressor" : RandomForestRegressor(random_state=i),
        "Decision Tree Regressor" : DecisionTreeRegressor(random_state=i),
        "Support Vector Regressor" : SVR(),
        "MLP Regressor" : MLPRegressor(max_iter=500)
    }
    Predictions = {
        "Linear Regressor": None,
        "Random Forest Regressor": None,
        "Decision Tree Regressor": None,
        "Support Vector Regressor": None,
        "MLP Regressor": None
    }

    for name, model in Models.items():
        start = time.perf_counter()
        model.fit(Xtrain, ytrain)
        end = time.perf_counter()
        AvgTrainingTime[name] += (end - start)/10

        start = time.perf_counter()
        Predictions[name] = model.predict(Xtest)
        end = time.perf_counter()
        AvgPredictionTime[name] += (end - start)/10

        AvgError[name][0] += mean_squared_error(ytest, Predictions[name])/10
        AvgError[name][1] += r2_score(ytest, Predictions[name])/10

console = Console()

table = Table(title="Results", show_header=True, header_style="bold magenta")

table.add_column("Algorithm", style="cyan", justify="left")
table.add_column("Training Time", justify="right")
table.add_column("Prediction Time", justify="right")
table.add_column("MSE", style="bold red",justify="right")
table.add_column("R-Squared", style="bold green", justify="right")

for name in Models.keys():
    table.add_row(
        name,
        f"{AvgTrainingTime[name]:.4f}s",
        f"{AvgPredictionTime[name]:.4f}s",
        f"{AvgError[name][0]:.4f}",
        f"{AvgError[name][1]:.4f}"
    )

console.print()
console.print(table)


#Results on my computer:
# ┌──────────────────────┬───────────────┬─────────────────┬────────┬───────────┐
# │ Algorithm            │ Training Time │ Prediction Time │    MSE │ R-Squared │
# ├──────────────────────┼───────────────┼─────────────────┼────────┼───────────┤
# │ Linear Regressor     │       0.0005s │         0.0001s │ 0.1244 │    0.8729 │
# │ Random Forest        │       0.1311s │         0.0063s │ 0.1640 │    0.8317 │
# │ Regressor            │               │                 │        │           │
# │ Decision Tree        │       0.0017s │         0.0002s │ 0.2929 │    0.6987 │
# │ Regressor            │               │                 │        │           │
# │ Support Vector       │       0.0147s │         0.0096s │ 0.1492 │    0.8473 │
# │ Regressor            │               │                 │        │           │
# │ MLP Regressor        │       0.2537s │         0.0002s │ 0.1417 │    0.8549 │
# └──────────────────────┴───────────────┴─────────────────┴────────┴───────────┘