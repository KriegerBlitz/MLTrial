from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

dataset = pd.read_csv('salary_prediction_data.csv')
dataset = pd.get_dummies(dataset, drop_first=True)

X_unscaled = dataset.drop('Salary', axis = 1)
y_unscaled = dataset['Salary']

XU_train, XU_test, yu_train, yu_test = train_test_split(X_unscaled, y_unscaled, test_size=0.2, random_state=42)

scalerX = StandardScaler()
X_train = scalerX.fit_transform(XU_train)
X_test = scalerX.transform(XU_test)

scalerY = StandardScaler()
y_train = scalerY.fit_transform(yu_train.values.reshape(-1, 1)).ravel()
y_test = scalerY.transform(yu_test.values.reshape(-1, 1)).ravel()

#We're going to optimize alpha vs learning_rate_init

params = {
    'alpha': np.logspace(-4, 1, 5),
    'learning_rate_init': np.logspace(-4, -1, 5),
}

grid_search = GridSearchCV(
    estimator=MLPRegressor(max_iter=500, random_state=42),
    param_grid=params,
    scoring='r2',
    cv=5,
    n_jobs=4,
    verbose=3,
)

grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)

z_matrix = results.pivot(index = 'param_alpha', columns = 'param_learning_rate_init', values = 'mean_test_score')

x_axis = z_matrix.columns.astype(float)
y_axis = z_matrix.index.astype(float)
z_axis = z_matrix.values

fig = go.Figure(data=[go.Surface(
    z=z_axis,
    x=x_axis,
    y=y_axis,
    colorscale='Plasma',
    colorbar_title='R-Squared'
)])

fig.update_layout(
    title='MLP Neural Network: Optimization Surface',
    scene=dict(
        xaxis=dict(title='Learning Rate (Gas)', type='log'),
        yaxis=dict(title='Alpha / L2 Penalty (Brakes)', type='log'),
        zaxis=dict(title='R-Squared (Accuracy)')
    ),
    width=900,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()