from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# 1. Load Data
dataset = pd.read_csv('salary_prediction_data.csv')
dataset = pd.get_dummies(dataset, drop_first=True)

X_unscaled = dataset.drop('Salary', axis=1)
y_unscaled = dataset['Salary']

XU_train, XU_test, yu_train, yu_test = train_test_split(X_unscaled, y_unscaled, test_size=0.2, random_state=42)

# 2. Scale X
scalerX = StandardScaler()
X_train = scalerX.fit_transform(XU_train)
X_test = scalerX.transform(XU_test)

# 3. Scale Y
scalerY = StandardScaler()
y_train = scalerY.fit_transform(yu_train.values.reshape(-1, 1)).ravel()
y_test = scalerY.transform(yu_test.values.reshape(-1, 1)).ravel()

# 4. We're going to optimize max_features vs min_samples_leaf
params = {
    'max_features': np.linspace(0.1, 1.0, 10),              # 10 steps from 10% to 100% features
    'min_samples_leaf': np.linspace(1, 20, 10, dtype=int)   # 10 integer steps from 1 to 20 leaves
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    param_grid=params,
    scoring='r2',
    cv=5,
    n_jobs=4,
    verbose=3,
)

print("Mapping the Random Forest landscape...")
grid_search.fit(X_train, y_train)

# 5. Extract Data
results = pd.DataFrame(grid_search.cv_results_)

# Pivot using the RF parameters
z_matrix = results.pivot(index='param_min_samples_leaf', columns='param_max_features', values='mean_test_score')

x_axis = z_matrix.columns.astype(float)
y_axis = z_matrix.index.astype(float)
z_axis = z_matrix.values

# 6. Plotting
fig = go.Figure(data=[go.Surface(
    z=z_axis,
    x=x_axis,
    y=y_axis,
    colorscale='Viridis', # Changed to Viridis for a fresh look!
    colorbar_title='R-Squared'
)])

fig.update_layout(
    title='Random Forest: Optimization Surface',
    scene=dict(
        xaxis=dict(title='Max Features (%)'),           # Removed type='log'
        yaxis=dict(title='Min Samples Leaf'),           # Removed type='log'
        zaxis=dict(title='R-Squared (Accuracy)')
    ),
    width=900,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()

print(f"Best R-Squared Score: {grid_search.best_score_:.4f}")
print(f"Best Parameters Found: {grid_search.best_params_}")