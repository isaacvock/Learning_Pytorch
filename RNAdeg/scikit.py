### PURPOSE OF THIS SCRIPT
## Get some more experience with scikit learn


##### TUTORIAL ######

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()


### Try loading my own data with Pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np

RNAdeg_df = pd.read_csv('C:/Users/isaac/Documents/Simon_Lab/EZbakR_Isoforms_and_NMD_Data/Figure3/Filtered_kdeg_feature_table.csv')

X = RNAdeg_df.drop(["transcript_id", "mean_treatmentDMSO", "mean_treatment11j",
                    "minus1_AA", "minus2_AA"], axis = 1)
y = RNAdeg_df["mean_treatmentDMSO"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 43
)

### Linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions
y_pred_lin = lin_reg.predict(X_test)

# Evalulate
mse_lin = mean_squared_error(y_test, y_pred_lin)

print("Linear Regression Performance:")
print("MSE =", mse_lin)

# Plot
plt.figure()
plt.scatter(y_test, y_pred_lin)
min_val = min(min(y_test), min(y_pred_lin))
max_val = max(max(y_test), max(y_pred_lin))
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Linear Regression)')
plt.show()

### AdaBoost model
ada = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=300,
    random_state=43
)


ada.fit(X_train, y_train)

# Make predictions
y_pred_ada = ada.predict(X_test)

# Evalulate
mse_ada = mean_squared_error(y_test, y_pred_ada)

print("AdaBoost Performance:")
print("MSE =", mse_ada)

# Plot
plt.figure()
plt.scatter(y_test, y_pred_ada)
min_val = min(min(y_test), min(y_pred_ada))
max_val = max(max(y_test), max(y_pred_ada))
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (AdaBoost)')
plt.show()


### Optimize hyperparameters for Adaboost

ada2 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4)
)

nests = list(range(50, 600, 50))

parameters = {
    'n_estimators' : nests,
    'learning_rate': list(10.0 ** np.linspace(start = -3,
                                         stop = 1,
                                         num = len(nests)))
}

# adacv = GridSearchCV(ada2, parameters)

# adacv.fit(X_train, y_train)

# adacv.best_params_

ada = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    learning_rate = 0.25,
    n_estimators= 200
)


ada.fit(X_train, y_train)

# Make predictions
y_pred_ada = ada.predict(X_test)

# Evalulate
mse_ada = mean_squared_error(y_test, y_pred_ada)

print("AdaBoost Optimized Performance:")
print("MSE =", mse_ada)

# Plot
plt.figure()
plt.scatter(y_test, y_pred_ada)
min_val = min(min(y_test), min(y_pred_ada))
max_val = max(max(y_test), max(y_pred_ada))
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (AdaBoost)')
plt.show()
