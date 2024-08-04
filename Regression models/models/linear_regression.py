from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train_linear_regression(x_train, y_train):
    lrmodel = LinearRegression().fit(x_train, y_train)
    return lrmodel

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae
