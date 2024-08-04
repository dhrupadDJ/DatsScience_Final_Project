from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_random_forest(x_train, y_train, n_estimators=200, criterion='absolute_error'):
    rf = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
    rfmodel = rf.fit(x_train, y_train)
    return rfmodel

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae
