from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def train_decision_tree(x_train, y_train, max_depth=3, max_features=10, random_state=567):
    dt = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state)
    dtmodel = dt.fit(x_train, y_train)
    return dtmodel

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae

def plot_tree(model, file_name='tree.png'):
    from sklearn import tree
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20,10))
    tree.plot_tree(model, feature_names=model.feature_names_in_, filled=True)
    plt.savefig(file_name, dpi=300)
    plt.show()
