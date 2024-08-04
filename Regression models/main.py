from data_preprocessing.load_data import load_data
from data_preprocessing.preprocess_data import prepare_data, split_data
from models.linear_regression import train_linear_regression, evaluate_model as evaluate_lr
from models.decision_tree import train_decision_tree, evaluate_model as evaluate_dt, plot_tree
from models.random_forest import train_random_forest, evaluate_model as evaluate_rf
from utils.save_model import save_model, load_model

# Load the data
file_path = r"C:\Users\jaisw\Desktop\Data Science Final Project\complete project\Regression models\Regression models\final.csv"  # Ensure this path is relative to the main.py script location
df = load_data(file_path)

# Preprocess the data
x, y = prepare_data(df)

# Split the data
x_train, x_test, y_train, y_test = split_data(x, y)

# Train and evaluate Linear Regression model
lrmodel = train_linear_regression(x_train, y_train)
lr_train_mae = evaluate_lr(lrmodel, x_train, y_train)
lr_test_mae = evaluate_lr(lrmodel, x_test, y_test)
print(f"Linear Regression Train MAE: {lr_train_mae}, Test MAE: {lr_test_mae}")

# Train and evaluate Decision Tree model
dtmodel = train_decision_tree(x_train, y_train)
dt_train_mae = evaluate_dt(dtmodel, x_train, y_train)
dt_test_mae = evaluate_dt(dtmodel, x_test, y_test)
print(f"Decision Tree Train MAE: {dt_train_mae}, Test MAE: {dt_test_mae}")
plot_tree(dtmodel)

# Train and evaluate Random Forest model
rfmodel = train_random_forest(x_train, y_train)
rf_train_mae = evaluate_rf(rfmodel, x_train, y_train)
rf_test_mae = evaluate_rf(rfmodel, x_test, y_test)
print(f"Random Forest Train MAE: {rf_train_mae}, Test MAE: {rf_test_mae}")

# Save the Random Forest model
save_model(rfmodel, 'RE_Model.pkl')

# Load and predict with the saved model
loaded_rf_model = load_model('RE_Model.pkl')
sample_data = [[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]]
sample_prediction = loaded_rf_model.predict(sample_data)
print(f"Sample Prediction: {sample_prediction}")