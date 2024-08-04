from sklearn.model_selection import train_test_split

def prepare_data(df):
    # Drop the target column from the input features
    x = df.drop('price', axis=1)
    y = df['price']
    return x, y

def split_data(x, y, test_size=0.2, random_state=1234):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
