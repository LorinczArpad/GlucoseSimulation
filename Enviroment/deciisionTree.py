import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
class DecisionTree():
    def __init__(self):
        self.dtr = DecisionTreeRegressor(max_depth=3)
    def train(self):
        data = pd.read_csv("./Results/Common.csv")[:-1]  # Replace "your_data.csv" with the path to your data file
        print(data.head())
        X = data.drop('insulin', axis=1)
        Y = data['insulin'].round(4)
        #print(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        # Train the DecisionTreeRegressor
        self.dtr.fit(X_train, y_train)
    def create_result(self, test_data):
        test_df = pd.DataFrame.from_dict(test_data)
        # Drop the 'Time' column as it's not needed for prediction

        # Convert DataFrame to a NumPy array
        test_array = test_df.values

        # Print the test array
        predicted_insulin = self.dtr.predict(test_array)

        # Extract the single predicted value
        single_predicted_insulin = predicted_insulin[0]

        # Print the single predicted value
        return single_predicted_insulin
