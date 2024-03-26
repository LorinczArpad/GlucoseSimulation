import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
class DecisionTree():
    def __init__(self):
        self.dtr = DecisionTreeRegressor(max_depth=3)
    def train(self):
        data = pd.read_csv("./Results/EditedTestResults.csv")[:-1]  # Replace "your_data.csv" with the path to your data file
        fixed_Data = data.drop(columns='BG')
        # Split the data into features (X) and target variable (y)
        X = fixed_Data .drop(columns=['Time', 'insulin'])  # Assuming 'Time' and 'insulin' are not features
        y = fixed_Data ['insulin']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
        # Train the DecisionTreeRegressor
        self.dtr.fit(X_train, y_train)
    def create_result(self, test_data):
        test_df = pd.DataFrame.from_dict(test_data)
        # Drop the 'Time' column as it's not needed for prediction
        test_df = test_df.drop(columns=['Time'])

        # Convert DataFrame to a NumPy array
        test_array = test_df.values

        # Print the test array
        predicted_insulin = self.dtr.predict(test_array)

        # Extract the single predicted value
        single_predicted_insulin = predicted_insulin[0]

        # Print the single predicted value
        return single_predicted_insulin

test = DecisionTree()
test.train()

test_data = {
    'Time': ['2024-03-26 18:30:01.768573'],
    'CGM': [89.90822635385112],
    'CHO': [0.0],
    'LBGI': [0.0],
    'HBGI': [2.7552919270419367],
    'Risk': [2.7552919270419367,]
}
print(f"GECIIIIIIIIIIIIIIIIII {test.create_result(test_data)}")
