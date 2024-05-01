import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
class DecisionTreeForPID():
    def __init__(self):
        self.dtr = DecisionTreeRegressor(max_depth=3)

    def train(self):
        data = pd.read_csv("./Results/pid_train_data.csv")
        X = data.drop(columns=['P','I','D','CHO','CGM','BG','Time','insulin','LBGI','HBGI','Risk'], axis=1) #Real
        X['CGM'] = data['CGM']
        Y = data[['P','I','D']]
        print(X.head())
        self.dtr.fit(X,Y)

    def create_PID_result(self, sample):
        predicted_PID = self.dtr.predict(sample)
        #print(predicted_PID)
        return pd.DataFrame(predicted_PID, columns=['P','I','D']).iloc[0].to_dict()