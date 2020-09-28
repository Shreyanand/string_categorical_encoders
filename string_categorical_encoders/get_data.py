import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class dataloader():
    def __init__(self, filename, data_name, header='infer'):
        self.data_name = data_name
        self.data = pd.read_csv(filename, header=header)
        self.clean()
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.col_encoders = None
    
    def clean(self):
        if self.data_name == 'kaggle_cat':
            self.data = self.data.drop('id', axis=1)
    
    def get_input_target(self, supervised=True):
        if supervised:
            self.X = self.data.iloc[:, :-1].astype(str)
            self.y = self.data.iloc[:,-1]
            #self.y.reshape((len(self.y), 1))
        else:
            self.X = self.data

    def preprocess_data(df, cols):
        def string_normalize(s):
            res = str(s).lower()
            return res    
        for col in cols:
            print('Preprocessing column: %s' % col)
            df[col] = [string_normalize(str(r)) for r in df[col]]
        return df
