import pandas as pd
#from sklearn.preprocessing import LabelEncoder
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
        #dataset = self.data.values
        if supervised:
            self.X = self.data.iloc[:, :-1].astype(str)
            self.y = self.data.iloc[:,-1]
            #self.y.reshape((len(self.y), 1))
        else:
            self.X = self.data
             
    #mAybe put this in experiments and not here
    def test_train_split(self, X, y, test_size=0.33, random_state=1):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            sss.get_n_splits(X, y)
            for train_index, test_index in sss.split(X, y):
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
            return X_train, X_test, y_train, y_test


    def preprocess_data(df, cols):
        def string_normalize(s):
            res = str(s).lower()
            # res = ' ' + res + ' '
            return res
    
        for col in cols:
            print('Preprocessing column: %s' % col)
            df[col] = [string_normalize(str(r)) for r in df[col]]
        return df
