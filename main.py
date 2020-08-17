from get_data import dataloader
from column_encoder import ColumnEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import numpy as np
from lightgbm import LGBMClassifier
import get_data
#from xgboost import XGBClassifier, XGBRegressor

kc_train = dataloader('data/kaggle_cat_train.csv', "kaggle_cat")
kc_train.get_input_target()
kc_test = dataloader('data/kaggle_cat_train.csv', "kaggle_cat")
kc_test.get_input_target()

encoders = ['OneHotEncoder',
            'SimilarityEncoder',
            #'NgramNaiveFisherKernel',
            #'BackwardDifferenceEncoder',
            # 'BinaryEncoder',
            # 'HashingEncoder',
            # 'HelmertEncoder',
            # 'SumEncoder',
            # 'PolynomialEncoder',
            # 'BaseNEncoder',
            # 'NgramsCountVectorizer',
            # 'NgramsTfIdfVectorizer',
            # 'WordNMF',
            # 'WordNgramsTfIdfVectorizer',
    
]

not_working =  [ 
            #'ngrams_hot_vectorizer',
            #'NgramsLDA',
            # 'NgramsMultinomialMixture',
            
            #takes too long
            # 'NMF' 
            # 'AdHocNgramsMultinomialMixture',
            #'AdHocIndependentPDF',
    
            # These have some kmeans initialization errors cuz of matrix dims
            #'OnlineGammaPoissonFactorization',
            # 'OnlineGammaPoissonFactorization2',
            # 'OnlineGammaPoissonFactorization3',
            # 'OnlineGammaPoissonFactorization4',
            # 'WordOnlineGammaPoissonFactorization',
            # 'OnlineGammaPoissonFactorization_fast',
            
            # Other errors
            # 'MDVEncoder',
            # 'MinHashEncoder',
    
            # 'PretrainedFastText',
            # 'PretrainedFastText_hu',
    ]

supervised = [
    #'TargetEncoder', 'LeaveOneOutEncoder',
    ]



def experiments(enc):
    kc_train.col_encoders = {col:  ColumnEncoder(enc) 
        for col in kc_train.X.columns}
    
    nominal = ['nom_'+str(i) for i in range(0,10)]
    
    kc_train_sample_X = kc_train.X.iloc[:300][nominal]
    #kc_train_sample_y = kc_train.y.iloc[:300]
    kc_test_sample_X = kc_test.X.iloc[:50][nominal]
    #kc_test_sample_y = kc_test.y.iloc[:50]
    
    cT = ColumnTransformer([(col, kc_train.col_encoders[col], col) for col in kc_train_sample_X])
    return cT.fit_transform(kc_train_sample_X)
    #x_out = cT.fit_transform(kc_train_sample_X)
    
    # x_test_out = cT.transform(kc_test_sample_X)
    
    # lgbm = LGBMClassifier()
    # lgbm.fit(x_out, kc_train_sample_y)
    
    # print(classification_report(y_pred = lgbm.predict(x_test_out), y_true = kc_test_sample_y))
    
from joblib import Parallel, delayed
import multiprocessing

# inputs = range(10) 
# def processInput(i):
#     return i * i

num_cores = multiprocessing.cpu_count()

results = Parallel(backend="multiprocessing", n_jobs=num_cores)(delayed(experiments)(enc) for enc in encoders)
print(results)


# for enc in encoders:
#     print(enc)
#     kc_train.col_encoders = {col:  ColumnEncoder(enc) 
#         for col in kc_train.X.columns}
    
#     nominal = ['nom_'+str(i) for i in range(0,10)]
    
#     kc_train_sample_X = kc_train.X.iloc[:300][nominal]
#     kc_train_sample_y = kc_train.y.iloc[:300]
#     kc_test_sample_X = kc_test.X.iloc[:50][nominal]
#     kc_test_sample_y = kc_test.y.iloc[:50]
    
#     cT = ColumnTransformer([(col, kc_train.col_encoders[col], col) for col in kc_train_sample_X])
#     cT.fit(kc_train_sample_X)
#     #x_out = cT.fit_transform(kc_train_sample_X)
    
#     # x_test_out = cT.transform(kc_test_sample_X)
    
#     # lgbm = LGBMClassifier()
#     # lgbm.fit(x_out, kc_train_sample_y)
    
#     # print(classification_report(y_pred = lgbm.predict(x_test_out), y_true = kc_test_sample_y))





            