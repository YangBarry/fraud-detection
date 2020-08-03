import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

#file IO
print("starting loading files...")
train_data = pd.read_csv('newtrain_transaction_200000.csv',index_col='TransactionID')
test_data = pd.read_csv('newtest_transaction.csv',index_col='TransactionID')

train_label=train_data['isFraud'].copy()
test_label=test_data['isFraud'].copy()
print("files are successfully loaded!")

#data processing
print("starting data preprocessing...")

#first deal with strings
#convert ProductCD to numbers
tmp = pd.crosstab(train_data['ProductCD'], train_data['isFraud'], normalize='index') 
productCD_map={'C':float(100*tmp[1][0]/(tmp[0][0]+tmp[1][0])),'H':float(100*tmp[1][1]/(tmp[0][1]+tmp[1][1])),'R':float(100*tmp[1][2]/(tmp[1][2]+tmp[0][2])),'S':float(100*tmp[1][3]/(tmp[1][3]+tmp[0][3])),'W':float(100*tmp[1][4]/(tmp[1][4]+tmp[0][4]))}
for key in productCD_map:
    train_data.loc[train_data['ProductCD'] == key, 'ProductCD'] = productCD_map[key]
    test_data.loc[test_data['ProductCD'] == key, 'ProductCD'] = productCD_map[key]

#convert card4 to numbers
tmp_card4 = pd.crosstab(train_data['card4'], train_data['isFraud'], normalize='index') 
card4_map = {'american express':float(100*tmp_card4[1][0]/tmp_card4[0][0]), 'discover':float(100*tmp_card4[1][1]/tmp_card4[0][1]), 'mastercard':float(100*tmp_card4[1][2]/tmp_card4[0][2]), 'visa':float(100*tmp_card4[1][3]/tmp_card4[0][3])}
for key in card4_map:
    train_data.loc[train_data['card4'] == key, 'card4'] = card4_map[key]
    test_data.loc[test_data['card4'] == key, 'card4'] = card4_map[key]


#convert card6 into numbers
tmp_card6 = pd.crosstab(train_data['card6'], train_data['isFraud'], normalize='index') 
card6_map = {'charge card':float(100*tmp_card6[1][0]/tmp_card6[0][0]), 'credit':float(100*tmp_card6[1][1]/tmp_card6[0][1]), 'debit':float(100*tmp_card6[1][2]/tmp_card6[0][2]), 'debit or credit':float(100*tmp_card6[1][3]/tmp_card6[0][3])}
for key in card6_map:
    train_data.loc[train_data['card6'] == key, 'card6'] = card6_map[key]
    test_data.loc[test_data['card6'] == key, 'card6'] = card6_map[key]


#drop email domain
train_data = train_data.drop(['P_emaildomain', 'R_emaildomain'], axis = 1)
test_data = test_data.drop(['P_emaildomain', 'R_emaildomain'], axis = 1)
# print("done4")

#M1~M9 除去M4
M_features=['M1','M2',"M3",'M5','M6','M7','M8','M9']

for col_names in M_features:
	M_map={'T':1,'F':0}
	for key in M_map:
		train_data.loc[train_data[col_names] == key, col_names] = M_map[key]
		test_data.loc[test_data[col_names]==key, col_names] = M_map[key]
	train_data[col_names].fillna(0.5,inplace=True)
	test_data[col_names].fillna(0.5,inplace=True)

#M4
M4_map={'M2':10,'M1':2.5,'M0':3.75}
for key in M4_map:
	train_data.loc[train_data['M4']==key,'M4'] = M4_map[key]
	test_data.loc[test_data['M4']==key,'M4'] = M4_map[key]
train_data['M4'].fillna(1.25,inplace=True)
test_data['M4'].fillna(1.25,inplace=True)
# print("done5")

name_list=['ProductCD','card4','card4']
for name in name_list:
    train_data[name]=pd.to_numeric(train_data[name],errors='coerce')
    train_data[name].astype('float')
    test_data[name]=pd.to_numeric(test_data[name],errors='coerce')
    test_data[name].astype('float')


#dropping the labels
train_data=train_data.drop(['isFraud'],axis=1)
test_data=test_data.drop(['isFraud'],axis=1)

#separating the discrete data and continuous data
dis_data=[]
con_data=[]
for column in train_data.columns:
	unique_value=list(train_data[column].unique())
	if len(unique_value)<10: dis_data.append(column)
	else: con_data.append(column)

#group the features based on NAN(skip)
NAN_data_train=[]
for column in train_data.columns:
	if train_data[column].isnull().any(axis=0)==1: NAN_data_train.append(column)
NAN_data_test=[]
for column in test_data.columns:
	if test_data[column].isnull().any(axis=0)==1: NAN_data_test.append(column)
# print(NAN_data_train)
# print(NAN_data_test)

#fill NAN

for column in train_data.columns:
	train_data[column].fillna(-1.0,inplace=True)
for column in train_data.columns:
	if train_data[column].isnull().any(axis=0)==1: print(f"{column}: NAN")

for column in test_data.columns:
	test_data[column].fillna(-1.0,inplace=True)
for column in test_data.columns:
	if test_data[column].isnull().any(axis=0)==1: print(f"{column}: NAN")

print("Done with filling NAN")


# # one-hot
# from sklearn.preprocessing import OneHotEncoder
# from numpy import array
# from numpy import argmax

# for f in dis_data or NAN_data_test or NAN_data_train:
# 	encoder=OneHotEncoder(sparse=False)
# 	one_hot_data=train_data[f].values.reshape(-1,1)
# 	encoder.fit(one_hot_data)
# 	train_data[f]=encoder.transform(one_hot_data)
	

print("data preprocessing done!")


print("starting feature selection...")
#feature selection
#correlation calculation(skip)


#feature_selector
from feature_selector import FeatureSelector
fs=FeatureSelector(data=train_data,labels=train_label)

#find features with 0 variance
fs.identify_single_unique()

#recursive feature elimination
fs.identify_zero_importance(task='classification',eval_metric='auc',n_iterations=5,early_stopping=True)
print("finish zero importance analysis")
fs.identify_low_importance(cumulative_importance=0.99)
print("finish low importance analysis")
train_data=fs.remove(methods='all')
print("finish removing train_data")

for col in test_data.columns:
	if col in train_data.columns: continue
	else:  test_data=test_data.drop([col],axis=1)

print("done with feature selection!")
print(f"training data: {train_data.shape}")
print(f"testing data: {test_data.shape}")

#lightgbm
lgb_train=lgb.Dataset(train_data,train_label)
lgb_eval=lgb.Dataset(test_data,test_label,reference=lgb_train)
params={
	'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'metric': {'auc'},  #二进制对数损失
    'num_leaves': 5,  
    'max_depth': 4,  
    'min_data_in_leaf': 20,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.9,  
    'bagging_fraction': 0.90,  
    'bagging_freq': 5,  
    'lambda_l1': 1,    
    'lambda_l2': 0.01,  # 越小l2正则程度越高  
    'min_gain_to_split': 0.2,  
    'verbose': 5,  
    'is_unbalance': True  
}

#train
print("start training...")
gbm=lgb.train(params,lgb_train,num_boost_round=150,valid_sets=lgb_eval,early_stopping_rounds=7)

# save model
# print("saving model...")
# gbm.save_model('lightgbm/model.txt')

#predict
print('Start predicting...')
test_pred=gbm.predict(test_data,num_iteration=gbm.best_iteration)

#convert the probability to 0 or 1
# threshold = 0.5  
# for pred in np.nditer(test_pred,op_flags=['readwrite']):
#     if pred>0.5: pred[...]=1
#     else: pred[...]=0


#evaluation
print(test_pred)
value=roc_auc_score(test_label,test_pred)
print("The roc of prediction is: ",value)
# print('Feature importances: ', list(gbm.feature_importance()))

#write prediction to csv file
newsample_submission = pd.read_csv('newsample_submission.csv', index_col = 'TransactionID')
newsample_submission['isFraud'] = test_pred
print(newsample_submission['isFraud'])

#write auroc score
newsample_submission.to_csv('lightgbm.csv')
temp={'AUROC':float(value)}
AUROC=pd.DataFrame(pd.Series(temp))
AUROC=AUROC.reset_index().rename(columns={'index':'AUROC'})
AUROC.to_csv('lightgbm.csv',mode='a')