import time
import os
import glob
import pandas as pd
import re
import numpy as np
import missingno as msno
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# Concatenating Subsets
# path = r'./data/carbon'
# all_files = glob.glob(os.path.join(path, "*.csv"))
# concat_df = pd.concat((pd.read_csv(f) for f in all_files))
# concat_df.to_csv('./data/raw_concatenated.csv', index=False)


# Loading
print("Loading data...")
raw_df = pd.read_csv("./data/raw_concatenated.csv")


# Subsetting Columns
print("Dropping unnecessary columns...")
raw_df = raw_df.drop(columns=['batch_date', 'swab_type', 'test_name', 'temperature', 'pulse', 'sys', 'dia', 'rr', 'sats', 'rapid_flu_results', 'rapid_strep_results',
'ctab', 'labored_respiration', 'rhonchi', 'wheezes', 'days_since_symptom_onset', 'cough_severity', 'sob_severity', 'cxr_findings', 'cxr_impression', 
'cxr_label', 'cxr_link', 'er_referral'])


# Analyzing NA Distribution & Count NAs
print("Analyzing NA values distribution...")
na_chart = msno.matrix(raw_df)
na_chart_copy = na_chart.get_figure()
na_chart_copy.savefig('output/na_chart.png', bbox_inches = 'tight')
plt.clf()
# print(raw_df.shape)
# print(raw_df.isnull().sum())


# Converting objects to strings & Lowercasing
print("Converting to strings & lowercasing...")
string_col_list = raw_df.drop(columns=['age']).columns
raw_df[string_col_list] = raw_df[string_col_list].astype(str)
# don't use .apply(str) ever again. It force-applies a string type, which would include the newline character
for string_col in string_col_list:
    raw_df[string_col] = raw_df[string_col].str.lower()


# No NAN Encoding
print("Encoding...")
bool_col_list = ['high_risk_exposure_occupation', 'high_risk_interactions', 'diabetes', 'chd', 'htn', 'cancer', 'asthma', 'copd', 'autoimmune_dis', 
'smoker', 'cough', 'fever', 'sob', 'diarrhea', 'fatigue', 'headache', 'loss_of_smell', 'loss_of_taste', 'runny_nose', 'muscle_sore', 'sore_throat']
raw_df[bool_col_list] = raw_df[bool_col_list].replace({'true': 1, 'false': 0, 'nan': np.nan})
raw_df['covid19_test_results'] = raw_df['covid19_test_results'].replace({'negative': 0, 'positive': 1, 'nan': np.nan})
# bad encoders, need reshaping and doesn't work
# bool_label_encoder = OneHotEncoder(handle_unknown='ignore')
# bool_label_encoder = bool_label_encoder.fit(raw_df['high_risk_exposure_occupation']) 
# # did not use fit_transform() because I want encoding to be memorized/to be able to apply the same encoding for different columns of the same values
# # also good for inverse transform
# bool_col_list = ['high_risk_exposure_occupation', 'high_risk_interactions', 'diabetes', 'chd', 'htn', 'cancer', 'asthma', 'copd', 'autoimmune_dis', 
# 'smoker', 'cough', 'fever', 'sob', 'diarrhea', 'fatigue', 'headache', 'loss_of_smell', 'loss_of_taste', 'runny_nose', 'muscle_sore', 'sore_throat']
# for bool_col in bool_col_list:
#     raw_df[bool_col] = bool_label_encoder.transform(raw_df[bool_col])


# Age Outlier & Scaling
print("Checking outlier and scaling on Age")
raw_df.loc[raw_df['age'] < 150]
min_max_scaler = MinMaxScaler()
raw_df['age'] = min_max_scaler.fit_transform(raw_df[['age']])
# use double brackets to get a df format instead of series. This way scalers will work


# Converting to Categorical
print("Converting to Categorical...")
raw_df[string_col_list] = raw_df[string_col_list].astype("category")


# Dropping All Remaining NA
print("Creating full set for non-XGBoost methods...")
raw_df_full = raw_df.drop(columns=['high_risk_interactions', 'fever'])
# raw_df = raw_df.replace({'None': np.nan, 'Other': np.nan})
raw_df_full.dropna(inplace=True)
string_col_list_1 = raw_df.drop(columns=['age', 'high_risk_interactions', 'fever']).columns
raw_df_full[string_col_list_1] = raw_df_full[string_col_list_1].astype(int)
raw_df_full[string_col_list_1] = raw_df_full[string_col_list_1].astype("category")


# Analyzing Distribution of Class Labels
print("Analyzing distribution of class labels...")
#print(raw_df['covid19_test_results'].value_counts())
test_histo = raw_df['covid19_test_results'].hist()
test_histo_copy = test_histo.get_figure()
test_histo_copy.savefig('output/test_histo.png', bbox_inches = 'tight')



# Train/Validation Split
print("Train/Validation Split...")
X_train_boost, X_validation_boost, Y_train_boost, Y_validation_boost = train_test_split(raw_df.drop(['covid19_test_results'], axis=1), 
raw_df['covid19_test_results'], test_size=0.20, random_state=0, stratify=raw_df['covid19_test_results'])
X_train_full, X_validation_full, Y_train_full, Y_validation_full = train_test_split(raw_df_full.drop(['covid19_test_results'], axis=1), 
raw_df_full['covid19_test_results'], test_size=0.20, random_state=0, stratify=raw_df_full['covid19_test_results'])


# # Undersampling
# print("Undersampling...")
# raw_df = raw_df.sample(frac=1)
# one_star = raw_df.loc[raw_df['Score'] == 1.0]
# two_star = raw_df.loc[raw_df['Score'] == 2.0]
# three_star = raw_df.loc[raw_df['Score'] == 3.0]
# four_star = raw_df.loc[raw_df['Score'] == 4.0]
# five_star = raw_df.loc[raw_df['Score'] == 5.0]
# base = one_star.shape[0]
# rem_4 = four_star.shape[0] - 2*(one_star.shape[0])
# rem_5 = five_star.shape[0] - 2*(one_star.shape[0])
# drop_indices_4 = np.random.choice(four_star.index, rem_4, replace=False)
# four_star = four_star.drop(drop_indices_4)
# drop_indices_5 = np.random.choice(five_star.index, rem_5, replace=False)
# five_star = five_star.drop(drop_indices_5)
# raw_df = one_star.append(two_star).append(three_star).append(four_star).append(five_star)
# print(raw_df['Score'].value_counts())




# from sklearn.linear_model import LogisticRegression   
# Lr = LogisticRegression(class_weight='balanced')

'''
# Saving to Local
print("Saving to Local...")
X_train.to_pickle("./data/X_train.pkl")
X_validation.to_pickle("./data/X_validation.pkl")
Y_train.to_pickle("./data/Y_train.pkl")
Y_validation.to_pickle("./data/Y_validation.pkl")
X_submission.to_pickle("./data/X_submission.pkl")
'''

'''
# time
tfidf_s_time = time.perf_counter()
tfidf_f_time = time.perf_counter()
print('tfidf vectorizer took: ' + str(tfidf_f_time - tfidf_s_time) + ' seconds')
'''
