import time
import os
import glob
import pandas as pd
import re
import numpy as np
import missingno as msno
import nltk
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Concatenating Subsets
path = r'./data/carbon'
all_files = glob.glob(os.path.join(path, "*.csv"))
concat_df = pd.concat((pd.read_csv(f) for f in all_files))
concat_df.to_csv('./data/raw_concatenated.csv', index=False)

# Loading
print("Loading data...")
raw_df = pd.read_csv("./data/raw_concatenated.csv")

# Subsetting Columns
# print("Dropping unnecessary columns...")
# raw_df = raw_df.drop(columns=['test_date', 'test_indication'])

# Handling NA
print("Handling NA...")

# raw_df = raw_df.replace({'None': np.nan, 'Other': np.nan})
print(raw_df.head())
na_chart = msno.matrix(raw_df)
na_chart_copy = na_chart.get_figure()
na_chart_copy.savefig('output/na_chart.png', bbox_inches = 'tight')
plt.clf()

print(raw_df.isnull().sum())
print(raw_df.shape)

# raw_df.dropna(inplace=True)
print(raw_df['covid19_test_results'].value_counts())
test_histo = raw_df['covid19_test_results'].hist()
test_histo_copy = test_histo.get_figure()
test_histo_copy.savefig('output/test_histo.png', bbox_inches = 'tight')

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

# '''
# # Converting objects to strings
# print("Converting to strings...")
# raw_df['Summary'] = raw_df['Summary'].apply(str)
# raw_df['Text'] = raw_df['Text'].apply(str)
# X_submission['Summary'] = X_submission['Summary'].apply(str)
# X_submission['Text'] = X_submission['Text'].apply(str)


# # Lowercase
# print("Converting to lowercase...")
# raw_df['Summary'] = raw_df['Summary'].str.lower()
# raw_df['Text'] = raw_df['Text'].str.lower()
# X_submission['Summary'] = X_submission['Summary'].str.lower()
# X_submission['Text'] = X_submission['Text'].str.lower()


# # Punctuation, Special Character & Whitespace (adjusted for stopwords)
# print("Removing punctuations and special characters...")
# def fast_rem(my_string):
#     return(re.sub(r'[^a-z \']', '', my_string).replace('\'', ' '))
# raw_df['Summary'] = raw_df['Summary'].apply(fast_rem)
# raw_df['Text'] = raw_df['Text'].apply(fast_rem)
# X_submission['Summary'] = X_submission['Summary'].apply(fast_rem)
# X_submission['Text'] = X_submission['Text'].apply(fast_rem)


# # Tokenization, Lemmatization
# print("Tokenization and Lemmatization...")
# lemmatizer = WordNetLemmatizer()
# tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
# def fast_lemma(sentence):
#     return (" ".join([lemmatizer.lemmatize(key[0], tag_dict.get(key[1][0], wordnet.NOUN)) for key in nltk.pos_tag(word_tokenize(sentence))]))
# tk_start = time.perf_counter()
# raw_df['Summary'] = raw_df['Summary'].apply(fast_lemma)
# raw_df['Text'] = raw_df['Text'].apply(fast_lemma)
# X_submission['Summary'] = X_submission['Summary'].apply(fast_lemma)
# X_submission['Text'] = X_submission['Text'].apply(fast_lemma)
# tk_stop = time.perf_counter()
# print("Tokenization and Lemmatization took: " + str(tk_stop-tk_start) + ' seconds')


# # Stopword & Noise Removal (Token with length below 2)
# print("Removing Stopwords...")
# cachedStopWords = stopwords.words("english")
# def fast_stop(my_string):
#     return(' '.join([word for word in my_string.split() if word not in cachedStopWords and len(word) > 2]))
# raw_df['Summary'] = raw_df['Summary'].apply(fast_stop)
# raw_df['Text'] = raw_df['Text'].apply(fast_stop)
# X_submission['Summary'] = X_submission['Summary'].apply(fast_stop)
# X_submission['Text'] = X_submission['Text'].apply(fast_stop)


# # Vectorizer
# print("Vectorization - Fitting...")
# vectorizer = TfidfVectorizer(lowercase = False, ngram_range= (1,2), min_df = 5, max_df = 0.9, max_features = 5000).fit(raw_df['Text'])
# vectorizer_s = TfidfVectorizer(lowercase = False, ngram_range= (1,2), min_df = 5, max_df = 0.9, max_features = 1000).fit(raw_df['Summary'])
# print("Vectorization - Transforming...")
# raw_df_vect = vectorizer.transform(raw_df['Text'])
# X_submission_vect = vectorizer.transform(X_submission['Text'])
# raw_df_vect_s = vectorizer_s.transform(raw_df['Summary'])
# X_submission_vect_s = vectorizer_s.transform(X_submission['Summary'])
# print("Vectorization - Merging Sparse Matrices")
# raw_df_vect = hstack((raw_df_vect, raw_df_vect_s))
# X_submission_vect = hstack((X_submission_vect, X_submission_vect_s))
# print("Vectorization - SVD...")
# svd_s_time = time.perf_counter()
# svd = TruncatedSVD(n_components=200, random_state=0)
# raw_df_vect = svd.fit_transform(raw_df_vect)
# print(svd.explained_variance_ratio_.sum())
# X_submission_vect = svd.fit_transform(X_submission_vect)
# print(svd.explained_variance_ratio_.sum())
# svd_f_time = time.perf_counter()
# print('SVD took: ' + str(svd_f_time - svd_s_time) + ' seconds')
# print("Vectorization - Creating Pandas df...")
# raw_df_df = pd.DataFrame(raw_df_vect, columns=np.arange(200)).set_index(raw_df.index.values)
# X_submission_df = pd.DataFrame(X_submission_vect, columns=np.arange(200)).set_index(X_submission.index.values)
# # raw_df_df = pd.DataFrame(raw_df_vect.toarray(), columns=vectorizer.get_feature_names()).set_index(raw_df.index.values)
# # X_submission_df = pd.DataFrame(X_submission_vect.toarray(), columns=vectorizer.get_feature_names()).set_index(X_submission.index.values)
# print("Vectorization - Joining with Original df...")
# # raw_df and X_submission below contains original columns with tfidf 
# raw_df = raw_df.join(raw_df_df)
# X_submission = X_submission.join(X_submission_df)


# # Train/Validation Split
# print("Train/Validation Split...")
# X_train, X_validation, Y_train, Y_validation = train_test_split(raw_df.drop(['Score'], axis=1), raw_df['Score'], test_size=0.20, random_state=0, stratify=raw_df['Score'])



'''
# Saving to Local
print("Saving to Local...")
X_train.to_pickle("./data/X_train.pkl")
X_validation.to_pickle("./data/X_validation.pkl")
Y_train.to_pickle("./data/Y_train.pkl")
Y_validation.to_pickle("./data/Y_validation.pkl")
X_submission.to_pickle("./data/X_submission.pkl")
'''


# Old Functions
'''
# time
tfidf_s_time = time.perf_counter()
tfidf_f_time = time.perf_counter()
print('tfidf vectorizer took: ' + str(tfidf_f_time - tfidf_s_time) + ' seconds')
'''

'''
# old WordNet lemmatizer, used list comprehension instead, performance similar
def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], tag_dict.get(x[1][0])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if len(word) <= 2:
            continue
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
'''

'''
# spaCy Tokenization and Lemmatization with noise removal below length 2. Too slow
nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"]) 
def fast_lemma(my_string):
    spacy_form = nlp(my_string)
    return(" ".join([word.lemma_ for word in spacy_form if len(word) > 2]))
testtext = fast_lemma(testtext)
print(testtext)
t1_start = time.perf_counter()
X_train['Summary'] = X_train['Summary'].apply(fast_lemma)
X_train['Text'] = X_train['Text'].apply(fast_lemma)
t1_stop = time.perf_counter()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
'''