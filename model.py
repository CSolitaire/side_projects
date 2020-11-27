from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

######################## Logistic Regression ##########################

def logistic_regression(X_train, y_train, X_train_bow, X_train_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs logistic
    regression giving us accuracy of the model and the classification report
    '''
    # create the object
    lm = LogisticRegression()
    # fit the object
    lm_bow = lm.fit(X_train_bow, y_train)
    lm_tfidf = lm.fit(X_train_tfidf, y_train)
    # make predictions
    X_train['pred_bow'] = lm_bow.predict(X_train_bow)
    X_train['pred_tfidf'] = lm_tfidf.predict(X_train_tfidf)
    
    # X_bow results
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.pred_bow)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_bow)}\n' )
    print('-----------------------')
    print("X_bow Logistic Regression Classification Report:\n", classification_report(y_train, X_train.pred_bow))

    # TF-IDF results
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Logistic Regression Classification Report:\n", classification_report(y_train, X_train.pred_tfidf))
    return lm_bow, lm_tfidf


######################## Random Forest ##########################

def random_forest(X_train, y_train, X_train_bow, X_train_tfidf):
    # Random forest object
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=123)
    # Fitting the data to the train data
    rf_bow = rf.fit(X_train_bow, y_train)
    rf_tfidf = rf.fit(X_train_tfidf, y_train)
    # make predictions
    X_train['pred_bow'] = rf_bow.predict(X_train_bow)
    X_train['pred_tfidf'] = rf_tfidf.predict(X_train_tfidf)

    # BOW results
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_train.language, X_train.pred_bow)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_bow)}\n' )
    print('-----------------------')
    print("X_bow Random Forest Classification Report:\n", classification_report(y_train.language, X_train.pred_bow))

   # TF-IDF results
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train.language, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Random Forest Classification Report:\n", classification_report(y_train.language, X_train.pred_tfidf))
    return rf_bow, rf_tfidf


######################## Complement Naive Bayes ##########################

def complement_naive_bayes(X_train, y_train, X_train_tfidf):
    
    # create object, fit, and predict
    cnb = ComplementNB(alpha=1.0)
    cnb_tfidf = cnb.fit(X_train_tfidf, y_train)
    X_train['pred_tfidf'] = cnb_tfidf.predict(X_train_tfidf)

    # TF-IDF results
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Complement Niave Bayes Classification Report:\n", classification_report(y_train, X_train.pred_tfidf))
    return cnb_tfidf
 
######################## Validate Logistic Regression ##########################

def validate_logistic_regression(X_validate, y_validate, X_val_bow, X_val_tfidf, lm_bow, lm_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs logistic
    regression giving us accuracy of the model and the classification report
    '''
    # create predictions
    X_validate['pred_bow'] = lm_bow.predict(X_val_bow)
    X_validate['pred_tfidf'] = lm_tfidf.predict(X_val_tfidf)

    # X_bow results
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_validate, X_validate.pred_bow)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_bow)}\n' )
    print('-----------------------')
    print("X_bow Logistic Regression Classification Report:\n", classification_report(y_validate, X_validate.pred_bow))

    # TF-IDF results
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_validate, X_validate.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Logistic Regression Classification Report:\n", classification_report(y_validate, X_validate.pred_tfidf))

######################## Validate Random Forest ##########################

def validate_random_forest(X_validate, y_validate, X_val_bow, X_val_tfidf, rf_bow, rf_tfidf):
    # create predictions
    X_validate['pred_bow'] = rf_bow.predict(X_val_bow)
    X_validate['pred_tfidf'] = rf_tfidf.predict(X_val_tfidf)
    
    # X_bow results
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_validate.language, X_validate.pred_bow)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_bow)}\n' )
    print('-----------------------')
    print("X_bow Random Forest Classification Report:\n", classification_report(y_validate.language, X_validate.pred_bow))

   # TF-IDF results
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_validate.language, X_validate.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Random Forest Classification Report:\n", classification_report(y_validate.language, X_validate.pred_tfidf))

######################## Validate Complement Naive Bayes ##########################

def validate_complement_naive_bayes(X_validate, y_validate, X_val_tfidf, cnb_tfidf):
    # makes predictions
    X_validate['pred_tfidf'] = cnb_tfidf.predict(X_val_tfidf)

    # TF-IDF results
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_validate, X_validate.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Complement Niave Bayes Classification Report:\n", classification_report(y_validate, X_validate.pred_tfidf))

######################## Test  ##########################

def test_random_forest(X_test, y_test, X_test_tfidf, rf_tfidf):
    # Creaye predictions
    X_test['pred_tfidf'] = rf_tfidf.predict(X_test_tfidf)

   # Confusion matrix
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_test.language, X_test.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_test.language, X_test.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Random Forest Classification Report:\n", classification_report(y_test.language, X_test.pred_tfidf))

def test_logistic_regression(X_test, y_test, X_test_bow, lm_bow):
    # Create prediction
    X_test['pred_bow'] = lm_bow.predict(X_test_bow)

    # X_bow results
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_test, X_test.pred_bow)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_test.language, X_test.pred_bow)}\n' )
    print('-----------------------')
    print("X_bow Logistic Regression Classification Report:\n", classification_report(y_test, X_test.pred_bow))