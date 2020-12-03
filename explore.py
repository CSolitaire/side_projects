import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import re
import nltk


def set_plotting_defaults():
    '''
    sets default sizes for plots
    '''
    # plotting defaults
    plt.rc('figure', figsize=(13, 7))
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', size=16)

####### NLP ########

def check_proportion(train_exp):
    '''
    creates dataframe that shows the proportion of observations in each language
    '''
    labels = pd.concat([train_exp.language.value_counts(), # get total counts of languages
                        train_exp.language.value_counts(normalize=True)], axis=1) # getting the prop of languages
    labels.columns = ['n', 'percent']
    return labels


def create_lang_word_list(train_exp):
    '''
    creates a list of words in the readme text by language and removes single letter words
    '''
    # create a list of words for each language category
    jupyter_words = ' '.join(train_exp[train_exp.language=='Jupyter'].lemmatized)
    js_words = ' '.join(train_exp[train_exp.language=='JavaScript'].lemmatized)
    r_words = ' '.join(train_exp[train_exp.language=='R'].lemmatized)

    # remove single letter words to reduce noise
    jupyter_words = re.sub(r'\s.\s', '', jupyter_words)
    js_words = re.sub(r'\s.\s', '', js_words)
    r_words = re.sub(r'\s.\s', '', r_words)
    return jupyter_words, js_words, r_words


def get_count_word_freq(jupyter_words, js_words, r_words):
    '''
    split the list of words and get frequency count
    '''
    # get the count of words by category
    jupyter_freq = pd.Series(jupyter_words.split()).value_counts()
    js_freq = pd.Series(js_words.split()).value_counts()
    r_freq = pd.Series(r_words.split()).value_counts()
    return jupyter_freq, js_freq, r_freq


def create_df_word_counts(jupyter_freq, js_freq, r_freq):
    '''
    combines the frequencies to create new dataframe, word_counts
    '''
    # combine list of word counts into df for further exploration
    word_counts = (pd.concat([jupyter_freq, js_freq, r_freq], axis=1, sort=True)
                .set_axis(['Jupyter', 'JavaScript', 'R'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int))
                )
    # create a column of all words as well
    word_counts['all_words'] = word_counts['Jupyter'] + word_counts['JavaScript'] + word_counts['R'] 
    return word_counts


def word_counts_proportion(word_counts):
    '''
    compute proportion of each string that for each language
    '''
    # add columns for each langauge proportion
    word_counts['prop_jupyter'] = word_counts['Jupyter']/word_counts['all_words']
    word_counts['prop_js'] = word_counts['JavaScript']/word_counts['all_words']
    word_counts['prop_r'] = word_counts['R']/word_counts['all_words']
    return word_counts


def proportion_visualization(word_counts):
    '''
    creates a plot that shows the distribution of the top 20 words by language
    '''
    ## visualize the % of the term in each language
    plt.figure(figsize=(12,8))
    (word_counts
    .assign(p_jupyter=word_counts.Jupyter / word_counts['all_words'],
            p_js=word_counts.JavaScript / word_counts['all_words'],
            p_r=word_counts.R / word_counts['all_words']
            )
    .sort_values(by='all_words')
    [['p_jupyter', 'p_js', 'p_r']]
    .tail(20)
    .sort_values('p_jupyter')
    .plot.barh(stacked=True))

    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title('Distribution of languages for the 20 most common words')
    plt.xlabel('Distribution')
    plt.ylabel('Words')
    plt.show()


def create_bigrams(jupyter_words, js_words, r_words):
    '''
    creates bigrams for each language
    '''
    # create bigrams by category
    jupyter_bigrams = pd.Series(list(nltk.ngrams(jupyter_words.split(), 2))).value_counts().head(20)
    js_bigrams = pd.Series(list(nltk.ngrams(js_words.split(), 2))).value_counts().head(20)
    r_bigrams = pd.Series(list(nltk.ngrams(r_words.split(), 2))).value_counts().head(20)
    return jupyter_bigrams, js_bigrams, r_bigrams


# def plot_bigrams(jupyter_bigrams, js_bigrams, r_bigrams:
#     '''
#     creates subplots of bigrams for each category
#     '''
#     # plot bigrams
#     plt.subplot(4,2,1)
#     jupyter_bigrams.plot.barh(color='red', width=.9, figsize=(10,10))
#     plt.title("20 most frequent Jupyter bigrams")
#     plt.ylabel("Bigram")
#     plt.xlabel("Frequency")

#     plt.subplot(4,2,2)
#     js_bigrams.plot.barh(color='orange', width=.9, figsize=(10,10))
#     plt.title("20 most frequent JavaScript bigrams")
#     plt.ylabel("Bigram")
#     plt.xlabel("Frequency")

#     plt.subplot(4,2,3)
#     r_bigrams.plot.barh(color='blue', width=.9, figsize=(10,10))
#     plt.title("20 most frequent R bigrams")
#     plt.ylabel("Bigram")
#     plt.xlabel("Frequency")

#     plt.tight_layout()
#     plt.show()


def sns_boxplot(train_exp):
    '''create boxplot for exploration'''
    sns.boxplot(data=train_exp, x='language', y='doc_length')
    plt.title('Average Readme length by Language')
    plt.show()


def ttest_1samp(sample, overall_mean):
    '''
    runs one sample t-test
    '''
    # set alpha value
    alpha = .05
    t, p = stats.ttest_1samp(sample, overall_mean)
    print(p < alpha)
    return t, p


def pearson(continuous_var1, continuous_var2):
    '''
    runs pearson r test on 2 continuous variables
    '''
    alpha = .05
    r, p = stats.pearsonr(continuous_var1, continuous_var2)
    # print('r=', r)
    # print('p=', p)
    # if p < alpha:
    #     print("We reject the null hypothesis")
    # else:
    #     print("We fail to reject the null hypothesis")
    return r, p

def chi2test(categorical_var1, categorical_var2):
    '''
    runs chi squared test on 2 categorical variables
    '''
    alpha = 0.05
    contingency_table = pd.crosstab(categorical_var1, categorical_var2)

    chi2, p, degf, expected = stats.chi2_contingency(contingency_table)

    if p < alpha:
        print("We reject the null hypothesis")
        print(f'p     = {p:.4f}')
    else:
        print("We fail to reject the null hypothesis")
    return p