import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import acquire



def basic_clean(text):
    '''
    Initial basic cleaning/normalization of text string
    '''
    # change to all lowercase
    low_case = text.lower()
    # remove special characters, encode to ascii and recode to utf-8
    recode = unicodedata.normalize('NFKD', low_case).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Replace anything that is not a letter, number, whitespace or a single quote
    cleaned = re.sub(r"[^a-z0-9'\s]", '', recode)
    # Remove numbers from text
    cleaned = re.sub(r'\d+', '', cleaned)
    return cleaned

def tokenize(text):
    '''
    Use NLTK TlktokTokenizer to seperate/tokenize text
    '''
    # create the NLTK tokenizer object
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(text, return_str=True)

def stem(text):
    '''
    Apply NLTK stemming to text to remove prefix and suffixes
    '''
    # Create the nltk stemmer object, then use it
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in text.split()]
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmatize(text):
    '''
    Apply NLTK lemmatizing to text to remove prefix and suffixes
    '''
    # Create the nltk lemmatize object, then use it
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized

def remove_stopwords(text, extra_words=[], exclude_words=["accuracy","accurate","acknowledgement",
    "acquired","adhere","adheres","along","also","accessing","provided","directory", "datasets",
    "including","pip","problem","pull","python" "according","account","across","addition","additional","aim",
    "anaconda","analyse","analysed","analysis","andor","answer","according","public","proxy","properly","prompt","program",
    "produce", "produced","reference","need","optimization","viewer","open","norequestrendermode","preform","performed","point","pointcloud",
    "question", "install","data","import","using","please","different","installation","preprocessing","library","type","implemented","minimum","github",
    "run","using","file","license","result","record","possible","number","could","see","use","source","define","parameter","geology","disable",
    "limit","inspector","quality","default","display", "activate","alternative","anacona","associated","assumption","attempt","author","automated",
    "automatic","available","purpose","published","publically","prevented","different","navigation","quality",'nolimit',"package",
    "al","application","assetids","based","better","bias","biased","birthdeath","raw","rather","present","prerequisitewindowv","particular",
    "partially","oxford","requires","happy","request","report","high","provide","project","processing","potential","calculated","cannot","capture",
    "carried","cause",'cc','cd']):
    '''
    Removes stopwords from text, allows for additional words to exclude, or words to not exclude
    '''
    # define initial stopwords list
    stopword_list = stopwords.words('english')
    # add additional stopwords
    for word in extra_words:
        stopword_list.remove(word)
    # remove stopwords to exclude from stopword list
    for word in exclude_words:
        stopword_list.append(word)
    # split the string into words
    words = text.split()
    # filter the words
    filtered_words = [w for w in words if w not in stopword_list]
    # print number of stopwords removed
    # print('Removed {} stopwords'.format(len(words) - len(filtered_words)))
    # produce string without stopwords
    article_without_stopwords = ' '.join(filtered_words)
    return article_without_stopwords


###### my GitHub version
def prep_data(df, column):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    # Formatts language makes it easier to read
    df['text_cleaned'] = df.content.apply(basic_clean)
    df['text_tokenized'] = df.text_cleaned.apply(tokenize)
    df['text_lemmatized'] = df.text_tokenized.apply(lemmatize)
    df['text_filtered'] = df.text_lemmatized.apply(remove_stopwords)
    # Add column with list of words
    words = [re.sub(r'([^a-z0-9\s]|\s.\s)', '', doc).split() for doc in df.text_filtered]
    df = pd.concat([df, pd.DataFrame({'words': words})], axis=1)
    # Adds colum with lenght of word list
    df['doc_length'] = [len(wordlist) for wordlist in df.words]
      # removing unpopular languages 
    #language_list = ['JavaScript', 'Java', 'HTML', 'Python']
    #df = df[df.language.isin(language_list)]
    return df

  