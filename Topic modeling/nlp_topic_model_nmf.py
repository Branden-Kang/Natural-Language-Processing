# -*- coding: utf-8 -*-
"""NLP Topic Model NMF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fNy3riKqpYeLLfDzVMToX3a9DvIAmDFC

[Reference](https://medium.com/@obianuju.c.okafor/automatic-topic-classification-of-research-papers-using-the-nlp-topic-model-nmf-d4365987ec82)

![Step](https://miro.medium.com/max/1400/1*wKi39V-CUbtjr5cvW7a6tQ.png)
"""

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import re
import gensim
import gensim.corpora as corpora
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
!pip install pyLDAvis
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda

#Load data file
dataset = pd.read_csv(r'https://raw.githubusercontent.com/Branden-Kang/Python-practice/master/Data/research_papers.csv', encoding='ISO-8859–1')
dataset.head()

"""# Clean Data"""

#Remove the unecessary columns
dataset = dataset.drop(columns=['Id', 'Reference', 'Codes', 'Authors', 'Year', 'Conference/ Journal'], axis=1)
#Fill in the empty cells
dataset = dataset.fillna('No conclusion')
#Merge abstract and conclusion
dataset['Paper_Text'] = dataset["Abstract"] + dataset["Conclusion"]
#show first 5 records
dataset.head()

"""# Preprocess Data"""

#function for lemmatization
def get_lemma(word):
  lemma = wn.morphy(word)
  if lemma is None:
    return word
  else:
    return lemma# tokenization
tokenized_data = dataset['Paper_Text'].apply(lambda x: x.split())# Remove punctuation
tokenized_data = tokenized_data.apply(lambda x: [re.sub('[-,()\\!?]', '', item) for item in x])
tokenized_data = tokenized_data.apply(lambda x: [re.sub('[.]', ' ', item) for item in x])# turn characters to lowercase
tokenized_data = tokenized_data.apply(lambda x: [item.lower() for item in x])# remove stop-words
stop_words = stopwords.words('english')
stop_words.extend(['from','use', 'using','uses','user', 'users', 'well', 'study', 'survey', 'think'])# remove words of length less than 3
tokenized_data = tokenized_data.apply(lambda x: [item for item in x if item not in stop_words and len(item)>3])# lemmatize by calling lemmatization function
tokenized_data= tokenized_data.apply(lambda x: [get_lemma(item) for item in x])

"""# Create Bigrams and Trigrams"""

# Build the bigram and trigram models
bigram = gensim.models.Phrases(tokenized_data, min_count=5, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[tokenized_data], threshold=10)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Define functions for creating bigrams and trigrams.
def make_bigrams(texts):
  return [bigram_mod[doc] for doc in texts]
  
def make_trigrams(texts):
  return [trigram_mod[bigram_mod[doc]] for doc in texts]
  
# Form Bigrams
tokenized_data_bigrams = make_bigrams(tokenized_data)

# Form Trigrams
tokenized_data_trigrams = make_trigrams(tokenized_data)

# de-tokenization, combine tokens together
detokenized_data = []
for i in range(len(dataset)):
    t = ' '.join(tokenized_data_trigrams[i])
    detokenized_data.append(t)

dataset['clean_text']= detokenized_data
documents = dataset['clean_text']

"""# Create Document-Term Matrix"""

#Set variable number of terms
no_terms = 1000

# NMF uses the tf-idf count vectorizer
# Initialise the count vectorizer with the English stop words
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=no_terms, stop_words='english')

# Fit and transform the text
document_matrix = vectorizer.fit_transform(documents)

#get features
feature_names = vectorizer.get_feature_names()

"""# Generate topics using Topic Model"""

#Set variables umber of topics and top words.
no_topics = 10
no_top_words = 10

# Function for displaying topics
def display_topic(model, feature_names, num_topics, no_top_words, model_name):
    print("Model Result:")
    word_dict = {}
    for i in range(num_topics):
      #for each topic, obtain the largest values, and add the words they map to into the dictionary.
       words_ids = model.components_[i].argsort()[:-no_top_words - 1:-1]
       words = [feature_names[key] for key in words_ids]
       word_dict['Topic # ' + '{:02d}'.format(i)] = words    
       
    dict = pd.DataFrame(word_dict)
    dict.to_csv('%s.csv' % model_name)
    return dict

# Apply NMF topic model to document-term matrix
nmf_model = NMF(n_components=no_topics, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd').fit(document_matrix)

#display topics for nmf model
display_topic(nmf_model, feature_names, no_topics, no_top_words, 'NMF_Model_Result')

"""# Analyzing our NMF model"""

# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = sklearn_lda.prepare(nmf_model, document_matrix, vectorizer)
LDAvis_prepared

"""# Classify papers under topics"""

#Use NMF model to assign topic to papers in corpus
nmf_topic_values = nmf_model.transform(document_matrix)
dataset['NMF Topic'] = nmf_topic_values.argmax(axis=1)

#Save dataframe to csv file
# dataset.to_csv('final_results.csv')
dataset.head(10)