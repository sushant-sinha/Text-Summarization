import nltk
DOCUMENT = """
China net cafe culture crackdown

Chinese authorities closed 12,575 net cafes in the closing months of 2004, the country's government said.

According to the official news agency most of the net cafes were closed down because they were operating illegally. Chinese net cafes operate under a set of strict guidelines and
many of those most recently closed broke rules that limit how close they can be to schools. The move is the latest in a series of steps the Chinese government has taken to crack
down on what it considers to be immoral net use.

The official Xinhua News Agency said the crackdown was carried out to create a "safer environment for young people in China". Rules introduced in 2002 demand that net cafes be
at least 200 metres away from middle and elementary schools. The hours that children can use net cafes are also tightly regulated. China has long been worried that net cafes are
an unhealthy influence on young people. The 12,575 cafes were shut in the three months from October to December. China also tries to dictate the types of computer games people
can play to limit the amount of violence people are exposed to.


Net cafes are hugely popular in China because the relatively high cost of computer hardware means that few people have PCs in their homes. This is not the first time that the
Chinese government has moved against net cafes that are not operating within its strict guidelines. All the 100,000 or so net cafes in the country are required to use software
that controls what websites users can see. Logs of sites people visit are also kept. Laws on net cafe opening hours and who can use them were introduced in 2002 following a
fire at one cafe that killed 25 people. During the crackdown following the blaze authorities moved to clean up net cafes and demanded that all of them get permits to operate.
In August 2004 Chinese authorities shut down 700 websites and arrested 224 people in a crackdown on net porn. At the same time it introduced new controls to block overseas sex
sites. The Reporters Without Borders group said in a report that Chinese government technologies for e-mail interception and net censorship are among the most highly developed
in the world.

""";
print("The Original Document:",DOCUMENT)
print(' ')
L1=len(DOCUMENT)
c0=0
c1=0
for i in range (0,L1):
    if(DOCUMENT[i]==' '):
        c1=c1+1
    elif(DOCUMENT[i]=='.'):
        c0=c0+1
print("Length of original document=",c1," words")
print("The number of sentences in the original article=",c0,' ')

#TEXT PRE-PROCESSING
import re#re is abuilt-in package which can be used to work with Regular Expressions. This module provides regular expression matching operations.
DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)#removal of the line breaks or paragraph seperators
DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
DOCUMENT = DOCUMENT.strip()#removes spaces at the begining and end of the input article
sentences = nltk.sent_tokenize(DOCUMENT)#forms an array of sentences. These sentences arre those which are present in the input article seperated by a '.'

import numpy as np
stop_words = nltk.corpus.stopwords.words('english')#returns the array of designated stopwords defined by nltk

def normalize_document(doc):
    
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)#removes special characters (mainly the puncuation marks)
    doc = doc.lower()#entire document is converted into lower case
    doc = doc.strip()#removes whitespace from the beginning/end of the document
    tokens = nltk.word_tokenize(doc)#tokenize the document. Returns the array of all the words present in the input article 
    filtered_tokens = [token for token in tokens if token not in stop_words]#filtering of the stopwords from the array of tokenized document
    doc = ' '.join(filtered_tokens)# re-create document from filtered tokens
    return doc
normalize_corpus = np.vectorize(normalize_document)#normalization of the document

norm_sentences = normalize_corpus(sentences)
print("The processed text:",' ')
print(norm_sentences)

#FEATURE EXTRACTION
from sklearn.feature_extraction.text import TfidfVectorizer#Used for geerating TF-IDF scores
import pandas as pd
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)#Convert a collection of sentences in the normalized document to a matrix of TF-IDF features
dt_matrix = tv.fit_transform(norm_sentences)
dt_matrix = dt_matrix.toarray()
vocab = tv.get_feature_names()#Array of words (in the normalized document) sorted on the basis of their respective TF-IDF features
td_matrix = dt_matrix.T#Transpose of matrix dt_matrix
td_matrix.shape
m=int(input("Enter the value of 'm'"))
print("The selection matrix on vectorization for top 'm' words:,':",' ')
print(pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(m))

from scipy.sparse.linalg import svds   
def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt
n=int(input("Enter the number of segments for summary:"))
num_sentences = int(input("Enter the number of sentences you require in the summary:"))
num_topics = n
u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
u.shape, s.shape, vt.shape
term_topic_mat, singular_values, topic_document_mat = u, s, vt
sv_threshold = 0.5
min_sigma_value = max(singular_values) * sv_threshold
singular_values[singular_values < min_sigma_value] = 0
salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))
print("Saline scores:",' ')
print(salience_scores)

top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
print(top_sentence_indices.sort())
similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
similarity_matrix.shape
print("The similarity matrix:",' ')
print(np.round(similarity_matrix, 3))

import networkx
similarity_graph = networkx.from_numpy_array(similarity_matrix)
similarity_graph
scores = networkx.pagerank(similarity_graph)
ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
ranked_sentences[:10]
top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
top_sentence_indices.sort()


print(' ')
print(' ')
print("The summary:",' ')
print(' ')
print('\n'.join(np.array(sentences)[top_sentence_indices]))
print(' ')
L2=len('\n'.join(np.array(sentences)[top_sentence_indices]))
c2=0
for i in range (0,L2):
    if(DOCUMENT[i]==' '):
        c2=c2+1
print("Length of summary=",c2," words")

from rouge import Rouge
human_summary="""
The Kyrgyz Republic, a small, mountainous state of the former Soviet republic, is using invisible ink and ultraviolet readers in the country's elections as part of a drive to
prevent multiple voting. In an effort to live up to its reputation in the 1990s as "an island of democracy", the Kyrgyz President, Askar Akaev, pushed through the law requiring
the use of ink during the upcoming Parliamentary and Presidential elections. The use of ink is only one part of a general effort to show commitment towards more open
elections - the German Embassy, the Soros Foundation and the Kyrgyz government have all contributed to purchase transparent ballot boxes. At the entrance to each polling station,
one election official will scan voter's fingers with UV lamp before allowing them to enter, and every voter will have his/her left thumb sprayed with ink before receiving the
ballot.
""";
hypothesis = '\n'.join(np.array(sentences)[top_sentence_indices])
reference =human_summary
rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
score = scorer.score(hypothesis,reference)
print(score)
