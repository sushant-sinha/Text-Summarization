Following we are presenting an aprroach to summarize the document using automated techniques.
we have used extractive-based techniques and the name itself justifies that no new content is being generated.

For running the below code, it can be easily executed on any local ide with the suitable dependencies already present.

There is also a more convenient way of involving the use of Google Colabratory.

if using the second option, the user can just paste the code run without any issues as the required dependencies will be downloaded accordingly.

for grammatical errors one needs to use the steps shown by https://www.geeksforgeeks.org/grammar-checker-in-python-using-language-check/ on his local machine
we have displayed the rouge, cosine values directly for the google colab platform.


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

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer#Used for geerating TF-IDF scores
import pandas as pd
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)#Convert a collection of sentences in the normalized document to a matrix of TF-IDF features
dt_matrix = tv.fit_transform(norm_sentences)
dt_matrix = dt_matrix.toarray()
vocab = tv.get_feature_names()#Array of words (in the normalized document) sorted on the basis of their respective TF-IDF features
td_matrix = dt_matrix.T#Transpose of matrix dt_matrix
td_matrix.shape
m=int(input("Enter the value of 'm':"))
print("The selection matrix on vectorization for top 'm' words:,':",' ')
print(pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(m))

#Latent Semantic Analysis
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
final_summary='\n'.join(np.array(sentences)[top_sentence_indices])
print(final_summary)
print(' ')
L2=len(final_summary)
c2=0
for i in range (0,L2):
    if(DOCUMENT[i]==' '):
        c2=c2+1
print("Length of summary=",c2," words")

#Rouge score calculation. For Rouge score calculation, the human generated summary is needed for comparison with the proposed architecture generated one.
#You can acquire the Original docmument and the corresponding human summary of that document from the Data Set server.
!git clone https://github.com/pltrdy/rouge
%cd rouge
!python setup.py install 
from rouge import Rouge
human_summary="""
Chinese authorities closed 12,575 net cafes in the closing months of 2004, the country's government said.
Chinese net cafes operate under a set of strict guidelines and many of those most recently closed broke rules that limit how close they can be to schools.
This is not the first time that the Chinese government has moved against net cafes that are not operating within its strict guidelines.
""";
hypothesis =final_summary ;
reference =human_summary
rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

X = DOCUMENT.lower() 
Y = final_summary.lower() 

X_list = word_tokenize(X)  
Y_list = word_tokenize(Y) 

sw = stopwords.words('english')  
l1 =[];l2 =[] 

X_set = {w for w in X_list if not w in sw}  
Y_set = {w for w in Y_list if not w in sw} 

rvector = X_set.union(Y_set)  
for w in rvector: 
    if w in X_set: l1.append(1)
    else: l1.append(0) 
    if w in Y_set: l2.append(1) 
    else: l2.append(0) 
c = 0

for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
cosine = c / float((sum(l1)*sum(l2))**0.5) 
print("Cosine Similarity: ", cosine)
