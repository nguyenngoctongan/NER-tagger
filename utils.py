
# coding: utf-8

# In[1]:


import spacy
from spacy.tokens import Doc
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# In[2]:


def data_transform(file):
    """
Read file, change file format into Python object
"""
    with open (file, "r", encoding = "utf-8") as infile:
        lines = infile.readlines()
        token =[]
        pos = []
        phrase = []
        label = []
        for line in lines:
            items = line.strip("\n").split(" ")
            emptylist = ['']
            if items != emptylist:
                token.append(items[0])
                pos.append(items[1])
                phrase.append(items[2])
                label.append(items[3])
        data_structure = {"tokens": token,
                     "pos": pos,
                     "phrase": phrase,
                     "labels": label}
        data = pd.DataFrame(data_structure)
        data.drop(data.loc[data["tokens"] == "-DOCSTART-" ].index, inplace=True)
        training_file = "ner_training.csv"
        data.to_csv(training_file, sep = "\t")
    
    return training_file


# In[3]:


def data_process(training_file):
    """
    Process data, extract linguistic information from data with Spacy
    
"""
    
    read_file = pd.read_csv(training_file, sep = "\t")
    token_list = []
    label_list = list(read_file["labels"])
    for token in list(read_file["tokens"]):
        token_list.append(str(token))
    
    nlp = spacy.load('en_core_web_sm')
    doc = Doc(nlp.vocab, words= token_list)
    lemma = []
    is_low= []
    pos_tag = []
    fg_tag = []
    dep_re = []
    get_h = []
    is_al = []
    is_dgt = []
    lik_num = []
    for name, proc in nlp.pipeline:
        doc = proc(doc)
        for token in doc:
            if name == "parser":
                lemma.append(token.lemma_) #get lemma of each token
                is_low.append(token.is_lower) #check if token capitalised or not
                pos_tag.append(token.pos_) #get part-of-speech of each token
                dep_re.append(token.dep_) #get syntactic dependency relation of each token
                get_h.append(token.head.lemma_) #get the head of the dependency
                is_al.append(token.is_alpha) #check if token contains all alphabetical characters
                is_dgt.append(token.is_digit) #check if token is number or not
                lik_num.append(token.like_num) #check if token contain numbers
            elif name == "tagger":
                fg_tag.append(token.tag_) #get fine grained part-of-speech of each token
                
    data_structure = {"tokens":token_list, 
           "lemmas": lemma,
            "get_head": get_h,
           "pos_tags": pos_tag,
           "tags": fg_tag,
           "dependencies": dep_re,
           "uncapitalize": is_low,
           "is_alpha": is_al,
           "is_digit": is_dgt,
           "like_number": lik_num,
          "labels": label_list}
    processed_data = pd.DataFrame(data_structure)
    processed_file = "processed.csv"
    processed_data.to_csv(processed_file, sep = "\t")
    
    return processed_file


# In[4]:


def split_sentences(token_list):
    """
    Turn a list of tokens into a list of sentences
    """

    #Find index of each splitter value in the list
    sentence_end = [".", "!","?"]
    
    split_position = []
    for index, token in enumerate(token_list):
        if token in sentence_end:
            try:
                split_position.append(index)
            except ValueError:
            #Splitter not found in list
                pass
            
 #Split the iterable into sublists based on indices
    sentence_list = []
    start = 0
    for index in sorted(split_position):
        sentence_list.append(token_list[start:index+1])
        start = index + 1
    sentence_list.append(token_list[start:])

    return sentence_list


# In[5]:


def training_w2v(token_list):
    """
    Training word2vect model with a list of tokens
    """
    sentence_list = split_sentences(token_list)
    
    path = get_tmpfile("word2vec.model")
    
    model = Word2Vec(sentence_list, size = 100, window=3, min_count=1, workers=4, sg=1)
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
    vectoriser = model.wv
    
    return vectoriser


# In[6]:


def data_vectorise(processed_file):
    """
    Turn processed data into vectors
    """
    read_file = pd.read_csv(processed_file, sep = "\t")
    word_list = []
    for lemma in list(read_file["lemmas"]):
        word = str(lemma)
        word_list.append(word)
    
    w_vectoriser = training_w2v(word_list)
    
    dv = DictVectorizer(sparse = False)
    
    X = []
    y = read_file["labels"]
    for index, row in read_file.iterrows():
        
        tokenvect = w_vectoriser.word_vec(str(row["lemmas"]))
        headvect = w_vectoriser.word_vec(str(row["get_head"]))
    
        features = row.iloc[4:11]
        feature_dict = features.to_dict()
    
        featvect = dv.fit_transform(feature_dict).flatten()

        row_vec = np.concatenate([tokenvect, headvect, featvect])

        X.append(row_vec)
    return X, y


# In[7]:


def count_labels(y):
    """
    Count the number of each label
    """
    count_dict = {}
    for label in y:
        if label in count_dict:
            count_dict[label] += 1
        if label not in count_dict:
            count_dict[label] = 1
    print(count_dict)


# In[8]:


def classify(X_train, y_train, X_test):
    """
    Training svm classifier using GridSearch for parameter tunning
    """
    svm_clf = svm.SVC(class_weight = 'balanced', decision_function_shape = "ovo")
    
    Cs = [9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
    gammas =   [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    kernels = ['linear','rbf']
    
    param_grid = {'kernel': kernels,'C': Cs, 'gamma' : gammas}
    
    grid_search = GridSearchCV(svm_clf, param_grid, cv = 3)
    grid_search.fit(X_train, y_train)
    
    print("Best prameters: ", grid_search.best_params_)
    print(grid_search)
    
    
    y_pred = grid_search.predict(X_test)
    
    return y_pred


# In[9]:


def evaluate(y_test, y_pred, label_list):
    """
    Evaluate the model
    """
   
    print(f"SVM:\n {metrics.classification_report(y_pred, y_test)}")   
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("")
    
    cof_matr_svm = confusion_matrix(y_test, y_pred, labels=label_list)
    
    print("Confusion matrix:\n", pd.DataFrame(cof_matr_svm, index = set(label_list), columns = set(label_list)))

