#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: akhilesh.koul

"""


import requests
import pandas as pd
import praw
from tqdm import tqdm
import datetime
import json
import re
import time
import numpy as np
import nltk
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing
from ast import literal_eval

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors



class suggestModel():
    
    def __init__(self,df_corpus_path=None,df_bookauthor_path=None,df_merge_path=None,notebook=False):
        
        self.df_corpus_path=df_corpus_path
        self.df_text=pd.read_csv(df_corpus_path)
        self.df_text["postTitle"].replace(np.nan, "",inplace=True)
        self.df_text["text"] = self.df_text["postTitle"].astype(str) + str(" ") + self.df_text["postText"].astype(str)
        
        self.df_bookauthor_path=df_bookauthor_path
        self.df_book=pd.read_csv(df_bookauthor_path)
        self.df_book=self.df_book[self.df_book['bookPrinttype']=='BOOK']
   
        self.df_merge_path=df_merge_path
        self.notebook=notebook
       
     
        
    def cleanText(self):
       
        self.df_text["text"] = self.df_text["text"].str.lower()
        #remove_punctuation
        self.df_text["text"] = self.df_text["text"].apply(lambda text: self.remove_punctuation(text,train=True))
        #remove_stopwords
        self.df_text["text"] = self.df_text["text"].apply(lambda text: self.remove_stopwords(text,train=True)) 
        # remove_freqwords
        self.df_text["text"] = self.df_text["text"].apply(lambda text: self.remove_freqwords(text,train=True))
        #lemmatize_words
        self.df_text["text"] = self.df_text["text"].apply(lambda text: self.lemmatize_words(text))
        return self.df_text


    def remove_punctuation(self,text,train=False):
        
        if train==True:
            PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~”“`'
            if self.notebook== False:
                with open('../pkl_files/PUNCT_TO_REMOVE.pkl', 'wb') as f:
                       pickle.dump(PUNCT_TO_REMOVE, f)
            elif self.notebook == True:
                with open('pkl_files/PUNCT_TO_REMOVE.pkl', 'wb') as f:
                       pickle.dump(PUNCT_TO_REMOVE, f)           
        if self.notebook== False:
            with open('../pkl_files/PUNCT_TO_REMOVE.pkl', 'rb') as f:
                PUNCT_TO_REMOVE = pickle.load(f)   
        elif self.notebook == True:
            with open('pkl_files/PUNCT_TO_REMOVE.pkl', 'rb') as f:
                PUNCT_TO_REMOVE = pickle.load(f)        
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


    def remove_stopwords(self,text,train=False):
        STOPWORDS = set(stopwords.words('english'))
      
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
    
    def remove_freqwords(self, text,train=False):
        if train==True:
            cnt = Counter()
            for text in self.df_text["text"].values:
                for word in text.split():
                    cnt[word] += 1
            if self.notebook== False:        
                with open('../pkl_files/mostCommonWords.pkl', 'wb') as f:
                    pickle.dump(cnt.most_common(10), f)
                    
            elif self.notebook== False:
                with open('pkl_files/mostCommonWords.pkl', 'wb') as f:
                    pickle.dump(cnt.most_common(10), f)
            mostCommonWords= cnt.most_common(10)
                
            
        if self.notebook== False:  
            with open('../pkl_files/mostCommonWords.pkl', 'rb') as f:
                mostCommonWords = pickle.load(f) 
        elif self.notebook== True:  
            with open('pkl_files/mostCommonWords.pkl', 'rb') as f:
                mostCommonWords = pickle.load(f) 
        
        FREQWORDS = set([w for (w, wc) in mostCommonWords])
        return " ".join([word for word in str(text).split() if word not in FREQWORDS])

    def lemmatize_words(self,text):
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


    def trainData(self):
        self.df_text['bookID']=self.df_text['postID']
        
        self.df = pd.merge(self.df_book, self.df_text, how="left", on=["bookID"])
        self.df=self.df[self.df['bookPrinttype']=='BOOK']
        self.df.to_csv(self.df_merge_path)
        
    
        tfidfvectorizer = TfidfVectorizer()
        df_text_vector = tfidfvectorizer.fit_transform(self.df['text'])
        if self.notebook== False:  
            with open('../pkl_files/tfidfvectorizer.pkl', 'wb') as f:
                   pickle.dump(tfidfvectorizer, f)
        elif self.notebook== True:  
            with open('pkl_files/tfidfvectorizer.pkl', 'wb') as f:
                   pickle.dump(tfidfvectorizer, f)
        tfidftransformer = TfidfTransformer()
        df_text_tfidf = tfidftransformer.fit_transform(df_text_vector)
     
        if self.notebook== False: 
            with open('../pkl_files/tfidftransformer.pkl', 'wb') as f:
                   pickle.dump(tfidftransformer, f)
        elif self.notebook== True: 
            with open('pkl_files/tfidftransformer.pkl', 'wb') as f:
                   pickle.dump(tfidftransformer, f)
        neighModel = NearestNeighbors(n_neighbors=5)
        neighModel.fit(df_text_tfidf)
        if self.notebook== False: 
            with open('../pkl_files/neighModel.pkl', 'wb') as f:
                   pickle.dump(neighModel, f)
        elif self.notebook== True: 
            with open('pkl_files/neighModel.pkl', 'wb') as f:
                   pickle.dump(neighModel, f)    

    def suggestmeabook(self,text):
        
        text=self.remove_punctuation(text)
        text=self.remove_stopwords(text)
        text=self.remove_freqwords(text)
        text=self.lemmatize_words(text)
        text_list=[]
        text_list.append(text)
        df_single=pd.DataFrame()
        df_single['text']=text_list
        if self.notebook== False: 
            with open('../pkl_files/tfidfvectorizer.pkl', 'rb') as f:
                tfidfvectorizer = pickle.load(f)  
        elif self.notebook== True: 
            with open('pkl_files/tfidfvectorizer.pkl', 'rb') as f:
                tfidfvectorizer = pickle.load(f)       
        
        if self.notebook== False: 
            with open('../pkl_files/tfidftransformer.pkl', 'rb') as f:
                tfidftransformer = pickle.load(f)  
        elif self.notebook== True: 
            with open('pkl_files/tfidftransformer.pkl', 'rb') as f:
                tfidftransformer = pickle.load(f)  
                
        df_text_vector = tfidfvectorizer.transform(df_single['text'])
        df_text_tfidf = tfidftransformer.transform(df_text_vector)
        if self.notebook== False: 
            with open('../pkl_files/neighModel.pkl', 'rb') as f:
                neighModel = pickle.load(f)  
        elif self.notebook== True: 
            with open('pkl_files/neighModel.pkl', 'rb') as f:
                neighModel = pickle.load(f)  
        y_pred = neighModel.kneighbors(df_text_tfidf)
        
        df_merge=pd.read_csv(self.df_merge_path)
        df_book=pd.read_csv(self.df_bookauthor_path)
       
        for i in range(len(y_pred[1][0])):
            if i == 0:
                sample_dfs=df_book[df_book['bookID']==df_merge.iloc[y_pred[1][0][i]]['bookID']]
            else:
                sample_df=df_book[df_book['bookID']==df_merge.iloc[y_pred[1][0][i]]['bookID']]
                sample_dfs=sample_dfs.append(sample_df)
        
        first_pass=sample_dfs.drop_duplicates(subset='book', keep="first")   
        second_pass=first_pass.loc[first_pass.book.str.extract(r'(.*):').drop_duplicates().index]
        second_pass.dropna(subset=['bookImage'],inplace=True)
        
        sampleFive=second_pass.sample(n=5)
        sampleFive.reset_index(drop=True,inplace=True)
        
        for i in range(len(sampleFive)):
            print(sampleFive.iloc[i]['book'])
            response = requests.get(sampleFive.iloc[i]['bookImage'])
            img = Image.open(BytesIO(response.content))
            imgplt=plt.imshow(img)
            plt.show()
            
            
        # return sampleFive    
        



if __name__ == '__main__':
    
    suggestModelClass=suggestModel(df_corpus_path='../data/df_corpus.csv',df_bookauthor_path='../data/df_bookauthor.csv',df_merge_path='../data/df_merge.csv',notebook=False)
    df_clean=suggestModelClass.cleanText()
    suggestModelClass.trainData()
    text= 'books like fiction, like the alchemist and the hundred years of solitude'
    suggestModelClass.suggestmeabook(text)
    
