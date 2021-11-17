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


class RedditCorpus():
    
    def __init__(self,start_time=None, end_time=None,deltaDay=1,subreddit=None,PostScoreThresh=10,CommentScoreThresh=3,saveLocal=True):
        self.start_time=start_time
        self.end_time=end_time
        self.deltaDay=deltaDay
        self.subreddit=subreddit
        self.PostScoreThresh=PostScoreThresh
        self.CommentScoreThresh=CommentScoreThresh
        self.saveLocal=saveLocal
        
    def getRedditPost(self):
        self.df_corpus=pd.DataFrame()
        postTitle=[]
        postText=[]
        postScore=[]
        postID=[]
        postURL=[]

        self.df_comments=pd.DataFrame()
        commentID=[]
        commentScore=[]
        comment=[]
        
        start_time=self.start_time
        start_time_unix=int(start_time.timestamp())
        end_time=start_time+datetime.timedelta(days=self.deltaDay)
        end_time_unix=int(end_time.timestamp())

        range_=(self.end_time - self.start_time)
        pbar = tqdm(total=int(range_.days/self.deltaDay))

        while end_time <= self.end_time:
            
            redditAPI="https://api.pushshift.io/reddit/search/submission/?subreddit="+str(self.subreddit)+"&after="+str(start_time_unix)+"&before="+str(end_time_unix)+"&score=>"+str(self.PostScoreThresh)+"&size=500"
           
            try:
                postjson=requests.get(redditAPI).json()
                if len(postjson['data'])==0:
                    pass;
                if len(postjson['data'])>0:
                    for i in range(len(postjson['data'])):
                        postTitle.append(postjson['data'][i]['title'])
                        try:
                            postText.append(postjson['data'][i]['selftext'])
                        except:
                            postText.append('NA')
                            
                        postScore.append(postjson['data'][i]['score'])
                        postID.append(postjson['data'][i]['id'])
                        postURL.append(postjson['data'][i]['full_link'])
                        try:
                            redditPostCmntListAPI="https://api.pushshift.io/reddit/submission/comment_ids/"+str(postjson['data'][i]['id'])
                            cmntListjson=requests.get(redditPostCmntListAPI).json()['data']
                            
                            cmntList = [str(element) for element in cmntListjson]
                            cmntListChunks = [cmntList[x:x+100] for x in range(0, len(cmntList), 100)]
                            
                            for k in range(len(cmntListChunks)):
                                
                                cmntIDListSTR = ",".join(cmntListChunks[k])
                               
                                redditPostCmntAPI="https://api.pushshift.io/reddit/comment/search?ids="+str(cmntIDListSTR)+"&sort=desc&sort_type=score"
                                
                                try:
                                    cmntJson=requests.get(redditPostCmntAPI).json()['data']
                                    cmntJsonFilter = [x for x in cmntJson if x['score'] >= self.CommentScoreThresh]
                                    if len(cmntJsonFilter)==0:
                                         pass;
                                    if len(cmntJsonFilter)>0:   
                                        for j in range(len(cmntJsonFilter)):
                                            commentID.append(postjson['data'][i]['id'])
                                            commentScore.append(cmntJsonFilter[j]['score'])
                                            comment.append(cmntJsonFilter[j]['body'])
                                except:
                                    pass;
                                    
                        except:
                            pass;
            except:
                pass;                
            start_time=end_time
            start_time_unix=int(start_time.timestamp())
            end_time=start_time+datetime.timedelta(days=1)
            end_time_unix=int(end_time.timestamp())
            pbar.update(1)
                   
        pbar.close()
                
        self.df_corpus['postTitle']=postTitle
        self.df_corpus['postText']=postText
        self.df_corpus['postScore']=postScore
        self.df_corpus['postID']=postID
        self.df_corpus['postURL']=postURL
        


        self.df_comments['commentID']=commentID
        self.df_comments['commentScore']=commentScore
        self.df_comments['comment']=comment
        if self.saveLocal==True:  
            self.df_corpus.to_csv('../data/df_corpus.csv',)
            self.df_comments.to_csv('../data/df_comments.csv')
        return self.df_corpus, self.df_comments  


    def getBooks(self,df_comments=None):
        
        self.df_bookauthor=pd.DataFrame()
        bookIDList=[]
        bookAuthorList=[]
        bookCategoryList=[]
        bookImageList=[]
        bookDescriptionList=[]
        bookPrinttypeList=[]
        
        for i in tqdm(range(len(self.df_comments))):
            stringQuery=self.df_comments['comment'][i]
            stringQuery=re.sub('&|\*','',stringQuery)
            stringQueryLine=re.split(r"\s*\n", stringQuery)
            for j in range(len(stringQueryLine)):
                try:
                    googlebooksAPI="https://www.googleapis.com/books/v1/volumes?q="+str(stringQueryLine[j])
                    booksJson = requests.get(googlebooksAPI)  
                    try:
                        bookData = json.loads(booksJson.text)['items']
                        for k in range(1):
                           
                            title=bookData[k]['volumeInfo']['title']
                            try:
                                subtitle=bookData[k]['volumeInfo']['subtitle']
                                title=title +": " + subtitle
                            except:
                                pass;
                            try:
                                author=bookData[k]['volumeInfo']['authors']
                            except:
                                author='NA'        
                            try:
                                category=bookData[k]['volumeInfo']['categories']
                            except:
                                category='NA'
                            try:
                                image=bookData[k]['volumeInfo']['imageLinks']['thumbnail']
                            except:
                                image='NA' 
                            try:
                                desc=bookData[k]['volumeInfo']['description']
                            except:
                                desc='NA'
                            try:
                                ptype=bookData[k]['volumeInfo']['printType']
                            except:
                                ptype='NA'     
                       
                            bookIDList.append(self.df_comments['commentID'][i])
                            bookAuthorList.append([title,author])  
                            bookCategoryList.append(category) 
                            bookImageList.append(image) 
                            bookDescriptionList.append(desc)
                            bookPrinttypeList.append(ptype)
                        
                          
                    except:
                        pattern_all=['\[(.*?)\]','\"(.*?)\"','\*(.*?)\*']
                        substring=[]
                        
                        for l in range(len(pattern_all)):
                            substring_tmp = re.findall(pattern_all[l], stringQueryLine[j])
                            substring.extend(substring_tmp)
                        try:
                            for m in range(len(substring)):
                                try:
                                    googlebooksAPI="https://www.googleapis.com/books/v1/volumes?q="+str(substring[m])
                                    booksJson = requests.get(googlebooksAPI)   
                                    try:
                                        bookData = json.loads(booksJson.text)['items']
                                    
                                        for k in range(1):
                                            title=bookData[k]['volumeInfo']['title']
                                            try:
                                                subtitle=bookData[k]['volumeInfo']['subtitle']
                                                title=title +": " + subtitle
                                            except:
                                                pass;
                                            try:
                                                author=bookData[k]['volumeInfo']['authors']
                                            except:
                                                author='NA'        
                                            try:
                                                category=bookData[k]['volumeInfo']['categories']
                                            except:
                                                category='NA'
                                            try:
                                                image=bookData[k]['volumeInfo']['imageLinks']['thumbnail']
                                            except:
                                                image='NA'     
                                            try:
                                                desc=bookData[k]['volumeInfo']['description']
                                            except:
                                                desc='NA'
                                            try:
                                                ptype=bookData[k]['volumeInfo']['printType']
                                            except:
                                                ptype='NA'     
                                         
                                            bookIDList.append(self.df_comments['commentID'][i])
                                            bookAuthorList.append([title,author])  
                                            bookCategoryList.append(category) 
                                            bookImageList.append(image) 
                                            bookDescriptionList.append(desc)
                                            bookPrinttypeList.append(ptype)
                            
                                       
                                            
                                    except:
                                        pass;
                                except:
                                    pass;
                                                
                        except:
                            pass;                         
                  
                except:  
                    pass;
                                  
      
        self.df_bookauthor['bookID']=bookIDList
        self.df_bookauthor['bookAuthor']=bookAuthorList
        self.df_bookauthor['bookCategory']=bookCategoryList
        self.df_bookauthor['bookImage']=bookImageList
        self.df_bookauthor['bookDescription']=bookDescriptionList                                    
        self.df_bookauthor['bookPrinttype']=bookPrinttypeList                                    
        self.df_bookauthor['book']= self.df_bookauthor.bookAuthor.apply(lambda x: x[0])
        self.df_bookauthor=self.df_bookauthor.copy()
        self.df_bookauthor.drop_duplicates(subset=['bookID', 'book'], keep='first',inplace=True)
        if self.saveLocal==True:  
            self.df_bookauthor.to_csv('../data/df_bookauthor.csv')
            
        return self.df_bookauthor
             

if __name__ == '__main__':
    
    
    RedditCorpusClass=RedditCorpus(start_time=datetime.datetime(2018, 1, 1),
                                   end_time=datetime.datetime(2018, 1, 10),
                                   deltaDay=1,
                                   subreddit='suggestmeabook',
                                   PostScoreThresh=10,
                                   CommentScoreThresh=3,
                                   saveLocal=True)
    
                                 
    df_corpus,df_comments=RedditCorpusClass.getRedditPost()
    df_bookauthor=RedditCorpusClass.getBooks()                
    
    
