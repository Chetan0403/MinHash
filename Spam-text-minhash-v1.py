import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import warnings
warnings.filterwarnings('ignore')
from contextlib import contextmanager
import mysql.connector
from sqlalchemy import create_engine
import pygsheets
from tqdm import tqdm
import yaml
import os
import binascii

def read_config_yml():
    file = open(os.getcwd() + '/Spam_detection_algo/app/config/config.yml' , "rb")
    config = yaml.load(file)
    return config

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print('%s done in %.0f s'% (name,time.time()-t0))

class TextSimilarity(object):
    
    def __init__(self, id_name="survey_response_id",text_column="answer",shinglesize=5):
        self.id = id_name
        self.text_column = text_column
        self.shinglesize = shinglesize
    
    def _clean_text(self, text):
        remove_punct_dict = dict((ord(punct), " ") for punct in string.punctuation)
        # To lower and remove punctuation
        text = text.lower().translate(remove_punct_dict)      
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def create_shingles(self, data):
        # Create a dictionary of the texts, mapping the text identifier to actual texts
        id_list = list(data[self.id])
        text_list = list(data[self.text_column])
        docsAsShingleSets = {}
               
        totalShingles = 0
        for i in tqdm(range(0, len(text_list))):
            
            # Maintain a list of document IDs
            docID = id_list[i]

            # Convert text paragraphs into a list of words
            words = self._clean_text(text_list[i])
            words = words.split(" ") 
                    
            # 'shinglesInDoc' will hold all of the unique shingle IDs present in the current document
            shinglesInDoc = set()
              
            # For each word in the document...
            for index in range(0, len(words) - (self.shinglesize - 1)):
            
                # Construct the shingle text by combining words together, depending on the shingle size passed
                shingle = ""
                for  word in words[index:(index + (self.shinglesize - 1))]:
                    shingle = shingle + word + " "
                    
                # Hash the shingle to a 32-bit integer.
                shingle = bytes(shingle.strip(), encoding='utf-8')
                crc = binascii.crc32(shingle) & 0xffffffff
            
                # Add the hash value to the list of shingles for the current document. 
                shinglesInDoc.add(crc)
          
            # Store the completed list of shingles for this document in the dictionary.
            docsAsShingleSets[docID] = shinglesInDoc
         
            # Count the number of shingles across all documents.
            totalShingles = totalShingles + (len(words) - (self.shinglesize - 1))

        return docsAsShingleSets
        

if __name__ == "__main__":
        
    with timer("Fetching data from feedback prod db"):
        
        # Establishing connection with feedback prod db
        configuration = read_config_yml()
        conn = mysql.connector.connect(user = configuration['mysql']['user'],
                               password = configuration['mysql']['password'],
                               host = configuration['mysql']['host'],
                               database = configuration['mysql']['database'], 
                               port = configuration['mysql']['port'])
        
        mycursor = conn.cursor()
        print("Connection to feedback db established")
        
        # Columns with which dataframe needs to be saved
        columns = ['doctor_id', 'survey_response_id', 'recommendation', 'created_at', 
                   'rm_deleted_at', 'sr_deleted_at', 'sra_deleted_at', 's_deleted_at',
                   'is_spam','user_verified','is_contested','mobile', 'channel', 
                   'answer', 'status', 'owning_service', 'anonymous']
        
        # Fetch query
        mycursor.execute("""select doctor_id, 
                            rm.survey_response_id as survey_response_id,
                            rm.recommendation, 
                            rm.created_at, 
                            rm.deleted_at as rm_deleted_at,
                            sr.deleted_at as sr_deleted_at,
                            sra.deleted_at as sra_deleted_at,
                            s.deleted_at as s_deleted_at,
                            s.is_spam,
                            s.user_verified,
                            s.is_contested,
                            r.mobile, 
                            channel, 
                            answer, 
                            rm.status, 
                            c.owning_service, 
                            s.anonymous                          
                            FROM 
                            feedback.survey_responses AS sr
                            JOIN feedback.survey_response_answers AS sra ON sr.id=sra.survey_response_id
                            JOIN feedback.review_moderations AS rm ON rm.survey_response_id=sra.survey_response_id
                            JOIN feedback.surveys s ON sr.survey_id = s.id
                            JOIN feedback.campaigns c ON s.campaign_id = c.id
                            JOIN feedback.respondees as r on s.respondee_id=r.id
                            WHERE rm.review_for = 'DOCTOR'""")
        
        # Store fetched data into dataframe
        df_rev_text = mycursor.fetchall()
        df_rev_text = pd.DataFrame(df_rev_text, columns = columns)
       
        # Closing feedback prod db connection
        conn.close()    
        mycursor.close()
 
    with timer("Computing review similarity scores"):
        """ 
        1. Computation of review text similarity below.
        2. valid incremental reviews for each doctor are compared with each other and with past reviews on a range of 1-10 ngrams for similarity
        3. Cosine similarity computed
        """      
        # Shortlisting reviews for similarity checking
        rev = df_rev_text[(df_rev_text['rm_deleted_at'].isnull()) &
                          (df_rev_text['sra_deleted_at'].isnull()) &
                          (df_rev_text['sr_deleted_at'].isnull()) &
                          (df_rev_text['s_deleted_at'].isnull()) &
                          (df_rev_text['is_spam'] == 0) &
                          (df_rev_text['user_verified'] == 1) &
                          (df_rev_text['is_contested'] == 0) &
                          (df_rev_text['status'] == 'PUBLISHED')]
        
        # Create similarity score
        print("Starting similarity computations")

        sim = TextSimilarity(id_name = "survey_response_id",
                             text_column = "answer",
                             shinglesize=5)
        
        shingles_text = sim.create_shingles(df_rev_text[~df_rev_text['answer'].isnull()])
       
        


