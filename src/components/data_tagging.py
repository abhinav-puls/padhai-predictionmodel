### Aim of this py file is to fetch the data from the database and
### Tag for the mistakes that individual students are making during the assessment


import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.components.database import engine

from dataclasses import dataclass

@dataclass
class DataTaggingConfig:
    '''
    used to create the path variables
    Path: any input that is required in the data tagging component
    1. Saving the data by each reading level is the path 
    
    '''

    letterPath: str=os.path.join('artifact','tags','letter.csv')
    wordPath: str=os.path.join('artifact','tags','word.csv')
    paragraphPath: str=os.path.join('artifact','tags','paragraph.csv')
    storyPath: str=os.path.join('artifact','tags','story.csv')


class DataTagging:
    def __init__(self):
        self.tagging_config = DataTaggingConfig() #Initialize the input

    def initiate_data_tagging(self):
        logging.info('Entered the data Tagging Component')
        try:
            try:
                ## Read using the SQLAlchemy engine  when availabel or else use raw psycopg2 connection
                if engine is not None:
                    parakh_v1_testdetails = pd.read_sql("SELECT * FROM parakh_v1_testdetail WHERE section='reading'", con=engine)

                else:
                    ## Fallback to the psycopg2 connection helper
                    from src.components.database import get_connection
                    conn = get_connection()

                    try:
                        parakh_v1_testdetails = pd.read_sql('SELECT * FROM parakh_v1_testdetail', con=conn)
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                logging.info("Read the dataset parakh_v1_test details from the db in tagging pipeline")
            except Exception as e:
                logging.exception("Failed to read the test_details table from the db in the tagging pipeline")
                raise CustomException(e, sys)
            
        
            print(parakh_v1_testdetails.head(3))
        
        except Exception as e:
            logging.exception("Failed to execute the tagging pipeline")
            raise CustomException(e, sys)
        







    
