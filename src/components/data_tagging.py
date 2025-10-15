##########################################################################
#Aim:
#Fetch data from the database and tag the mistakes that individual students make during assessments.
##########################################################################

import os
import sys
import re
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.database import engine


@dataclass
class DataTaggingConfig:
    """
    Stores file paths for saving tagged data at different reading levels.
    """
    letterPath: str = os.path.join('artifact', 'tags', 'letter.csv')
    wordPath: str = os.path.join('artifact', 'tags', 'word.csv')
    paragraphPath: str = os.path.join('artifact', 'tags', 'paragraph.csv')
    storyPath: str = os.path.join('artifact', 'tags', 'story.csv')


class DataTagging:
    """Handles data fetching and tagging for student reading assessment data."""

    def __init__(self):
        self.config = DataTaggingConfig()

        # Similarity and phonetic mappings
        self.SIMILAR_LOOKING_PAIRS = [
            ('व', 'ब'), ('ध', 'घ'), ('घ', 'ध'), ('द', 'ट'),
            ('ध', 'छ'), ('च', 'ज'), ('म', 'भ')
        ]

        self.PHONETIC_PAIRS = [
            ('ठ', 'ट'), ('घ', 'ग'), ('ध', 'द'), ('ग', 'घ'),
            ('ख', 'क'), ('ड', 'द'), ('ट', 'ठ'), ('ज', 'झ'),
            ('ड', 'ढ'), ('द', 'ध'), ('क', 'ख'), ('च', 'छ'),
            ('द', 'क'), ('ब', 'भ')
        ]

        self.FLAG_MAP = {
            '1': "No mistake",
            '2': "Decoding issue",
            '3a': "Visual Mismatch",
            '3b': "Phonetic issue",
            '3c': "Decoding Issue",
            '4': "Decoding Issue",
            '9': "Accent"
        }

    
    ##### Database Functions################################


    def _fetch_test_details(self) -> pd.DataFrame:
        """
        Fetches the reading section test details from the database.
        Falls back to psycopg2 if SQLAlchemy engine is unavailable.
        """
        logging.info("Fetching test details from the database...")

        try:
            if engine is not None:
                df = pd.read_sql(
                    "SELECT id, student_id, level, question, answer,answer_check_status, no_del, no_sub, no_mistakes, wcpm FROM parakh_v1_testdetail WHERE section='reading'",
                    con=engine
                )
            else:
                from src.components.database import get_connection
                conn = get_connection()
                try:
                    df = pd.read_sql(
                        "SELECT * FROM parakh_v1_testdetail WHERE section='reading'",
                        con=conn
                    )
                finally:
                    conn.close()

            logging.info(f"Fetched {len(df)} records from the database.")
            return df

        except Exception as e:
            logging.exception("Failed to read test_details from database.")
            raise CustomException(e, sys)

    ####################### Utility Functions ####################################

    @staticmethod
    def _build_lookup_set(pairs):
        """Creates a bidirectional lookup set for quick comparison."""
        return {(a, b) for a, b in pairs} | {(b, a) for a, b in pairs}

    @staticmethod
    def _create_matra_pattern():
        """Compiles a regex for identifying Devanagari matras."""
        return re.compile(r"[\u093E-\u094C\u0902\u0903]")

    def _compare_texts(self, canonical, transcribed, similar_set, phonetic_set, matra_pattern):
        """
        Compare canonical and transcribed Devanagari letters.
        Returns a flag code (string) based on mismatch type.
        """
        try:
            if canonical is None or transcribed is None:
                return 0

            canonical, transcribed = str(canonical).strip(), str(transcribed).strip()

            if canonical == transcribed:
                return '1'  # Exact match

            if len(transcribed) >= 2 and canonical and canonical[0] == transcribed[0]:
                if len(transcribed) == 2 and matra_pattern.match(transcribed[1]):
                    return '9'  # Accent difference
                elif len(transcribed) == 3 and all(matra_pattern.match(ch) for ch in transcribed[1:]):
                    return '9'
                return '2'  # Partial decoding issue

            if len(transcribed) == 1:
                c, t = canonical[0], transcribed[0]
                if (c, t) in similar_set:
                    return '3a'
                elif (c, t) in phonetic_set:
                    return '3b'
                return '3c'

            return '4'  # Multi-character substitution

        except Exception as e:
            logging.exception("Error comparing texts: %s", e)
            return 0

    ########### Tagging Logic #######################################

    def _tag_letter_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters letter-level rows and applies tagging logic.
        """
        logging.info("Applying letter-level tagging...")

        similar_set = self._build_lookup_set(self.SIMILAR_LOOKING_PAIRS)
        phonetic_set = self._build_lookup_set(self.PHONETIC_PAIRS)
        matra_pattern = self._create_matra_pattern()

        letters_df = df[df.get('level') == 'Letter'].copy()
        if letters_df.empty:
            logging.info("No 'Letter' level records found.")
            return letters_df

        letters_df['question'] = letters_df.get('question', '').astype(str).str.strip()
        letters_df['answer'] = letters_df.get('answer', '').astype(str).str.strip()

        letters_df['letFlag'] = letters_df.apply(
            lambda row: self._compare_texts(
                row.get('question'),
                row.get('answer'),
                similar_set,
                phonetic_set,
                matra_pattern
            ),
            axis=1
        )

        letters_df['tags'] = letters_df['letFlag'].astype(str).map(self.FLAG_MAP).fillna('Unknown')

        logging.info("Letter-level tagging completed.")
        return letters_df

    ################# Main Entry Point ####################################
    
    def initiate_data_tagging(self):
        """Main function to orchestrate the tagging process."""
        logging.info("Starting data tagging process...")

        try:
            df = self._fetch_test_details()
            logging.debug(df.head(3))

            letter_tagged = self._tag_letter_level(df)

            # Save output
            if not letter_tagged.empty:
                os.makedirs(os.path.dirname(self.config.letterPath), exist_ok=True)
                letter_tagged.to_csv(self.config.letterPath, index=False)
                logging.info(f"Letter-level tagged data saved to {self.config.letterPath}")

            logging.info("Data tagging pipeline completed successfully.")

        except Exception as e:
            logging.exception("Data tagging pipeline failed.")
            raise CustomException(e, sys)
