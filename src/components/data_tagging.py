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
from fuzzywuzzy import fuzz



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
                    "SELECT id, student_id, level, question, answer,answer_check_status, no_del, no_sub, no_mistakes, no_mistakes_edited, wcpm FROM parakh_v1_testdetail WHERE section='reading'",
                    con=engine
                )
            else:
                from src.components.database import get_connection
                conn = get_connection()
                try:
                    df = pd.read_sql(
                        "SELECT id, student_id, level, question, answer,answer_check_status, no_del, no_sub, no_mistakes, no_mistakes_edited, wcpm FROM parakh_v1_testdetail WHERE section='reading'",
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
    

    ################## WORD LEVEL TAGGING ##########################################

    @staticmethod
    def _strip_matras(word: str) -> str:
        """Remove Devanagari matras and vowel signs."""
        if not isinstance(word, str):
            return ""
        return re.sub(r'[\u093E-\u094C\u0900-\u0903\u094D]', '', word)

    def _generate_word_tag(self, q: str, a: str) -> str:
        """Generate tag for a word-level response."""
        q, a = str(q).strip(), str(a).strip()

        if not q or not a:
            return "Omission"

        # Missing matras — consonants same after removing vowel signs
        if self._strip_matras(q) == self._strip_matras(a) and q != a:
            return "Phonetic issue - Missing matras"

        # Decoding issue — missing or deleted letters
        if len(a) < len(q) and (a in q or q.startswith(a) or q.endswith(a)):
            return "Decoding issue - Not able to identify all letters"

        # Decoding issue — partial reading
        if len(a) != len(q) and (a[:1] == q[:1] or a[-1:] == q[-1:]):
            return "Decoding issue - Partial reading"

        # Substitution issue — completely different
        ratio = fuzz.ratio(q, a)
        if ratio < 40:
            return "Substitution issue - Unrelated word"

        # Phonetic issue — similar sounding or looking
        if 40 <= ratio < 85:
            return "Phonetic issue - Similar looking/sounding"

        # Correct
        if q == a:
            return "Correct"

        return "Uncategorized"

    def _tag_word_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies tagging logic for word-level responses."""
        logging.info("Applying word-level tagging...")

        words_df = df[df.get('level') == 'Word'].copy()
        if words_df.empty:
            logging.info("No 'Word' level records found.")
            return words_df

        words_df['question'] = words_df.get('question', '').astype(str).str.strip()
        words_df['answer'] = words_df.get('answer', '').astype(str).str.strip()

        words_df['tags'] = words_df.apply(
            lambda row: self._generate_word_tag(row['question'], row['answer']),
            axis=1
        )

        logging.info("Word-level tagging completed.")
        return words_df



    ################# Paragraph Level Tagging ######################################

    def _categorize_paragraph_profile(self, row: pd.Series) -> str:
        """Categorize the student's paragraph-level mistake pattern."""
        mistakes = row.get('no_mistakes_edited', 0)
        subs = row.get('no_sub', 0)
        dels = row.get('no_del', 0)

        if mistakes == 0:
            return 'No Mistake'
        elif mistakes < 3:
            if subs > dels:
                return 'Substitution Minor'
            elif subs < dels:
                return 'Deletion Minor'
            return 'Mixed Minor'
        elif 3 <= mistakes < 6:
            if subs > dels:
                return 'Substitution Moderate'
            elif subs < dels:
                return 'Deletion Moderate'
            return 'Mixed Moderate'
        elif mistakes >= 6:
            if subs > dels:
                return 'Substitution Dominant'
            elif subs < dels:
                return 'Deletion Dominant'
            return 'Mixed Dominant'
        return 'Uncategorized'
    

    def _tag_paragraph_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies tagging logic for paragraph-level responses."""
        logging.info("Applying paragraph-level tagging...")

        para_df = df[df.get('level') == 'Paragraph'].copy()
        if para_df.empty:
            logging.info("No 'Paragraph' level records found.")
            return para_df

        # Compute question length (number of words)
        para_df['question_length'] = para_df['question'].apply(lambda x: len(str(x).split()))

        # Apply classification logic
        para_df['tags'] = para_df.apply(self._categorize_paragraph_profile, axis=1)

        logging.info("Paragraph-level tagging completed.")
        return para_df
    

    ################# STORY TAGGING #############################################

    def _tag_story_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies tagging logic for story-level responses."""
        logging.info("Applying story-level tagging...")

        story_df = df[df.get('level') == 'Story'].copy()
        if story_df.empty:
            logging.info("No 'Story' level records found.")
            return story_df

        # Compute question length (number of words)
        story_df['question_length'] = story_df['question'].apply(lambda x: len(str(x).split()))

        # Apply classification logic
        story_df['tags'] = story_df.apply(self._categorize_paragraph_profile, axis=1)

        logging.info("Story-level tagging completed.")
        return story_df
    





    ################# Main Entry Point #############################################
    
    def initiate_data_tagging(self):
        """Main function to orchestrate the tagging process."""
        logging.info("Starting data tagging process...")

        try:
            df = self._fetch_test_details()
            logging.debug(df.head(3))

            letter_tagged = self._tag_letter_level(df)
            word_tagged = self._tag_word_level(df)
            paragraph_tagged = self._tag_paragraph_level(df)
            story_tagged = self._tag_story_level(df)

            # Save output
            if not letter_tagged.empty:
                os.makedirs(os.path.dirname(self.config.letterPath), exist_ok=True)
                letter_tagged.to_csv(self.config.letterPath, index=False)
                logging.info(f"Letter-level tagged data saved to {self.config.letterPath}")


            if not word_tagged.empty:
                os.makedirs(os.path.dirname(self.config.wordPath), exist_ok=True)
                word_tagged.to_csv(self.config.wordPath, index=False)
                logging.info(f"Word-level data saved to {self.config.wordPath}")


            if not paragraph_tagged.empty:
                os.makedirs(os.path.dirname(self.config.paragraphPath), exist_ok=True)
                paragraph_tagged.to_csv(self.config.paragraphPath, index=False)
                logging.info(f"Paragraph-level data saved to {self.config.paragraphPath}")


            if not story_tagged.empty:
                os.makedirs(os.path.dirname(self.config.storyPath), exist_ok=True)
                story_tagged.to_csv(self.config.storyPath, index=False)
                logging.info(f"Story-level data saved to {self.config.storyPath}")

            logging.info("Data tagging pipeline completed successfully.")

        except Exception as e:
            logging.exception("Data tagging pipeline failed.")
            raise CustomException(e, sys)
