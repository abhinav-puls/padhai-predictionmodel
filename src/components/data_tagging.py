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
    combinedPath: str = os.path.join('artifact', 'tags', 'combinedTagged.csv')
    finalProfiles: str = os.path.join('artifact', 'output', 'profiles.csv')


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
            '2': "Decoding Issue",
            '3a': "Visual Mismatch",
            '3b': "Phonetic Issue",
            '3c': "Decoding Issue",
            '4': "Decoding Issue",
            '9': "Phonetic Issue"
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
                df = pd.read_sql("""
                                SELECT td.id, td.student_id, td.level, 
                                td.question, td.answer,td.answer_check_status, 
                                td.no_del, td.no_sub, td.no_mistakes, 
                                td.no_mistakes_edited, td.wcpm,
                                t.manual_proficiency, t.test_type, t.community_id
                                FROM parakh_v1_testdetail td LEFT JOIN parakh_v1_test t 
                                ON td.test_id = t.id
                                WHERE td.section='reading'""",
                    con=engine
                )
            else:
                from src.components.database import get_connection
                conn = get_connection()
                try:
                    df = pd.read_sql(
                        """SELECT td.id, td.student_id, td.level, 
                                    td.question, td.answer,td.answer_check_status, 
                                    td.no_del, td.no_sub, td.no_mistakes, 
                                    td.no_mistakes_edited, td.wcpm,
                                    t.manual_proficiency, t.test_type, t.community_id
                                 FROM parakh_v1_testdetail td LEFT JOIN parakh_v1_test t 
                                 ON td.test_id = t.id
                                 WHERE td.section='reading'""",
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

            # --- New: Combine them into one final DataFrame ---
            common_cols = [
                "id", "community_id", "student_id", "test_type","level","manual_proficiency",
                "question", "answer", "answer_check_status",
                "no_del", "no_sub", "no_mistakes", "no_mistakes_edited", "wcpm","tags"
            ]

            # Ensure all DataFrames have these columns
            def align_columns(df):
                for col in common_cols:
                    if col not in df.columns:
                        df[col] = None
                return df[common_cols]
            

            letter_df = align_columns(letter_tagged)
            word_df = align_columns(word_tagged)
            paragraph_df = align_columns(paragraph_tagged)
            story_df = align_columns(story_tagged)

            combined_df = pd.concat([letter_df, word_df, paragraph_df, story_df], ignore_index=True)

            if not combined_df.empty:
                os.makedirs(os.path.dirname(self.config.combinedPath), exist_ok=True)
                combined_df.to_csv(self.config.combinedPath, index=False)
                logging.info(f"Combined Tagged data saved successfully {self.config.combinedPath}")

            return combined_df

        except Exception as e:
            logging.exception("Data tagging pipeline failed.")
            raise CustomException(e, sys)
        
    def data_profiling(self, df_combined: "pd.DataFrame"):
        """
        Accepts df_combined and returns a dict with profiling results.
        Implement profiling logic inside the try block.
        """
        try:
            # basic validation
            if df_combined is None:
                raise ValueError("df_combined is None")
            if not isinstance(df_combined, pd.DataFrame):
                df_combined = pd.DataFrame(df_combined)

            print(f"The combined dataframe  {df_combined.head(2)}")

            BL = df_combined[(df_combined['manual_proficiency']=='Beginner') & (df_combined['level']=='Letter')]
            LL = df_combined[(df_combined['manual_proficiency']=='Letter') & (df_combined['level']=='Word')]
            WL = df_combined[(df_combined['manual_proficiency']=='Word') & (df_combined['level']=='Paragraph')]
            PL = df_combined[(df_combined['manual_proficiency']=='Paragraph') & (df_combined['level']=='Story')]
            SL = df_combined[(df_combined['manual_proficiency']=='Story') & (df_combined['level']=='Story')]



            ############### BEGINNER STUDENT PROFILES ########################################
            BL = BL[(BL['answer_check_status']=='False') | (BL['answer_check_status']=='')]
            
            BL_pivot = BL.pivot_table(index=['test_type','student_id'], columns='tags', 
                           values='id', aggfunc='count', fill_value=0).reset_index()
            
            required_cols = ["Decoding Issue", "Phonetic issue", "Visual Mismatch"]

            for col in required_cols:
                if col not in BL_pivot.columns:
                    BL_pivot[col] = None

            def classify_student(row):
                # Safely get numeric values (default to 0 if missing or NaN)
                decoding = row.get("Decoding Issue", 0) or 0
                phonetic = row.get("Phonetic issue", 0) or 0
                visual = row.get("Visual Mismatch", 0) or 0

                # Total mistakes
                total = decoding + phonetic + visual

                # Determine severity
                severity = "Major" if total >= 3 else "Minor"

                # Determine dominant error type
                types = {
                    "Decoding": decoding,
                    "Phonetic": phonetic,
                    "Visual": visual
                }

                max_type = max(types, key=types.get)
                max_val = types[max_type]

                # If multiple categories share the same max value → Mixed
                if list(types.values()).count(max_val) > 1:
                    return f"{severity} Mixed"
                else:
                    return f"{severity} {max_type}"

            BL_pivot['Fluency Band'] = None

            BL_pivot['Profile'] = None
            BL_pivot['Profile'] = BL_pivot.apply(classify_student, axis = 1)
            print(f"Begginer students profiles are created{BL_pivot.head(2)}")



            ############### LETTER STUDENT PROFILES ########################################
            ## Static Function
            def strip_matras(word):
                """Remove Devanagari matras, vowel signs, and make safe for any input."""
                try:
                    return re.sub(r'[\u093E-\u094C\u0900-\u0903\u094D]', '', str(word)).strip()
                except Exception:
                    return ""
            
            def classify_student_word(row):
                """
                Classify student based on total mistakes and dominant type.
                
                Args:
                    row (pd.Series): A row containing 'Total Mistakes', 'Decoding issue', and 'Phonetic issue'.
                
                Returns:
                    str: Classification label like 'Major Decoding', 'Minor Phonetic', or 'Major Mixed'.
                """
                total = row.get("Total Mistakes", 0)
                decoding = row.get("Decoding issue", 0)
                phonetic = row.get("Phonetic issue", 0)
                
                # Determine severity
                severity = "Major" if total >= 3 else "Minor"
                
                # Determine dominant type
                if decoding > phonetic:
                    profile_type = "Decoding"
                elif phonetic > decoding:
                    profile_type = "Phonetic"
                else:
                    profile_type = "Mixed"
                
                # Combine into profile label
                return f"{severity} {profile_type}"

            
            LL = LL[(LL['answer_check_status']=='False') | (LL['answer_check_status']=='')]

            LL_pivot = LL.pivot_table(index=['test_type','student_id'], columns='tags', 
                           values='id', aggfunc='count', fill_value=0).reset_index()
            
            required_cols_word = ["Omission","Phonetic issue - Missing matras",
                             "Decoding issue - Not able to identify all letters",
                            "Decoding issue - Partial reading",
                            "Substitution issue - Unrelated word",
                            "Phonetic issue - Similar looking/sounding",
                            "Correct",
                            "Uncategorized"]
             
            for col in required_cols_word:
                if col not in LL_pivot.columns:
                    LL_pivot[col] = None

            ## Aggregate the decoding issues
            decoding_cols = ["Decoding issue - Not able to identify all letters",
                             "Decoding issue - Partial reading",
                             "Omission",
                             "Substitution issue - Unrelated word"]
            
            LL_pivot["Decoding issue"] = LL_pivot.get(decoding_cols[0], 0)
            for col in decoding_cols[1:]:
                LL_pivot["Decoding issue"] += LL_pivot.get(col, 0)

            # Aggregate Phonetic issue
            phonetic_cols = ["Phonetic issue - Missing matras","Phonetic issue - Similar looking/sounding"]

            LL_pivot["Phonetic issue"] = LL_pivot.get(phonetic_cols[0], 0)
            for col in phonetic_cols[1:]:
                LL_pivot["Phonetic issue"] += LL_pivot.get(col, 0)

            LL_pivot["Total Mistakes"] = LL_pivot["Decoding issue"] + LL_pivot["Phonetic issue"]
            LL_pivot['Profile'] = None
            LL_pivot["Profile"] = LL_pivot.apply(classify_student_word, axis=1)
            LL_pivot['Fluency Band'] = None

            print(f"Letter students profiles are created{LL_pivot.head(2)}")


            ############### WORD STUDENT PROFILES ########################################
            WL['Profile'] = WL['tags']
            
            def wcpm_band(wcpm):
                if wcpm< 60:
                    return "Low wcpm"
                elif 60<= wcpm <=100:
                    return 'Mid wcpm'
                else:
                    return 'High wcpm'
            
            WL['Fluency Band'] = WL['wcpm'].apply(wcpm_band)
            print(f"Word students profiles are created{WL.head(2)}")

            ############### PARA STUDENT PROFILES ########################################

            PL['Profile'] = PL['tags'] 
            PL['Fluency Band'] = PL['wcpm'].apply(wcpm_band)
            print(f"Paragraph students profiles are created{PL.head(2)}")


            ############### STORY STUDENT PROFILES ########################################
            SL['Profile'] = SL['tags'] 
            SL['Fluency Band'] = SL['wcpm'].apply(wcpm_band)

            print(f"Story students profiles are created{SL.head(2)}")


            ############### COMBINE ALL PROFILES ########################################
            def select_final_cols(df, level_name):
                df = df.copy()
                cols_needed = ['student_id', 'test_type', 'Profile', 'Fluency Band']
                for col in cols_needed:
                    if col not in df.columns:
                        df[col] = None
                df['manual_proficiency'] = df.get('manual_proficiency', level_name)
                return df[cols_needed]

            BL_final = select_final_cols(BL_pivot, 'Beginner')
            LL_final = select_final_cols(LL_pivot, 'Letter')
            WL_final = select_final_cols(WL, 'Word')
            PL_final = select_final_cols(PL, 'Paragraph')
            SL_final = select_final_cols(SL, 'Story')

            final_profiles_df = pd.concat([BL_final, LL_final, WL_final, PL_final, SL_final], ignore_index=True)
            print(f"Combined profiling dataframe shape: {final_profiles_df.shape}")
            print(final_profiles_df.head(5))
            # final_profiles_df.to_csv('profiles.csv')
            final_profiles_df.to_csv(self.config.finalProfiles , index = False, header = True)


            results = {
                    "total_rows": len(df_combined),
                    "summary": None,   
                    "by_level": None,
                    "by_group": None,
                    "examples": None}

            return final_profiles_df


        except Exception as e:
            # keep exception raising consistent with project error handling
            logging.exception("dataprofiling failed: %s", e)
            raise CustomException(e, sys)
