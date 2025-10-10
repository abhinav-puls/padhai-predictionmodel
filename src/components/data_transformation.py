# aim: reading the data from database/ datasource & 
# Trasnforming it in the way that our model needs it

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

# from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    '''
    Used to create path variables
    Path: Any input that is required in data Transformation component
    1. Saving student data - path
    2. Saving baseline data- path
    3. Saving community summary data- path
    '''
    # train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    # raw_data_path: str=os.path.join('artifact','data.csv')
    student_df_path: str=os.path.join('artifact','student_df.csv')
    community_summary_path: str=os.path.join('artifact','community_summary.csv')
    baseline_dataset_path: str=os.path.join('artifact','baseline_dataset.csv')
    skills_dist_path: str=os.path.join('artifact','skills_dist_df.csv')
    bl_test_path:  str=os.path.join('artifact','bl_test.csv')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig() #initialize the input

    def initiate_data_transformation(self):
        logging.info('Entered the Data Transformation component.')
        try:
            parakh_v1_student = pd.read_csv('assets/parakh_v1_student.csv')
            logging.info('Read the Dataset as parakh_v1_student')

            test = pd.read_csv('assets/parakh_v1_test.csv')
            logging.info('Read the Dataset as parakh_v1_test')

            # os.makedirs(os.path.dirname(self.transformation_config.train_data_path), exist_ok = True)
            print(parakh_v1_student.head(3))
            student_df = parakh_v1_student[['community_id','id','grade']]
            student_df.to_csv(self.transformation_config.student_df_path, index = False, header = True)
            logging.info('Student_df is now being saved')

            class_size = student_df.groupby('community_id')['id'].nunique().rename('class_size')

            # Step 2: Calculate grade distribution (ratios per community)
            grade_counts = (
                student_df.groupby(['community_id', 'grade'])['id']
                .nunique()
                .unstack(fill_value=0)
            )

            # Ensure only grades 3, 4, 5 are considered (others will be ignored)
            for g in [3, 4, 5]:
                if g not in grade_counts.columns:
                    grade_counts[g] = 0

            # Step 3: Convert counts to ratios
            grade_ratios = grade_counts.div(grade_counts.sum(axis=1), axis=0)

            # Step 4: Rename columns
            grade_ratios = grade_ratios.rename(columns={
                3: 'class_3_ratio',
                4: 'class_4_ratio',
                5: 'class_5_ratio'
            })[['class_3_ratio', 'class_4_ratio', 'class_5_ratio']]

            # Step 5: Combine everything
            community_summary = (
                pd.concat([class_size, grade_ratios], axis=1).reset_index())

            # Step 6: Fill missing ratios with 0
            community_summary[['class_3_ratio', 'class_4_ratio', 'class_5_ratio']] = (
                community_summary[['class_3_ratio', 'class_4_ratio', 'class_5_ratio']].fillna(0))

            community_summary.to_csv(self.transformation_config.community_summary_path, index = False, header = True)


            ### Lets save the test dataframe
            test['subject'] = test['subject'].str.strip()
            test['test_type'] = test['test_type'].str.strip()

            # Step 2: Create separate Reading and Maths datasets
            reading_df = test[test['subject'].str.lower() == 'reading']
            maths_df = test[test['subject'].str.lower().isin(['maths', 'math', 'mathematics'])]

            # Step 3: Define test types of interest
            test_types = [
                'Baseline',
                'Endline 1',
                'Endline 2',
                'Baseline 2',
                'Endline 3',
                'Endline 4'
            ]

            # Step 4: Create separate DataFrames for each combination
            datasets = {}

            for ttype in test_types:
                # filter for reading
                datasets[f'reading_{ttype.lower().replace(" ", "")}'] = reading_df[
                    reading_df['test_type'].str.lower() == ttype.lower()
                ].copy()
                
                # filter for maths
                datasets[f'maths_{ttype.lower().replace(" ", "")}'] = maths_df[
                    maths_df['test_type'].str.lower() == ttype.lower()
                ].copy()

            rename_map = {
                ('reading', 'baseline'): 'bl_language',
                ('maths', 'baseline'): 'bl_mathematics',
                ('reading', 'baseline 2'): 'bl2_language',
                ('maths', 'baseline 2'): 'bl2_mathematics',
                ('reading', 'endline 1'): 'el1_language',
                ('maths', 'endline 1'): 'el1_mathematics',
                ('reading', 'endline 2'): 'el2_language',
                ('maths', 'endline 2'): 'el2_mathematics',
                ('reading', 'endline 3'): 'el3_language',
                ('maths', 'endline 3'): 'el3_mathematics',
                ('reading', 'endline 4'): 'el4_language',
                ('maths', 'endline 4'): 'el4_mathematics'
            }
                
            for name, df in datasets.items():
                # Extract subject and test_type from the dataset name
                parts = name.split('_')
                subject = parts[0]        # 'reading' or 'maths'
                testtype = parts[1]
                # Fix names like 'endline1' → 'endline 1'
                if testtype.startswith('endline'):
                    testtype = 'endline ' + testtype.replace('endline', '').strip()
                elif testtype.startswith('baseline2'):
                    testtype = 'baseline 2'
                elif testtype.startswith('baseline'):
                    testtype = 'baseline'

                # Get new column name
                new_col_name = rename_map.get((subject, testtype.lower()))
                if new_col_name and 'manual_proficiency' in df.columns:
                    df.rename(columns={'manual_proficiency': new_col_name}, inplace=True)
                    datasets[name] = df  # update in dict  

            lang_baseline = datasets['reading_baseline']
            maths_baseline = datasets['maths_baseline']

            lang_baseline = lang_baseline[['community_id','student_id','bl_language']]
            maths_baseline = maths_baseline[['community_id','student_id','bl_mathematics']]

            baselinedataset = pd.merge(lang_baseline, maths_baseline, on=['community_id','student_id'])
            baselinedataset.to_csv(self.transformation_config.baseline_dataset_path, index = False, header = True)

            logging.info('baseline dataset created')

            # train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

            # train_set.to_csv(self.transformation_config.train_data_path, index = False, header = True)
            # test_set.to_csv(self.transformation_config.test_data_path, index = False, header = True)

            lang_dist = (
                lang_baseline.groupby(['community_id', 'bl_language'])['student_id']
                .nunique()
                .unstack(fill_value=0)
            )

            lang_dist = lang_dist.div(lang_dist.sum(axis=1), axis=0).reset_index()

            # Rename columns
            lang_dist = lang_dist.rename(columns={
                'Beginner': 'Lang_Beginner',
                'Letter': 'Lang_Letter',
                'Word': 'Lang_Word',
                'Paragraph': 'Lang_Paragraph',
                'Story': 'Lang_Story'
            })[['community_id','Lang_Beginner', 'Lang_Letter', 'Lang_Word', 'Lang_Paragraph', 'Lang_Story']]

            maths_dist = (
                maths_baseline.groupby(['community_id', 'bl_mathematics'])['student_id']
                .nunique()
                .unstack(fill_value=0)
            )

            # Convert to ratios
            maths_dist = maths_dist.div(maths_dist.sum(axis=1), axis=0).reset_index()

            maths_dist = maths_dist.rename(columns={
                'Beginner': 'Maths_Beginner',
                '0-9': 'Maths_NR1',
                '10-99': 'Maths_NR2',
                'Subtraction': 'Maths_Sub',
                'Division': 'Maths_Div'
            })[['community_id','Maths_Beginner', 'Maths_NR1', 'Maths_NR2', 'Maths_Sub', 'Maths_Div']]
            
            
            skills_dist_df = pd.merge(lang_dist, maths_dist, on='community_id')
            skills_dist_df.to_csv(self.transformation_config.skills_dist_path, index = False, header = True)

            logging.info('skills_dist_df is created')

            community_skills_grade = pd.merge(community_summary, skills_dist_df, on ='community_id')
            print(community_skills_grade.head(2))
            
            student_df1 = student_df[['id','grade']]
            bl_t_student = pd.merge(student_df1, baselinedataset, left_on = 'id',right_on = 'student_id', how='left')
            bl_test = pd.merge(bl_t_student,community_skills_grade, on='community_id', how='left' )
            bl_test.to_csv(self.transformation_config.bl_test_path, index = False, header = True)

            logging.info('bl_test is created')


            # Create three versions of bl_test
            phase1 = bl_test.copy()
            phase1['Phase'] = 1

            phase2 = bl_test.copy()
            phase2['Phase'] = 2

            phase3 = bl_test.copy()
            phase3['Phase'] = 3

            # Concatenate them
            final_df = pd.concat([phase1, phase2, phase3], ignore_index=True)
            final_df.to_csv(self.transformation_config.test_data_path, index = False, header = True)

            logging.info('Transformation of the data is completed')

            # return the path for the data transformation step
            return (
                self.transformation_config.student_df_path, 
                self.transformation_config.community_summary_path,
                self.transformation_config.baseline_dataset_path,
                self.transformation_config.skills_dist_path,
                self.transformation_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    student_df, community_summary, baseline_dataset, skills_dist, test_data = obj.initiate_data_transformation()
    print('code ran successfully!')
    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_data, test_data)
    # print(train_arr[0])
    
    # model_trainer = ModelTrainer()
    # print(model_trainer.initiate_model_trainer(train_arr, test_arr)) # gives the r2_score