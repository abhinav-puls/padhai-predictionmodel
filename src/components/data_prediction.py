import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data_transformed):
        logging.info("PredictPipeline.predict started")
        try:

            # basic input validation / coercion
            if data_transformed is None:
                logging.error("Prediction input is None")
                raise CustomException("Prediction input is None", sys)

            # Accept common types and coerce to DataFrame safely
            if isinstance(data_transformed, pd.DataFrame):
                X = data_transformed.copy()
                logging.debug("Input is DataFrame with shape %s", X.shape)
            elif isinstance(data_transformed, pd.Series):
                X = data_transformed.to_frame().T.copy()
                logging.debug("Input is Series converted to DataFrame with shape %s", X.shape)
            elif isinstance(data_transformed, (np.ndarray, list, dict)):
                try:
                    X = pd.DataFrame(data_transformed)
                    logging.debug("Input coerced to DataFrame with shape %s", X.shape)
                except Exception as e:
                    logging.exception("Unable to convert input to DataFrame")
                    raise CustomException(f"Unable to convert input to DataFrame: {e}", sys)
            else:
                logging.error("Unsupported input type for prediction: %s", type(data_transformed))
                raise CustomException(f"Unsupported input type for prediction: {type(data_transformed)}", sys)

            lang_model_path = 'artifact/models/xgb_lang.pkl'
            maths_model_path = 'artifact/models/xgb_math.pkl'
            logging.info("Using model paths lang=%s maths=%s", lang_model_path, maths_model_path)

            # load models
            try:
                lang_model = load_object(file_path=lang_model_path)
                logging.info("Loaded language model from %s", lang_model_path)
            except Exception:
                logging.exception("Failed to load language model")
                raise

            try:
                maths_model = load_object(file_path=maths_model_path)
                logging.info("Loaded maths model from %s", maths_model_path)
            except Exception:
                logging.exception("Failed to load maths model")
                raise

            # feature subsets as specified
            lang_features = [
                'bl_language','grade','Phase','Lang_Beginner','Lang_Letter',
                'Lang_Word','Lang_Paragraph','Lang_Story','class_size',
                'class_3_ratio','class_4_ratio','class_5_ratio'
            ]
            maths_features = [
                'bl_mathematics','grade','Phase','Maths_Beginner','Maths_NR1',
                'Maths_NR2','Maths_Sub','Maths_Div','class_size',
                'class_3_ratio','class_4_ratio','class_5_ratio'
            ]

            missing_lang = [c for c in lang_features if c not in X.columns]
            missing_maths = [c for c in maths_features if c not in X.columns]
            if missing_lang or missing_maths:
                logging.error("Missing required features. lang missing: %s, maths missing: %s", missing_lang, missing_maths)
                raise CustomException(f"Missing required features. lang missing: {missing_lang}, maths missing: {missing_maths}", sys)

            X_input_lang = X[lang_features].values
            X_input_maths = X[maths_features].values
            logging.debug("Prepared X_input_lang shape=%s X_input_maths shape=%s", X_input_lang.shape, X_input_maths.shape)

            lang_pred = lang_model.predict(X_input_lang)
            lang_pred_confidence = lang_model.predict_proba(X_input_lang).max(axis=1)

            maths_pred = maths_model.predict(X_input_maths)
            maths_pred_confidence = maths_model.predict_proba(X_input_maths).max(axis=1)

            logging.info("Models predicted: lang count=%d, maths count=%d", len(np.asarray(lang_pred).ravel()), len(np.asarray(maths_pred).ravel()))
            print(maths_pred_confidence)
            X['el_prediction_lang'] = np.array(lang_pred).ravel()
            X['el_prediction_maths'] = np.array(maths_pred).ravel()

            X['el_lang_confidence'] = np.array(lang_pred_confidence).ravel()
            X['el_maths_confidence'] = np.array(maths_pred_confidence).ravel()
            print(f"the shape of X dataframe is shape:{X.shape} and {X['Phase'].unique()}")
            print(f"The X dataframe has the following entries:{X.columns}")

            Phase1_df = X[X['Phase']==1][['community_id','student_id','bl_language','bl_mathematics','el_prediction_lang','el_prediction_maths','el_lang_confidence','el_maths_confidence']]
            Phase2_df = X[X['Phase']==2][['community_id','student_id','el_prediction_lang','el_prediction_maths','el_lang_confidence','el_maths_confidence']]
            Phase3_df = X[X['Phase']==3][['community_id','student_id','el_prediction_lang','el_prediction_maths','el_lang_confidence','el_maths_confidence']]

            Ph1_df = Phase1_df.rename(columns={'el_prediction_lang':'el1_prediction_lang',
                                        'el_prediction_maths':'el1_prediction_maths',
                                        'el_lang_confidence':'el1_lang_confidence',
                                        'el_maths_confidence':'el1_maths_confidence'})
            
            Ph2_df = Phase2_df.rename(columns={'el_prediction_lang':'el2_prediction_lang',
                                        'el_prediction_maths':'el2_prediction_maths',
                                        'el_lang_confidence':'el2_lang_confidence',
                                        'el_maths_confidence':'el2_maths_confidence'})
            
            Ph3_df = Phase3_df.rename(columns={'el_prediction_lang':'el3_prediction_lang',
                                        'el_prediction_maths':'el3_prediction_maths',
                                        'el_lang_confidence':'el3_lang_confidence',
                                        'el_maths_confidence':'el3_maths_confidence'})
            
            print(Ph1_df.student_id.nunique(),Ph2_df.student_id.nunique(),Ph3_df.student_id.nunique())
            mrg1 = pd.merge(Ph1_df,Ph2_df,on=['community_id','student_id'],how='left')
            final_merged = pd.merge(mrg1, Ph3_df, on=['community_id','student_id'], how='left')
            final_merged = final_merged.drop_duplicates(subset=['student_id'])
            print(X.head())
            logging.info("Attached prediction columns to DataFrame; returning DataFrame with shape %s", final_merged.shape)
            return final_merged
        except Exception as e:
            logging.exception("Error in PredictPipeline.predict")
            raise CustomException(e, sys)

