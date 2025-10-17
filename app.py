from flask import Flask, jsonify

from dotenv import load_dotenv

import pandas as pd
import traceback
import subprocess
import sys
import os
import datetime
from src.components.data_prediction import PredictPipeline
from src.components.data_transformation import DataTransformation
from src.components.data_tagging import DataTagging
from src.components.class_profiles import compute_class_profiles
from flask_cors import CORS



# path to the test dataset produced by your data transformation step
TEST_DATA_PATH = os.path.join("artifact", "test.csv")
OUTPUT_DIR = os.path.join("artifact", "output")


def create_app():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    app = Flask(__name__)

    CORS(app)
    load_dotenv()

    # basic config (override with environment variables as needed)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
    app.config["WTF_CSRF_ENABLED"] = False

    
    # simple health checkup is the flask server running really?
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})


    @app.route("/tagging", methods=["GET"])
    def tagging():
        """
        Endpoint to run the DataTagging pipeline and tag mistakes made by students.
        """
        try:
            app.logger.info("Running the Tagging of the mistakes (In Progress ..)")

            dataTaggingVar = DataTagging()
            combined_df = dataTaggingVar.initiate_data_tagging()
            app.logger.info("Tagging process completed successfully.")

            dataProfiling = dataTaggingVar.data_profiling(combined_df)
            app.logger.info("Profiling process completed successfully.")

            return jsonify({
                "status": "success",
                "message": "Data tagging pipeline completed successfully.",
                "output_paths": {
                    "final_profiles": dataTaggingVar.config.finalProfiles,
                }
            }), 200

        except Exception as e:
            app.logger.exception("Error while running tagging pipeline.")
            return jsonify({
                "status": "error",
                "message": str(e),
                "trace": traceback.format_exc().splitlines()[-1]
            }), 500


    @app.route("/predict", methods=["GET"])
    def predict():
        try:
            app.logger.info("Running data transformation (in-process)")
            dt = DataTransformation()
            test_df = None
            try:
                result = dt.initiate_data_transformation()
                # common return shapes: tuple with test df at index 4, direct DataFrame, path, or None
                if isinstance(result, tuple) and len(result) >= 5 and isinstance(result[4], pd.DataFrame):
                    test_df = result[4]
                elif isinstance(result, pd.DataFrame):
                    test_df = result
                elif isinstance(result, str) and os.path.exists(result):
                    test_df = pd.read_csv(result)
                else:
                    # fallback: try to read artifact/test.csv
                    app.logger.info("Falling back to reading %s", TEST_DATA_PATH)
                    if os.path.exists(TEST_DATA_PATH):
                        test_df = pd.read_csv(TEST_DATA_PATH)
            except Exception:
                app.logger.exception("DataTransformation failed; attempting to load artifact/test.csv")
                if os.path.exists(TEST_DATA_PATH):
                    test_df = pd.read_csv(TEST_DATA_PATH)

            if test_df is None:
                msg = "No test data available after transformation"
                app.logger.error(msg)
                return jsonify({"status": "error", "message": msg}), 400

            # ensure DataFrame
            if not isinstance(test_df, pd.DataFrame):
                try:
                    test_df = pd.DataFrame(test_df)
                except Exception as e:
                    app.logger.exception("Failed to coerce test data to DataFrame")
                    return jsonify({
                        "status": "error",
                        "message": "Failed to coerce test data to DataFrame",
                        "detail": str(e),
                        "type": str(type(test_df))
                    }), 400

            app.logger.info("Calling PredictPipeline.predict on %d rows", len(test_df))
            pipeline = PredictPipeline()
            pred_df = pipeline.predict(test_df)

            if not isinstance(pred_df, pd.DataFrame):
                msg = "Prediction pipeline did not return a DataFrame"
                app.logger.error(msg)
                return jsonify({"status": "error", "message": msg}), 500


            # save output CSV
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"predictions.csv"
            output_path = os.path.join(OUTPUT_DIR, filename)
            pred_df.to_csv(output_path, index=False)
            app.logger.info("Saved predictions to %s", output_path)

            return jsonify({
                "status": "success",
                "rows": len(pred_df),
                "output_path": output_path
            })
        except Exception as e:
            app.logger.exception("Unhandled error in /predict")
            return jsonify({
                "status": "error",
                "message": str(e),
                "trace": traceback.format_exc().splitlines()[-1]
            }), 500


    @app.route("/classprofiles", methods=["GET"])
    def classprofiles():
        """
        Compute class/community profiles for all phases.
        This endpoint runs the prediction pipeline (in-process) to obtain predicted dataframe,
        then computes and saves class profiles and returns the CSV path and summary counts.
        """
        try:
            app.logger.info("Running prediction to obtain data for class profiling")
            pipeline = PredictPipeline()
            # attempt to load existing artifact first
            pred_df = None
            if os.path.exists(os.path.join(OUTPUT_DIR, "predictions.csv")):
                try:
                    pred_df = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))
                    app.logger.info("Loaded existing predictions from artifact/output/predictions.csv")
                except Exception:
                    app.logger.exception("Failed to load existing predictions file; will call pipeline.predict")

            if pred_df is None:
                # run full transformation + prediction
                dt = DataTransformation()
                res = dt.initiate_data_transformation()
                if isinstance(res, tuple) and len(res) >= 5 and isinstance(res[4], pd.DataFrame):
                    test_df = res[4]
                elif isinstance(res, pd.DataFrame):
                    test_df = res
                elif isinstance(res, str) and os.path.exists(res):
                    test_df = pd.read_csv(res)
                else:
                    if os.path.exists(TEST_DATA_PATH):
                        test_df = pd.read_csv(TEST_DATA_PATH)
                    else:
                        return jsonify({"status": "error", "message": "No test data available for profiling"}), 400

                pred_df = pipeline.predict(test_df)

            profiles_df, out_path = compute_class_profiles(pred_df)
            return jsonify({
                "status": "success",
                "rows": len(pred_df),
                "profiles_path": out_path
            }), 200
        except Exception as e:
            app.logger.exception("Error while computing class profiles")
            return jsonify({
                "status": "error",
                "message": str(e),
                "trace": traceback.format_exc().splitlines()[-1]
            }), 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)



