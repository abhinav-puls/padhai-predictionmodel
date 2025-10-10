from flask import Flask, jsonify
import pandas as pd
import traceback
import subprocess
import sys
import os
import datetime
from src.components.data_prediction import PredictPipeline
from src.components.data_transformation import DataTransformation

app = Flask(__name__)

# path to the test dataset produced by your data transformation step
TEST_DATA_PATH = "artifact/test.csv"

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Run the data transformation step in-process (use the same interpreter)
        dt = DataTransformation()
        try:
            result = dt.initiate_data_transformation()
            if isinstance(result, tuple) and len(result) >= 5:
                # expected: student_df, community_summary, baseline_dataset, skills_dist, test_data
                test_df = result[4]
            elif isinstance(result, pd.DataFrame):
                test_df = result
            else:
                # fallback to loading the artifact file written by the transformer
                test_df = pd.read_csv(TEST_DATA_PATH)
        except Exception:
            # if transformation raised, try to load the artifact file (allows idempotent runs)
            test_df = pd.read_csv(TEST_DATA_PATH)

        # instantiate prediction pipeline and run predictions
        # ensure test_df is a DataFrame (coerce if possible)
        if test_df is None:
            return jsonify({"status": "error", "message": "No test data available"}), 400

        # robust coercion/selection logic for common return types from DataTransformation
        if not isinstance(test_df, pd.DataFrame):
            try:
                # if tuple/list, prefer a contained DataFrame (commonly the test df is last)
                if isinstance(test_df, (list, tuple)):
                    dfs = [x for x in test_df if isinstance(x, pd.DataFrame)]
                    if dfs:
                        test_df = dfs[-1]
                    else:
                        last = test_df[-1]
                        if isinstance(last, str):
                            test_df = pd.read_csv(last)
                        else:
                            test_df = pd.DataFrame(last)

                # if dict, look for common keys or a DataFrame value
                elif isinstance(test_df, dict):
                    for key in ("test", "test_df", "test_data", "df"):
                        if key in test_df and isinstance(test_df[key], pd.DataFrame):
                            test_df = test_df[key]
                            break
                    else:
                        # take first value and try to coerce
                        val = next(iter(test_df.values()))
                        if isinstance(val, pd.DataFrame):
                            test_df = val
                        elif isinstance(val, str):
                            test_df = pd.read_csv(val)
                        else:
                            test_df = pd.DataFrame(val)

                # if it's a path string, load CSV
                elif isinstance(test_df, str):
                    test_df = pd.read_csv(test_df)

                # if Series, convert to single-row DataFrame
                elif isinstance(test_df, pd.Series):
                    test_df = test_df.to_frame().T

                else:
                    # final attempt
                    test_df = pd.DataFrame(test_df)

            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": "Failed to coerce test data to DataFrame",
                    "detail": str(e),
                    "type": str(type(test_df))
                }), 400

        pipeline = PredictPipeline()
        pred_df = pipeline.predict(test_df)

        # ensure we have a dataframe result
        if not isinstance(pred_df, pd.DataFrame):
            raise RuntimeError("Prediction pipeline did not return a DataFrame")

        # extract prediction columns and convert to plain lists for JSON
        lang_list = pred_df['el_prediction_lang'].astype(int).tolist()
        maths_list = pred_df['el_prediction_maths'].astype(int).tolist()

        # save preds to artifact/output with timestamped filename
        output_dir = os.path.join("artifact", "output")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join(output_dir, filename)
        pred_df.to_csv(output_path, index=False)

        return jsonify({
            "status": "success",
            "rows": len(pred_df),
            "lang_predictions": lang_list,
            "maths_predictions": maths_list,
            "output_path": output_path
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc().splitlines()[-1]
        }), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
