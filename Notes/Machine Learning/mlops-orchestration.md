# Machine Learning Pipelines

*Note*: These notes were generated using AI based on the following two videos from MLOps Zoomcamp:
1. [Machine Learning Pipelines](https://www.youtube.com/watch?v=uAR4BhVCNbI)
2. [Turning the Notebook into a Python Script](https://www.youtube.com/watch?v=3_Uu0rInxWI)

## Introduction
This module will cover the transformation of a Jupyter notebook containing machine learning experiments into a reproducible, re-runnable, and parameterized training pipeline.

## What is a Training Pipeline?
A training pipeline is a sequence of steps executed to train a machine learning model. A previous notebook, which included experiment tracking and parameter tuning, resulted in a model saved in a model registry. However, the notebook format can be messy, difficult to maintain, and not easily re-executable.

### Workflow Orchestration
Workflow orchestration is a more general term describing the scheduling and organisation of steps. A machine learning pipeline is a specific type of workflow orchestration tailored for machine learning model production.

## Simple Machine Learning Pipeline Steps
The following steps are commonly involved in creating a machine learning model.
1.  **Download Data (Ingestion):** Retrieving data from a source (local machine, database, etc.).
2.  **Transform Data:** Processing the data, including:
    *   Filtering.
    *   Removing outliers.
    *   Aggregates.
    *   Converting data types.
    *   Computing durations.
3.  **Prepare Data for Machine Learning:** Feature engineering and creating the feature matrix (`X`) and target array (`Y`).
4.  **Hyperparameter Tuning:** Finding the optimal set of parameters, often using libraries like `hyperopt`.
5.  **Train Model:** Training the final model using the best parameters found.

### Model Registry
The final trained model is saved in a model registry, making it accessible for deployment and other uses.

## Converting the Notebook to a Python Script
A simple approach is to convert the notebook into a Python script with well-defined functions for each step.

```python
def download_data():
    # Code to download the data
    pass

def transform_data(dataframe):
    # Code to transform the dataframe
    pass

def feature_engineering(dataframe):
    # Code to turn dataframe into a matrix and Y array
    pass

def find_best_model(x, y):
    # Hyperparameter tuning code
    pass

def train_model(x, y, parameters):
    # Code to train the best model
    pass

download_data()
data = transform_data()
x, y = feature_engineering(data)
parameters = find_best_model(x, y)
model = train_model(x, y, parameters)
```

This improves maintainability, allows for testing, and is a step towards a production-ready pipeline.

## Problems with a Simple Python Script
Using a single Python script has limitations:
*   **Scheduling:** How to schedule the script to run automatically (e.g., using cron)?
*   **Collaboration:** How to collaborate with other developers?
*   **Deployment:** Where to deploy and run the script (not on a local laptop)?
*   **Scalability:** How to handle increasing job volumes?
*   **Error Handling:** What happens if a step fails (e.g., due to a temporary network issue)? Adding retry mechanisms can complicate the code.

## Workflow Orchestrators
Workflow orchestrators are specialized tools that address the limitations of simple Python scripts.

### Benefits of Workflow Orchestrators
*   **Centralised:** Hosted on a server, enabling team collaboration and central code management.
*   **Scalable:** Resources can be added to the orchestrator as needed.
*   **Monitoring and Alerting:** Provide monitoring, alerting, and notifications.
*   **Dependency Management:** Manage dependencies between steps, ensuring they are executed in the correct order and handling failures appropriately.

### Examples of Workflow Orchestrators
*   **General Purpose:**
    *   Airflow
    *   Prefect
    *   Mage
    *   Duster
    *   Luigi
*   **Machine Learning Specific:**
    *   Kubeflow Pipelines
    *   MLflow (pipelines)

Machine learning specific tools are often less flexible but more focused on machine learning tasks.

# Turning Experiment Tracking into a Python Script
In this session, we will convert our existing experiment tracking setup (created in module 2) into a Python script.  This builds upon the previous discussion about machine learning pipelines, workflow orchestration, and its benefits. The goal is to prepare this script for integration with a workflow orchestration tool, resulting in a functional machine learning pipeline.

Right now, we have a Jupyter Notebook, which is a sort of machine learning pipeline already: executing the notebook produces a model as an artifact. The aim of a machine learning pipeline is to produce a model usable for making predictions.

## Initial Notebook Setup and Modifications
The starting point is a downloaded (but unexecuted) Jupyter Notebook. The code will be available in the course repository. Let's execute and adjust the notebook code to ensure it produces a model.
1.  **Opening the Notebook:** Open the downloaded notebook in Jupyter.
2.  **MLflow Setup:** The original notebook used a specific method for connecting to MLflow. We'll switch to connecting to a local instance of MLflow. The tracking URI should be set to `localhost:5000` (assuming your MLflow instance is running there). Example:
```python
mlflow.set_tracking_uri("localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")
```

3.  **Data Handling:**  The original notebook used CSV files. Update the code to work with parquet files, which don't require explicit datetime conversion.  Instead of downloading the files, read directly from URLs. For example:

```python
data_url = "url_to_parquet_file"  # Replace with the actual URL
df = pd.read_parquet(data_url)
```
Make sure that the URL ends in `.parquet`.

4.  **Feature Engineering:**  Feature engineering tasks, like creating new features, should be grouped in a dedicated function. The goal is for the "read data frame" step to focus solely on reading the data, not modifying it.

5.  **Model Training:** The original notebook used specific models (potentially scikit-learn models). Instead, we'll use XGBoost. Replace the relevant import statements, ensure you've installed the library in your virtual env if you haven't already!

```python
import xgboost as xgb
```

6.  **Hyperparameter Tuning:**  The notebook may contain code for hyperparameter optimisation.  For this example, we'll skip that and use pre-determined best parameters. However, this process is important for optimising your model's performance.

7.  **Root Mean Squared Error (RMSE):** Make sure that the metric being used is root mean squared error rather than mean squared error.

8.  **Data Conversion:** XGBoost requires data to be in a specific internal matrix format. Convert your matrices accordingly.

```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
```

9.  **Training Rounds:**  Reduce the number of training rounds for speed. The default 1000 rounds may be excessive. Monitoring the validation performance can help determine an appropriate number of rounds (e.g., 30). The return diminishes after a certain number of rounds.

10. **Model Saving:**  Ensure the "models" folder exists before saving the model. Create the folder programmatically using `pathlib`.

```python
from pathlib import Path
Path("models").mkdir(parents=True, exist_ok=True)
model.save_model("models/my_model.json") # Example - adjust as needed
```

After these modifications, executing the notebook from top to bottom should successfully train and save a model.

## Converting the Notebook to a Python Script
Now, transform the notebook into a more structured Python script using `nbconvert`.

```bash
jupyter nbconvert --to script duration_prediction.ipynb
```

This creates a `duration_prediction.py` file with the notebook's code.

## Cleaning and Structuring the Script
1.  **Remove Metadata:** Delete any unnecessary metadata at the beginning of the script.
2.  **Organise Imports:** Group standard Python imports first, followed by third-party library imports. For example:

```python
import os
import pathlib
from argparse import ArgumentParser

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import mlflow
```

3.  **Parameterisation:** Replace hardcoded file names with year and month parameters. Create a function to build the URL list based on these parameters.

4.  **Feature Engineering Function:** Create a function called `create_X` for feature engineering:

```python
def create_X(df, dv=None, categorical=None, sparse=True):
    if categorical is None:
        categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    df[categorical] = df[categorical].astype(str)
    dicts = df[categorical + numerical].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse=sparse)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv
```

   This function should handle both training and validation data and create the matrix X.

5.  **Training Function:** Encapsulate the model training code into a `train_model` function:

```python
def train_model(X_train, y_train, X_val, y_val, dv):
  # Training code here
  return model
```

6.  **Main/Run Function:** Create a `run` function to orchestrate the data loading, feature engineering, and model training steps. It can take year and month as arguments.

7.  **Command-Line Arguments:** Use `argparse` to allow users to specify the year and month when running the script.

```python
import argparse
def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--year', type=int, required=True, help='The year of the data.')
    parser.add_argument('--month', type=int, required=True, help='The month of the data.')
    args = parser.parse_args()

    year = args.year
    month = args.month

    # ... (rest of your run function logic, using year and month)
```

8.  **Next Month/Year Logic:** Implement logic to handle the transition to the next month and year correctly (e.g., December to January).

## Testing and Enhancements
1.  **Test the Script:** Run the script with different year and month combinations to ensure it works correctly.
```bash
python duration_prediction.py --year 2021 --month 1
```

2.  **Output Run ID:** Consider saving the MLflow run ID to a file for tracking.

## Addressing Potential Issues and Workflow Orchestration
This script represents a basic machine learning pipeline. However, it's important to consider potential issues like network failures or data availability problems.

*   **Retry Mechanisms:**  For robust error handling, implement retry mechanisms for tasks like reading data or logging parameters to MLflow.
*   **Workflow Orchestration:**  To manage these complexities and create a more reliable and scalable pipeline, use a workflow orchestration tool like Airflow, Prefect, Mage, or Dagster.

The workflow orchestrator can handle retries, dependencies between tasks, and provide monitoring and alerting capabilities.  Convert the script into a workflow within your chosen orchestrator. Each step of the process (data loading, feature engineering, model training) becomes a task in the workflow.

This process transforms a basic script into a robust and manageable machine learning pipeline.

