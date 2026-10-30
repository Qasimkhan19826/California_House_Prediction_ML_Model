# (10) Final Stage ,here we will complete ou project we use all the learned the concepts 
# we learned previously   .

# Model Persistence and Inference with Joblib
# Lets now summarize how to train a Random Forest model on California housing data,
# save the model and preprocessing pipeline using joblib, and reuse the model later 
# for inference on new data (input.csv). This approach helps avoid retraining the model every time, 
# improving performance and enabling reproducibility.

# 1. Why Train Once and Save?
# Training models repeatedly is time-consuming and computationally expensive.
# Saving the model (model.pkl) and preprocessing pipeline (pipeline.pkl) ensures you can quickly 
# load and run inference anytime in the future.(joblib)

# 2. Why Use a Preprocessing Pipeline?
# Raw data needs to be cleaned, scaled, and encoded before model training.
# A Pipeline automates this transformation and ensures identical preprocessing during inference.

# 3. Why Use Joblib?
# joblib efficiently serializes large NumPy arrays (like in sklearn models).
# Faster and more suitable than pickle for scikit-learn objects.

# 4. Why the If-Else Logic?
# The program checks if a saved model exists.
# If not, it trains and saves the model.
# If it does, it skips training and only runs inference, saving time.

'''
# 3. Why Use Joblib?
# Joblib is used to save a trained Machine Learning model into a file
# so that we don't need to train the model again and again.
# Later, we can load the saved model and use it to make predictions.

# Joblib efficiently serializes large NumPy arrays.
# Serialization means converting Python objects into a format
# that can be saved in a file and loaded again later.

# It is often faster and more suitable than pickle for
# scikit-learn models that contain large NumPy arrays.

# What is Inference?
# Inference means using a trained ML model to make predictions
# on new data without training the model again.
'''

# Let's start to write our final code :

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler ,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import os
import joblib

# Creating Constant Variable in Python .
MODEL_FILE = "model.pkl" # This will create our model file for inference  .(Created by joblib)
PIPELINE_FILE = "pipeline.pkl" # Why we are creating pkl for pipeline because our incoming data 
# will come with the raw feature .There will be no imputer ,no scaler ,no OneHotEncoding ,etc
# We want our incoming data get transform ,similarly like our train data for inference .

def build_pipeline(num_attributes,cat_attribute):
    # For numerical columns.
    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    # For categorical columns.
    cat_pipeline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore")) # handle_unknown="ignore": It will ignore the 
        # unknown values (means: Suppose in your training data there are 5 category and than 
        # one more category appears in the data due to any reason's ,it will ignore that value )
    ])

    #  Construct full pipeline
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attributes),
        ('cat',cat_pipeline,cat_attribute)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Let's train the model
    housing = pd.read_csv("housing.csv")

    # Creating StratifiedShuffle spilt .

    # Before spilting data ,adding income_cat column by using median_income (You already know the reason )
    housing['income_cat'] = pd.cut(housing["median_income"],
                        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                        labels=[1,2,3,4,5]) 
    # Spiliting the data 
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for Train_index, Test_index in split.split(housing,housing["income_cat"]):
        # Here we move the Test_index part upper to the Train_index because it overridens the housing 
        housing.loc[Test_index].drop("income_cat",axis=1).to_csv("input_data.csv",index=False)
        housing = housing.loc[Train_index].drop("income_cat",axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1) 

    # Taking numerical attribute column name and categorical attribute column name  
    num_attributes = housing_features.drop("ocean_proximity",axis=1).columns.tolist() 
    cat_attribute = ['ocean_proximity']

    pipeline = build_pipeline(num_attributes,cat_attribute)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    # Now we will use joblib which dumps our model in  MODEL_FILE for inference 

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("Model Is Trained Congrats !")

else:
    # Load the model for inference 
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # We will read input csv ,which is our test set
    input_data = pd.read_csv("input_data.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions # Median house set to predictions

    input_data.to_csv("output.csv",index=False)
    print("Inference is complete , results saved to output.csv ")

# With this setup, our ML pipeline is:

# Efficient – No retraining needed if the model exists.
# Reproducible – Same preprocessing logic every time.
# Production-ready – Can be deployed or reused across multiple systems.

# We do a lot of things in future to improve the model performance 
# We can use Flask ,Rest Api to show the output,etc .

# Conclusion and Wrapping up House Pricing Project
# In this project, we built a complete machine learning pipeline 
# to predict California housing prices using various regression algorithms.
# We started by:

# Loading and preprocessing the dataset (housing.csv) with careful treatment of missing values, scaling, and encoding using a custom pipeline.

# Stratified splitting was used to maintain income category distribution between train and test sets.

# We trained and evaluated multiple algorithms including:

# Linear Regression
# Decision Tree Regressor
# Random Forest Regressor
# Through cross-validation, we found that Random Forest performed the best, offering the lowest RMSE and most stable results.

# Finally, we built a script that:

# Trains the Random Forest model and saves it using joblib.
# Uses an if-else logic to skip retraining if the model exists.
# Applies the trained model to new data (input.csv) to predict median_house_value, storing results in output.csv.
# This pipeline ensures that predictions are accurate, efficient, and ready for production deployment.  



    







