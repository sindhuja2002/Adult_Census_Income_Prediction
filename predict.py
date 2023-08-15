import pandas as pd
import pickle

def predict_salary(inputs, scaler_model, gb_model):

    # Mapping of education to education-num
    education_mapping = {'Bachelors': 13, 'HS-grad': 9, '11th': 7, 'Masters': 14, '9th': 5,
                         'Some-college': 10, 'Assoc-acdm': 12, 'Assoc-voc': 11, '7th-8th': 4,
                         'Doctorate': 16, 'Prof-school': 15, '5th-6th': 3, '10th': 6,
                         '1st-4th': 2, 'Preschool': 1, '12th': 8}

    # Extract numeric and categorical features
    num_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_nom_features = ['occupation', 'workclass', 'marital-status', 'relationship', 'race', 'sex', 'country']

    # Create input DataFrame
    input_df = pd.DataFrame([inputs])
    input_df['education-num'] = input_df['education'].replace(education_mapping)
    input_df.drop(['education'], axis=1, inplace=True)

    # Initialize DataFrame with feature names from the Gradient Boosting model
    test_df = pd.DataFrame(columns=gb_model.feature_names_in_)
    test_df_copy = test_df.copy()

    # One-hot encode categorical features
    encoded_df = pd.get_dummies(input_df, columns=cat_nom_features)

    # Scale numeric features
    encoded_df[num_features] = scaler_model.transform(encoded_df[num_features])

    # Populate test_df_copy with encoded data
    for column in test_df.columns:
        if column in encoded_df.columns:
            test_df_copy[column] = encoded_df[column]
        else:
            test_df_copy[column] = False

    # Make predictions
    predictions = gb_model.predict(test_df_copy)
    return predictions[0]
