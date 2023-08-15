from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def main():
    
    data = pd.read_csv('data/adult.csv')
    data['salary'] = data['salary'].astype('string')
    
    num_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
    cat_features = []
    for i in data.columns:
        if i not in num_features:
            cat_features.append(i)
    data[cat_features] = data[cat_features].replace('?', str(np.nan))

    cat_nom_features= ['occupation','workclass','marital-status','relationship','race','sex','country']
    cat_ord_feature = []
    for i in cat_features:
        if i not in cat_nom_features:
            cat_ord_feature.append(i)
    cat_encoded_data = pd.get_dummies(data, columns=cat_nom_features)
    print(cat_encoded_data)
    scaler = StandardScaler()
    cat_encoded_data[num_features] = scaler.fit_transform(cat_encoded_data[num_features])
    cat_encoded_data = cat_encoded_data.drop(['education'],axis = 1)
    x = cat_encoded_data.drop(['salary'], axis =1)
    y = cat_encoded_data['salary']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    gradient_boosting_model = GradientBoostingClassifier()
    gradient_boosting_model.fit(x_train, y_train)
    y_pred_gradient_boosting = gradient_boosting_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_gradient_boosting)
    report = classification_report(y_test, y_pred_gradient_boosting)
    print(f" GradientBoostingClassifier Metrics:")
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    with open('Gradientmodel.pkl', 'wb') as f:
        pickle.dump(gradient_boosting_model, f)
        print("Model GB Generated")
    with open('Scaler.pkl','wb') as f:
        pickle.dump(scaler, f)
        print('Scaling model Generated')

main()
