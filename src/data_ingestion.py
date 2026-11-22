import pandas as pd
def data_ingestion():
    '''
    * Title : Telecom customer Chrun Prediction Model 

    * Independent Column [x] : 'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'

    * Target Column [y] : 'Churn'

    * Problem Statement:
    1. To predict the churn rate
    2. Use Clustering approach to predict and classify the cluster pattern such as {L1,L2 & L3}
    '''
    df = pd.read_csv(r"C:\Mlops_Telecom_CustomerChurnPredictionModel\data\raw\churn.csv")

    return df