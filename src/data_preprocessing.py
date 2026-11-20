from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from imblearn.over_sampling import SMOTE

def data_preprocessing(df):
    # Step1 : Drop Duplicate values
    # Step2 : Feature Engnieering (Optional)
    # Step3 : Split the dataset into x and y
    # Step4 : Split the dataset into train and test
    # Step5 : Segregate data into numerical and categorical
    # Step6 : Build numerical_pipeline : imputation and scailing
    # Step7 : Build categorical_pipeline : imputation and encoding
    # Step8 : Concat numerical and categorical pipeline using ColumnTransformer
    # Step9 : Fit the transformer on train and test
    # Step10 : fit_resample the data so that target column balance(only for classfication)
    # Step11 : Return the preprocessed X_train,X_test,y_train,y_test
    #---------------------------------------------------------------------------------------
    # Step1 : drop the duplicate values
    df.drop_duplicates(inplace = True)

    #Step2 : Feature Eng(Optional)

    # Step3 : Split the dataset into X and y 
    X = df.drop(columns = ['customerID','Churn'])
    y = df['Churn']

    # Step4 : Split the dataset into train and test
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                    test_size = 0.3,
                                                    random_state = 1) #shuffling the data

    # Step5 : Segregate numerical and categorical columns
    numerical_col = X_train.select_dtypes(exclude = 'object').columns
    categorical_col = X_train.select_dtypes(include = 'object').columns

    # Step6 Build numerical_pipeline
    numerical_pipeline = Pipeline([('imputation',SimpleImputer(strategy = 'median')),
                                   ('scailing',MinMaxScaler())])
    
    #  Step7 : Build categorical_pipeline
    categorical_pipeline = Pipeline([('imputation',SimpleImputer(strategy = 'most_frequent')),
                                     ('encoding',OneHotEncoder(handle_unknown = 'ignore'))])

    # Step8 : Use Column_Transformer: to concat numerical and categorical pipline
    transformer = ColumnTransformer([('numerical_pipeline',numerical_pipeline,numerical_col),
                                     ('categorical_pipeline',categorical_pipeline,categorical_col)])

    #  Step9 : Fit the transformer on train and test
    X_train = transformer.fit_transform(X_train)# Seen data
    X_test = transformer.transform(X_test) # Unseen data

    # Step10 : Use traget column balance technique
    '''
    * Generally used with classification model
    * Target Column balance technique basically classified into two..
    1) over_sampling technique : Based on rows you have,higher the data good amt of seen and unseen.
    2) under_sampling technique : when data is too large,ensure data is large enough to have seen and unseen data.
    (computation time: go for under sampling(Cluster Centroid))
    '''
    X_train,y_train = SMOTE().fit_resample(X_train,y_train) 

    #  Step11 : Return preprocessed X_train,X_test,y_train,y_test
    return X_train,X_test,y_train,y_test
