import streamlit as st
import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import optuna
import xgboost as xgb
import joblib
from io import BytesIO

# Function to preprocess data
def preprocess_data(train, test):
    all_data = pd.concat((test.loc[:,:], train.loc[:, :]))
    train['SalePrice'] = np.log1p(train['SalePrice'])
    y_train = train['SalePrice']

    train = train.drop("Id", axis=1)
    test = test.drop("Id", axis=1)
    train = train.drop(["Street", "Utilities"], axis=1)
    test = test.drop(["Street", "Utilities"], axis=1)

    clear_data = train.drop(train[(train['GrLivArea'] > 4500)].index)
    train_ = clear_data.drop(['SalePrice'], axis=1)
    all_data = pd.concat([train, test]).reset_index(drop=True)
    all_data.loc[2592, 'GarageYrBlt'] = 2007

    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Convert specified columns to string
    def convert_to_string(df, columns):
        df[columns] = df[columns].astype(str)
        return df

    num_to_categ_features = ['MSSubClass', 'OverallCond']
    all_data = convert_to_string(all_data, columns=num_to_categ_features)

    # Imputation and transformation setup
    num_features = all_data.select_dtypes(include=['int64', 'float64']).columns
    num_features_to_constant = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', "MasVnrArea"]
    num_features_to_median = [feature for feature in num_features if feature not in num_features_to_constant + ["SalePrice"]]

    imputer = ColumnTransformer(
        transformers=[
            ('num_to_median', SimpleImputer(strategy='most_frequent'), num_features_to_median),
            ('num_to_constant', SimpleImputer(strategy='constant', fill_value=0), num_features_to_constant),
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )

    categorical_features = all_data.select_dtypes(include=['object']).columns
    
    none_conversion = [("MasVnrType", "None"),
                       ("BsmtQual", "NA"),
                       ("Electrical", "SBrkr"),
                       ("BsmtCond", "TA"),
                       ("BsmtExposure", "No"),
                       ("BsmtFinType1", "No"),
                       ("BsmtFinType2", "No"),
                       ("CentralAir", "N"),
                       ("Condition1", "Norm"),
                       ("Condition2", "Norm"),
                       ("ExterCond", "TA"),
                       ("ExterQual", "TA"),
                       ("FireplaceQu", "NA"),
                       ("Functional", "Typ"),
                       ("GarageType", "No"),
                       ("GarageFinish", "No"),
                       ("GarageQual", "NA"),
                       ("GarageCond", "NA"),
                       ("HeatingQC", "TA"),
                       ("KitchenQual", "TA"),
                       ("Functional", "Typ"),
                       ("GarageType", "No"),
                       ("GarageFinish", "No"),
                       ("GarageQual", "No"),
                       ("GarageCond", "No"),
                       ("HeatingQC", "TA"),
                       ("KitchenQual", "TA"),
                       ("MSZoning", "None"),
                       ("Exterior1st", "VinylSd"),
                       ("Exterior2nd", "VinylSd"),
                       ("SaleType", "WD")]

    for col, new_str in none_conversion:
        all_data.loc[:, col] = all_data.loc[:, col].fillna(new_str)

    categorical_features = all_data.select_dtypes(include=['object']).columns
    
    my_encoder = ColumnTransformer(
        transformers=[
            ('ordinal_encoding', OrdinalEncoder(), categorical_features)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    
    my_scaler = ColumnTransformer(
        transformers=[
            ('scaling_num_columns', StandardScaler(), num_features)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    
    preprocessor = Pipeline(
        steps=[
            ('imputer', imputer),
            ('my_encoder', my_encoder),
            ('my_scaler', my_scaler)
        ]
    )

    all_data = preprocessor.fit_transform(all_data)
    train_preprocessed = all_data.iloc[:len(train),:]
    test_preprocessed = all_data.iloc[len(train_preprocessed):,:]

    X_train_preprocessed = train_preprocessed.drop(columns=['SalePrice'])
    X_test_preprocessed = test_preprocessed.drop(columns=['SalePrice'])

    X_train_preprocessed = X_train_preprocessed.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
    X_test_preprocessed = X_test_preprocessed.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_preprocessed, y_train, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid, X_test_preprocessed, preprocessor

# Function to optimize and train Lasso model
def optimize_and_train_lasso(X_train, y_train, X_valid, y_valid):
    def lasso_optimization(trial):
        alpha = trial.suggest_float('alpha', 1e-10, 5.0)
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_valid)
        
        rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
        return rmsle

    lasso_study = optuna.create_study(direction='minimize')
    lasso_study.optimize(lasso_optimization, n_trials=100)

    lasso_best_params = lasso_study.best_params
    best_lasso = Lasso(alpha=lasso_best_params['alpha'], random_state=42)
    best_lasso.fit(X_train, y_train)

    selected_features = np.where(abs(best_lasso.coef_) != 0)[0]
    X_train_selected = X_train.iloc[:, selected_features]
    X_val_selected = X_valid.iloc[:, selected_features]
    X_test_selected = X_test_preprocessed.iloc[:, selected_features]

    return X_train_selected, X_val_selected, X_test_selected, best_lasso, selected_features

# Function to optimize and train XGBoost model
def optimize_and_train_xgb(X_train_selected, y_train, X_val_selected, y_valid):
    def xgb_optimization(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'random_state': 42
        }

        xg_reg = xgb.XGBRegressor(**param)
        xg_reg.fit(X_train_selected, y_train) 
        y_pred = xg_reg.predict(X_val_selected) 
        
        rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
        return rmsle

    xgb_study = optuna.create_study(direction='minimize')
    xgb_study.optimize(xgb_optimization, n_trials=100)

    xgb_best_params = xgb_study.best_params
    best_xgb = xgb.XGBRegressor(**xgb_best_params)
    best_xgb.fit(X_train_selected, y_train) 

    return best_xgb

# Function to optimize and apply PCA
def optimize_and_apply_pca(X_train_selected, y_train, X_val_selected, y_valid):
    def pca_optimization(trial):
        n_components = trial.suggest_uniform('n_components', 0.8, 0.99)
        pca = PCA(n_components=n_components)
        
        X_train_pca = pca.fit_transform(X_train_selected)
        X_val_pca = pca.transform(X_val_selected)
        
        xgbr = xgb.XGBRegressor(random_state=42)
        xgbr.fit(X_train_pca, y_train)
        y_pred = xgbr.predict(X_val_pca)
        
        rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
        return rmsle

    pca_study = optuna.create_study(direction='minimize')
    pca_study.optimize(pca_optimization, n_trials=100)

    pca_best_params = pca_study.best_params
    pca = PCA(n_components=pca_best_params['n_components'])
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    return pca, X_train_pca, X_test_pca

# Streamlit UI
st.title("House Prices Prediction")

st.write("Upload your train and test CSV files to get the prediction results.")

uploaded_train = st.file_uploader("Choose a train CSV file", type="csv")
uploaded_test = st.file_uploader("Choose a test CSV file", type="csv")

if uploaded_train is not None and uploaded_test is not None:
    train = pd.read_csv(uploaded_train)
    test = pd.read_csv(uploaded_test)
    st.write("Files uploaded successfully. Processing data...")

    # Preprocess data
    X_train, X_valid, y_train, y_valid, X_test_preprocessed, preprocessor = preprocess_data(train, test)

    # Optimize and train models
    X_train_selected, X_val_selected, X_test_selected, best_lasso, selected_features = optimize_and_train_lasso(X_train, y_train, X_valid, y_valid)
    best_xgb = optimize_and_train_xgb(X_train_selected, y_train, X_val_selected, y_valid)
    pca, X_train_pca, X_test_pca = optimize_and_apply_pca(X_train_selected, y_train, X_val_selected, y_valid)
    
    best_xgb.fit(X_train_pca, y_train) 
    y_pred_train_xgb = best_xgb.predict(X_train_pca)
    rmsle = np.sqrt(mean_squared_log_error(y_train, y_pred_train_xgb))

    # Make predictions
    y_pred_test_xgb = best_xgb.predict(X_test_pca)
    submission = pd.read_csv('sample_submission.csv')
    submission['SalePrice'] = np.expm1(y_pred_test_xgb)

    # Download submission file
    st.write("## Download Prediction Results")
    buffer = BytesIO()
    submission.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(label="Download Submission CSV", data=buffer, file_name="submission.csv", mime="text/csv")

    st.stop()
    
else:
    st.stop()
