import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from scipy import sparse as sp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# IPQ extraction function
def extract_ipq(text):
    """
    Extract Item Pack Quantity (IPQ) from text using regex patterns.
    Returns 1 if no match is found.
    """
    if not isinstance(text, str):
        return 1
    text = text.lower()
    patterns = [
        r'\b(ipq|pack|count|pcs|pieces|qty|ct)\b[:\s-]*\s*(\d+)',
        r'(\d+)\s*[-x×]\s*(?:pack|pcs|pieces)?',
        r'(\d+)\s*(?:pack|count|pcs|pieces)\b'
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            numeric_match = re.search(r"(\d+)", m.group(0))
            if numeric_match:
                try:
                    return max(1, int(numeric_match.group(1)))
                except ValueError:
                    continue
    m = re.search(r'\b(\d+)\s*(?:items|units)?\b', text)
    return max(1, int(m.group(1))) if m else 1

# Feature engineering function
def create_features(df, embedder):
    catalog_content = df['catalog_content'].fillna('')
    embeddings = embedder.encode(catalog_content.tolist(), show_progress_bar=True, batch_size=32)
    ipq = np.log1p(catalog_content.apply(extract_ipq)).values.reshape(-1, 1)
    text_length = np.log1p(catalog_content.str.len().values.reshape(-1, 1))
    return np.hstack((embeddings, ipq, text_length))

# Predictor function
def predictor(sample_id, catalog_content, image_link):
    """
    Predict price using a trained LightGBM model based on text embeddings.
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image (unused in this version)
    
    Returns:
    - price: Predicted price as a float
    """
    global model, embedder
    embedding = embedder.encode([catalog_content], show_progress_bar=False)[0].reshape(1, -1)
    ipq = np.log1p(extract_ipq(catalog_content)).reshape(1, -1)
    text_len = np.log1p(len(catalog_content) if isinstance(catalog_content, str) else 0).reshape(1, -1)
    X = np.hstack((embedding, ipq, text_len))
    pred_log_price = model.predict(X)[0]
    return round(max(0.01, np.expm1(pred_log_price)), 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    # Load and preprocess training data
    train_path = os.path.join(DATASET_FOLDER, 'train.csv')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please ensure the dataset folder contains train.csv.")
        exit(1)
    
    df_train = pd.read_csv(train_path)
    df_train['price_num'] = pd.to_numeric(df_train['price'], errors='coerce')
    df_train = df_train[df_train['price_num'] > 0].reset_index(drop=True)
    y_train = np.log1p(df_train['price_num'].values)
    
    # Feature engineering with a stronger embedding model
    embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # 768-dimensional embeddings
    X_train = create_features(df_train, embedder)
    
    # Expanded hyperparameter tuning with regularization
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.005, 0.01],
        'max_depth': [20, 25],
        'num_leaves': [63, 127],
        'lambda_l1': [0.1, 0.5],
        'lambda_l2': [0.1, 0.5]
    }
    best_smape = float('inf')
    best_params = None
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    price_bins = pd.qcut(df_train['price_num'], 5, labels=False)  
    
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for md in param_grid['max_depth']:
                for nl in param_grid['num_leaves']:
                    for l1 in param_grid['lambda_l1']:
                        for l2 in param_grid['lambda_l2']:
                            cv_smape = []
                            for train_idx, val_idx in kf.split(X_train, price_bins):
                                X_tr, X_val_cv = X_train[train_idx], X_train[val_idx]
                                y_tr, y_val_cv = y_train[train_idx], y_train[val_idx]
                                model = lgb.LGBMRegressor(
                                    n_estimators=n_est,
                                    learning_rate=lr,
                                    max_depth=md,
                                    num_leaves=nl,
                                    lambda_l1=l1,
                                    lambda_l2=l2,
                                    random_state=42,
                                    verbose=-1
                                )
                                model.fit(X_tr, y_tr, early_stopping_rounds=50, eval_set=[(X_val_cv, y_val_cv)], 
                                          eval_metric='l1', verbose=False)
                                y_val_pred_cv = np.expm1(model.predict(X_val_cv))
                                smape_cv = 100 * np.mean(2 * np.abs(y_val_pred_cv - np.expm1(y_val_cv)) / 
                                                      (np.abs(y_val_pred_cv) + np.abs(np.expm1(y_val_cv)) + 1e-8))
                                cv_smape.append(smape_cv)
                            avg_smape = np.mean(cv_smape)
                            print(f"Params: n_est={n_est}, lr={lr}, md={md}, nl={nl}, l1={l1}, l2={l2}, Avg SMAPE={avg_smape:.2f}%")
                            if avg_smape < best_smape:
                                best_smape = avg_smape
                                best_params = {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': md, 
                                              'num_leaves': nl, 'lambda_l1': l1, 'lambda_l2': l2}
    
    print(f"Best parameters: {best_params}, Best CV SMAPE: {best_smape:.2f}%")
    
    # Train final model with best parameters on full data
    model = lgb.LGBMRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        num_leaves=best_params['num_leaves'],
        lambda_l1=best_params['lambda_l1'],
        lambda_l2=best_params['lambda_l2'],
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Validate on holdout set
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.fit(X_train_split, y_train_split)
    y_val_pred = np.expm1(model.predict(X_val))
    smape = 100 * np.mean(2 * np.abs(y_val_pred - np.expm1(y_val)) / (np.abs(y_val_pred) + np.abs(np.expm1(y_val)) + 1e-8))
    print(f"Final Validation SMAPE: {smape:.2f}%")
    
    # Read test data
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found. Please ensure the dataset folder contains test.csv.")
        exit(1)
    
    test = pd.read_csv(test_path)
    
    # Generate predictions
    print("Generating predictions...")
    X_test = create_features(test, embedder)
    
    pred_log = model.predict(X_test)
    pred_prices = np.expm1(pred_log)
    pred_prices = np.clip(pred_prices, 0.01, None)  # Ensure positive
    test['price'] = np.round(pred_prices.astype(float), 2)
    
    output_df = test[['sample_id', 'price']]
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")