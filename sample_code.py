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
import requests
from PIL import Image
import torch
from torchvision import models, transforms
import warnings
warnings.filterwarnings('ignore')

SMOKE = bool(int(os.environ.get('SMOKE', '0')))
NO_IMAGES = bool(int(os.environ.get('NO_IMAGES', '0')))
CACHE_DIR = os.path.join('dataset', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    from src.utils import download_images
except ImportError:
    def download_images(url, save_path):
        """Fallback function if download_images is not available."""
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception:
            return False
        return True

# IPQ extractoin function
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def create_features(df, embedder=None, image_dir="dataset/images/"):
    """
    Create combined features from text and images.
    """
    os.makedirs(image_dir, exist_ok=True)
    catalog_content = df['catalog_content'].fillna('')
    # Text features
    # Embedding or TF-IDF fallback
    if SMOKE or embedder is None:
        # TF-IDF + small SVD fallback for fast runs
        tfidf_cache = os.path.join(CACHE_DIR, f'tfidf_svd_smoke{int(SMOKE)}.npz')
        if os.path.exists(tfidf_cache):
            data = np.load(tfidf_cache)
            embeddings = data['emb']
        else:
            tfidf = TfidfVectorizer(max_features=2000 if SMOKE else 5000, ngram_range=(1,2), min_df=2)
            X_tfidf = tfidf.fit_transform(catalog_content.tolist())
            svd = TruncatedSVD(n_components=50 if SMOKE else 150, random_state=42)
            embeddings = svd.fit_transform(X_tfidf)
            np.savez_compressed(tfidf_cache, emb=embeddings)
    else:
        embed_cache = os.path.join(CACHE_DIR, f'emb_{embedder.__class__.__name__}_smoke{int(SMOKE)}.npz')
        if os.path.exists(embed_cache):
            data = np.load(embed_cache)
            embeddings = data['emb']
        else:
            embeddings = embedder.encode(catalog_content.tolist(), show_progress_bar=True, batch_size=32)
            np.savez_compressed(embed_cache, emb=embeddings)
    ipq = np.log1p(catalog_content.apply(extract_ipq)).values.reshape(-1, 1)
    text_length = np.log1p(catalog_content.str.len().values.reshape(-1, 1))
    
    # Image features
    image_links = df['image_link'].values
    image_features = np.zeros((len(df), 1000))  # 1000 dims from ResNet-50
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Load resnet once (only if images are used)
    resnet = None
    if not NO_IMAGES:
        resnet = models.resnet50(pretrained=True)
        # Ensure model runs on CPU
        resnet = resnet.cpu()
    resnet.eval()
    with torch.no_grad():
        if NO_IMAGES:
            # Skip downloading and processing images to speed up
            return np.hstack((embeddings, ipq, text_length, image_features))
        # Set default device to CPU
        device = torch.device('cpu')
        batch_size = 16  # Process images in batches
        for i in tqdm(range(0, len(image_links), batch_size), desc="Processing image batches"):
            batch_urls = image_links[i:i + batch_size]
            batch_features = []
            
            for j, url in enumerate(batch_urls):
                save_path = os.path.join(image_dir, f"img_{i+j}.jpg")
                if download_images(url, save_path):
                    try:
                        img = Image.open(save_path).convert('RGB')
                        img = preprocess(img).unsqueeze(0).to(device)
                        batch_features.append(img)
                    except Exception:
                        batch_features.append(None)
                    finally:
                        try:
                            os.remove(save_path)  # Clean up after processing
                        except Exception:
                            pass
                else:
                    batch_features.append(None)
            
            # Process valid images in batch
            valid_indices = [j for j, feat in enumerate(batch_features) if feat is not None]
            if valid_indices:
                valid_features = torch.cat([batch_features[j] for j in valid_indices], dim=0)
                with torch.no_grad():
                    features = resnet(valid_features).cpu().numpy()
                for idx, feat in zip(valid_indices, features):
                    image_features[i + idx] = feat
    
    return np.hstack((embeddings, ipq, text_length, image_features))

# Predictor function
def predictor(sample_id, catalog_content, image_link, batch_mode=False):
    """
    Predict price using a trained LightGBM model based on text and image features.
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    - batch_mode: If True, assumes inputs are lists/arrays for batch processing
    
    Returns:
    - price: Predicted price as a float (or list of floats in batch mode)
    """
    global model, embedder, resnet, preprocess
    
    if not batch_mode:
        catalog_content = [catalog_content]
        image_link = [image_link]
        sample_id = [sample_id]
    
    # Process text features in batch
    embeddings = embedder.encode(catalog_content, show_progress_bar=False, batch_size=32)
    ipqs = np.array([np.log1p(extract_ipq(text)) for text in catalog_content]).reshape(-1, 1)
    text_lens = np.array([np.log1p(len(text) if isinstance(text, str) else 0) for text in catalog_content]).reshape(-1, 1)
    
    # Process image
    save_path = f"temp_img_{sample_id}.jpg"
    if download_images(image_link, save_path):
        try:
            img = Image.open(save_path).convert('RGB')
            img = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                image_feature = resnet(img).flatten().numpy().reshape(1, -1)
        except Exception:
            image_feature = np.zeros((1, 1000))
        os.remove(save_path)
    else:
        image_feature = np.zeros((1, 1000))
    
    X = np.hstack((embedding, ipq, text_len, image_feature))
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
    
    # Feature engineering
    # Initialize SentenceTransformer with CPU device
    embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device='cpu')  # 768-dimensional embeddings
    X_train = create_features(df_train, embedder)
    
    # Hyperparameter tuning with regularization
    if SMOKE:
        param_grid = {
            'n_estimators': [200],
            'learning_rate': [0.01],
            'max_depth': [10],
            'num_leaves': [31],
            'lambda_l1': [0.0],
            'lambda_l2': [0.0]
        }
    else:
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
                                    verbose=-1,
                                    n_jobs=4
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
    
    # Train final model with best parameters
    model = lgb.LGBMRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        num_leaves=best_params['num_leaves'],
        lambda_l1=best_params['lambda_l1'],
        lambda_l2=best_params['lambda_l2'],
        random_state=42,
        verbose=-1,
        n_jobs=4
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