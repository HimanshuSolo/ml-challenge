import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
except Exception:
    def download_images(url, save_path):
        """Fallback download images; returns True if saved, False otherwise."""
        try:
            response = requests.get(url, stream=True, timeout=8)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception:
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            return False

model = None
embedder = None
resnet = None
preprocess = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_ipq(text):
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

def create_features(df, embedder_obj=None, image_dir="dataset/images/"):
    """
    Create features for text (embeddings or TF-IDF+SVD) and images (ResNet).
    Returns numpy array of shape (n_samples, feature_dim).
    """
    global resnet, preprocess, device

    os.makedirs(image_dir, exist_ok=True)
    catalog_content = df['catalog_content'].fillna('').astype(str)

    if SMOKE or embedder_obj is None:
        tfidf_cache = os.path.join(CACHE_DIR, f'tfidf_svd_smoke{int(SMOKE)}.npz')
        if os.path.exists(tfidf_cache):
            data = np.load(tfidf_cache)
            embeddings = data['emb']
        else:
            max_feats = 2000 if SMOKE else 5000
            tfidf = TfidfVectorizer(max_features=max_feats, ngram_range=(1,2), min_df=2)
            X_tfidf = tfidf.fit_transform(catalog_content.tolist())
            n_comp = 50 if SMOKE else 150
            svd = TruncatedSVD(n_components=min(n_comp, X_tfidf.shape[1]-1 or 1), random_state=42)
            embeddings = svd.fit_transform(X_tfidf)
            np.savez_compressed(tfidf_cache, emb=embeddings)
    else:
        embed_cache = os.path.join(CACHE_DIR, f'emb_{embedder_obj.__class__.__name__}_smoke{int(SMOKE)}.npz')
        if os.path.exists(embed_cache):
            data = np.load(embed_cache)
            embeddings = data['emb']
        else:
            encode_kwargs = dict(show_progress_bar=True, batch_size=64)
            try:
                embeddings = embedder_obj.encode(catalog_content.tolist(), device=str(device), **encode_kwargs)
            except TypeError:
                embeddings = embedder_obj.encode(catalog_content.tolist(), **encode_kwargs)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            np.savez_compressed(embed_cache, emb=embeddings)

    ipq = np.log1p(catalog_content.apply(extract_ipq)).values.reshape(-1, 1).astype(np.float32)
    text_length = np.log1p(catalog_content.str.len().values.reshape(-1, 1)).astype(np.float32)

    n = len(df)
    image_features = np.zeros((n, 1000), dtype=np.float32)  #

    if not NO_IMAGES:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if resnet is None:
            resnet = models.resnet50(pretrained=True)
            resnet = resnet.to(device)
            resnet.eval()
        else:
            resnet = resnet.to(device)
            resnet.eval()

        image_links = df.get('image_link', pd.Series(['']*n)).fillna('').values

        for i, url in enumerate(tqdm(image_links, desc="Processing images", disable=SMOKE)):
            if not url:
                continue
            save_path = os.path.join(image_dir, f"img_{i}.jpg")
            ok = download_images(url, save_path)
            if not ok or not os.path.exists(save_path):
                continue
            try:
                with Image.open(save_path) as img:
                    img = img.convert('RGB')
                    img_t = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = resnet(img_t)
                    feat = feat.cpu().numpy().reshape(-1)
                    if feat.shape[0] == 1000:
                        image_features[i] = feat.astype(np.float32)
            except Exception:
                pass
            finally:
                try:
                    os.remove(save_path)
                except Exception:
                    pass
    else:
        pass

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)

    assert embeddings.shape[0] == n, f"Embeddings rows {embeddings.shape[0]} != samples {n}"

    X = np.hstack((embeddings, ipq, text_length, image_features)).astype(np.float32)
    return X

def predictor(sample_id, catalog_content, image_link):
    global model, embedder, resnet, preprocess, device
    if model is None or embedder is None:
        raise RuntimeError("Model and embedder must be loaded before calling predictor()")

    try:
        emb = embedder.encode([catalog_content], device=str(device), show_progress_bar=False)[0]
    except TypeError:
        emb = embedder.encode([catalog_content], show_progress_bar=False)[0]
    emb = np.asarray(emb).reshape(1, -1).astype(np.float32)

    ipq = np.log1p(extract_ipq(catalog_content)).reshape(1, -1).astype(np.float32)
    text_len = np.log1p(len(catalog_content) if isinstance(catalog_content, str) else 0).reshape(1, -1).astype(np.float32)

    image_feature = np.zeros((1, 1000), dtype=np.float32)
    if (not NO_IMAGES) and image_link:
        tmp = f"temp_img_{sample_id}.jpg"
        ok = download_images(image_link, tmp)
        if ok and os.path.exists(tmp):
            try:
                with Image.open(tmp) as img:
                    img = img.convert('RGB')
                    img_t = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = resnet(img_t)
                    feat = feat.cpu().numpy().reshape(1, -1)
                    if feat.shape[1] == 1000:
                        image_feature = feat.astype(np.float32)
            except Exception:
                image_feature = np.zeros((1, 1000), dtype=np.float32)
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    X = np.hstack((emb, ipq, text_len, image_feature)).astype(np.float32)
    pred_log = model.predict(X)
    pred_price = np.expm1(pred_log[0])
    pred_price = float(max(0.01, pred_price))
    return round(pred_price, 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    train_path = os.path.join(DATASET_FOLDER, 'train.csv')
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        raise SystemExit(1)
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        raise SystemExit(1)

    df_train = pd.read_csv(train_path)
    if 'price' not in df_train.columns:
        raise KeyError("train.csv must contain 'price' column")
    df_train['price_num'] = pd.to_numeric(df_train['price'], errors='coerce')
    df_train = df_train[df_train['price_num'] > 0].reset_index(drop=True)
    y_train = np.log1p(df_train['price_num'].values).astype(np.float32)

    if not SMOKE:
        embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    else:
        embedder = None  

    print("Creating training features... (this may take a while)")
    X_train = create_features(df_train, embedder_obj=embedder, image_dir=os.path.join(DATASET_FOLDER, "images"))
    X_train = np.nan_to_num(X_train).astype(np.float32)

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
            'n_estimators': [500],  
            'learning_rate': [0.005, 0.01],
            'max_depth': [20],
            'num_leaves': [63],
            'lambda_l1': [0.1],
            'lambda_l2': [0.1]
        }

    try:
        price_bins = pd.qcut(df_train['price_num'], 5, labels=False, duplicates='drop')
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_func = lambda X, y: kf.split(X, price_bins)
    except Exception:
        print("Could not use StratifiedKFold (maybe too few unique bins). Falling back to KFold.")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        split_func = lambda X, y: kf.split(X)

    def smape_from_true_and_pred(y_true_vals, y_pred_vals):
        eps = 1e-8
        num = 2 * np.abs(y_pred_vals - y_true_vals)
        den = np.abs(y_pred_vals) + np.abs(y_true_vals) + eps
        return 100.0 * np.mean(num / den)

    best_smape = float('inf')
    best_params = None

    # grid search (nested loops)
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for md in param_grid['max_depth']:
                for nl in param_grid['num_leaves']:
                    for l1 in param_grid['lambda_l1']:
                        for l2 in param_grid['lambda_l2']:
                            cv_smape = []
                            for train_idx, val_idx in split_func(X_train, y_train):
                                X_tr, X_val_cv = X_train[train_idx], X_train[val_idx]
                                y_tr, y_val_cv = y_train[train_idx], y_train[val_idx]

                                reg = lgb.LGBMRegressor(
                                    n_estimators=n_est,
                                    learning_rate=lr,
                                    max_depth=md,
                                    num_leaves=nl,
                                    lambda_l1=l1,
                                    lambda_l2=l2,
                                    random_state=42,
                                    n_jobs=4,
                                )
                                try:
                                    reg.fit(X_tr, y_tr,
                                            eval_set=[(X_val_cv, y_val_cv)],
                                            early_stopping_rounds=50,
                                            verbose=False)
                                except TypeError:
                                    reg.fit(X_tr, y_tr)

                                y_val_pred_log = reg.predict(X_val_cv)
                                y_val_pred = np.expm1(y_val_pred_log)
                                y_val_true = np.expm1(y_val_cv)
                                smape_cv = smape_from_true_and_pred(y_val_true, y_val_pred)
                                cv_smape.append(smape_cv)

                            avg_smape = np.mean(cv_smape)
                            print(f"Params: n_est={n_est}, lr={lr}, md={md}, nl={nl}, l1={l1}, l2={l2}, Avg SMAPE={avg_smape:.2f}%")
                            if avg_smape < best_smape:
                                best_smape = avg_smape
                                best_params = {
                                    'n_estimators': n_est, 'learning_rate': lr,
                                    'max_depth': md, 'num_leaves': nl,
                                    'lambda_l1': l1, 'lambda_l2': l2
                                }

    print(f"Best parameters: {best_params}, Best CV SMAPE: {best_smape:.2f}%")

    if best_params is None:
        raise RuntimeError("No best params found (grid may be empty)")

    model = lgb.LGBMRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        num_leaves=best_params['num_leaves'],
        lambda_l1=best_params['lambda_l1'],
        lambda_l2=best_params['lambda_l2'],
        random_state=42,
        n_jobs=4
    )

    print("Training final model on full training set...")
    model.fit(X_train, y_train)

    X_tr_split, X_val, y_tr_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.fit(X_tr_split, y_tr_split)
    y_val_pred = np.expm1(model.predict(X_val))
    y_val_true = np.expm1(y_val)
    final_smape = smape_from_true_and_pred(y_val_true, y_val_pred)
    print(f"Final Validation SMAPE on holdout: {final_smape:.2f}%")

    print("Creating test features and generating predictions...")
    df_test = pd.read_csv(test_path)
    X_test = create_features(df_test, embedder_obj=embedder, image_dir=os.path.join(DATASET_FOLDER, "images"))
    X_test = np.nan_to_num(X_test).astype(np.float32)

    pred_log = model.predict(X_test)
    pred_prices = np.expm1(pred_log)
    pred_prices = np.clip(pred_prices, 0.01, None)
    df_test['price'] = np.round(pred_prices.astype(float), 2)

    output_df = df_test[['sample_id', 'price']] if 'sample_id' in df_test.columns else df_test[['price']].reset_index().rename(columns={'index':'sample_id'})

    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(output_df.head())
