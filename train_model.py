import os
import re
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse

import nltk
nltk.download('punkt')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB



# ===========================================================
# 1. LOAD ALL DATASETS
# ===========================================================

DATA_DIR = "data"

all_files = [
    "CEAS_08.csv",
    "Nazario.csv",
    "SpamAssasin.csv",
    "phishing_email.csv",
    "Nigerian_Fraud.csv",
    "Enron.csv",
    "Ling.csv"
]

dfs = []
for f in all_files:
    df_path = os.path.join(DATA_DIR, f)
    if os.path.exists(df_path):
        print("Loading:", df_path)
        dfs.append(pd.read_csv(df_path))

df = pd.concat(dfs, axis=0, ignore_index=True)
df = df.dropna(subset=["body"])
df.reset_index(drop=True, inplace=True)

print("\nLoaded Dataset Shape:", df.shape)



# ===========================================================
# 2. CLEAN EMAIL BODY & EXTRACT URLS
# ===========================================================

def clean_email_content_with_links(body):
    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        body
    )

    # Replace URLs with placeholders
    for i, u in enumerate(urls):
        body = body.replace(u, f"*URL_{i}*")

    # Keep URL placeholders + alphanumeric
    body = re.sub(r"[^a-zA-Z0-9\s*URL_\d*]", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    return body, urls


df["cleaned_body"], df["extracted_urls"] = zip(*df["body"].apply(clean_email_content_with_links))



# ===========================================================
# 3. SAFE URL HELPERS (NO MORE IPv6 ERRORS)
# ===========================================================

def safe_parse(url):
    try:
        return urlparse(url)
    except:
        return None

def get_domain_from_url(url):
    parsed = safe_parse(url)
    if parsed and parsed.netloc:
        return parsed.netloc
    return ""

def is_url_shortened(url):
    parsed = safe_parse(url)
    if not parsed:
        return 0
    domain = parsed.netloc or ""
    return int(any(s in domain for s in ["bit.ly", "tinyurl.com", "goo.gl"]))

def count_special_chars(url):
    try:
        return sum(c in "@%&*#$?+!" for c in url)
    except:
        return 0



# ===========================================================
# 4. URL FEATURE EXTRACTION
# ===========================================================

def extract_url_features(row):
    urls = row["extracted_urls"]

    if not isinstance(urls, list) or len(urls) == 0:
        return [0, 0, 0, 0, 0, 0]

    safe_urls = [u for u in urls if isinstance(u, str)]
    if len(safe_urls) == 0:
        return [0, 0, 0, 0, 0, 0]

    avg_len = np.mean([len(u) for u in safe_urls])
    specials = sum(count_special_chars(u) for u in safe_urls)
    shortened = sum(is_url_shortened(u) for u in safe_urls)
    params = sum(u.count("?") for u in safe_urls)
    https = sum("https://" in u.lower() for u in safe_urls)

    suspicious_words = ["login", "bank", "verify", "account", "security"]
    suspicious = sum(any(k in u.lower() for k in suspicious_words) for u in safe_urls)

    return [avg_len, specials, shortened, params, https, suspicious]


url_features = df.apply(extract_url_features, axis=1)
url_features_df = pd.DataFrame(url_features.tolist(), columns=[
    "avg_url_length", "num_special_chars", "has_shortener",
    "num_query_params", "has_https", "num_suspicious_kw"
])



# ===========================================================
# 5. ENCODE SENDER DOMAIN
# ===========================================================

df["sender_domain"] = df["sender"].apply(
    lambda x: x.split("@")[-1] if isinstance(x, str) and "@" in x else "unknown"
)
df["sender_domain"] = pd.factorize(df["sender_domain"])[0]

df["num_urls"] = df["extracted_urls"].apply(len)
df["body_length"] = df["cleaned_body"].apply(len)



# ===========================================================
# 6. VECTORIZE EMAIL TEXT
# ===========================================================

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf.fit_transform(df["cleaned_body"])
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

count_vec = CountVectorizer(stop_words="english", max_features=5000)
X_nb = count_vec.fit_transform(df["cleaned_body"])
X_nb_df = pd.DataFrame(X_nb.toarray(), columns=count_vec.get_feature_names_out())



# ===========================================================
# 7. COMBINE FEATURES FOR LIGHTGBM
# ===========================================================

X_lgb = pd.concat([
    X_tfidf_df,
    df[["num_urls", "sender_domain", "body_length"]],
    url_features_df
], axis=1)

y = df["label"]



# ===========================================================
# 8. TRAIN–TEST SPLIT
# ===========================================================

X_train_lgb, X_test_lgb, y_train, y_test = train_test_split(
    X_lgb, y, test_size=0.2, random_state=42, stratify=y
)

X_train_nb, X_test_nb, _, _ = train_test_split(
    X_nb_df, y, test_size=0.2, random_state=42, stratify=y
)



# ===========================================================
# 9. TRAIN MODELS
# ===========================================================

print("\nTraining LightGBM...")
lgb_model = LGBMClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=120,
    random_state=42
)
lgb_model.fit(X_train_lgb, y_train)

y_lgb = lgb_model.predict(X_test_lgb)
y_lgb_proba = lgb_model.predict_proba(X_test_lgb)[:, 1]


print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train)

y_nb = nb_model.predict(X_test_nb)
y_nb_proba = nb_model.predict_proba(X_test_nb)[:, 1]



# ===========================================================
# 10. ENSEMBLE PREDICTION
# ===========================================================

ensemble_proba = 0.7 * y_lgb_proba + 0.3 * y_nb_proba
ensemble_pred = (ensemble_proba >= 0.5).astype(int)



# ===========================================================
# 11. MODEL METRICS
# ===========================================================

print("\nLightGBM Report:\n", classification_report(y_test, y_lgb))
print("AUC:", roc_auc_score(y_test, y_lgb_proba))

print("\nNaive Bayes Report:\n", classification_report(y_test, y_nb))
print("AUC:", roc_auc_score(y_test, y_nb_proba))

print("\nEnsemble Report:\n", classification_report(y_test, ensemble_pred))
print("AUC:", roc_auc_score(y_test, ensemble_proba))



# ===========================================================
# 12. SAVE MODELS + VECTORIZERS
# ===========================================================

os.makedirs("model", exist_ok=True)

joblib.dump(lgb_model, "model/lgb_model.pkl")
joblib.dump(nb_model, "model/nb_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")
joblib.dump(count_vec, "model/countvec.pkl")
joblib.dump(X_lgb.columns.tolist(), "model/lgb_features.pkl")

print("\nModels saved successfully!")
