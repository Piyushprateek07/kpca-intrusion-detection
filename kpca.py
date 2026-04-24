import numpy as np
import pandas as pd
import os, warnings, gc
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================
BASE_PATH = "/Volumes/SamsungT7/Desktop/Datasets"

DATASETS = {
    "CICIDS": "CICIDS_MASTER.csv",
    "UNSW": "UNSW_MASTER.csv"
}

OUT_BASE = "/Volumes/SamsungT7/Desktop/FINAL_RESULTS"
os.makedirs(OUT_BASE, exist_ok=True)

TEST_SIZE = 0.4
RANDOM_STATE = 42
KPCA_COMPONENTS = 30
SUBSET_SIZE = 25000

FP_COST = 5
FN_COST = 10

KERNELS = ["linear", "rbf", "poly", "cosine", "sigmoid"]

# ============================================================
# LOAD DATA
# ============================================================
def load_dataset(path, name):

    print(f"\n📦 Loading {name}...")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"{name} → Label missing")

    df["Label"] = df["Label"].astype(int)

    y = df["Label"].copy()
    df = df.drop(columns=["Label"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df = df.select_dtypes(include=[np.number])
    df["Label"] = y.loc[df.index]

    print("Shape:", df.shape)
    print(df["Label"].value_counts())

    return df

# ============================================================
# REDUCE CICIDS
# ============================================================
def reduce_cicids(df, size=60000):

    df_attack = df[df["Label"] == 1]
    df_benign = df[df["Label"] == 0]

    df = pd.concat([
        df_attack.sample(size//2, random_state=42),
        df_benign.sample(size//2, random_state=42)
    ])

    return df.sample(frac=1, random_state=42)

# ============================================================
# PIPELINE
# ============================================================
def run_pipeline(df, dataset_name):

    print(f"\n🚀 Running pipeline for {dataset_name}")

    OUT_DIR = os.path.join(OUT_BASE, dataset_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    X = df.drop(columns=["Label"]).values.astype(np.float32)
    y = df["Label"].values

    # SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        stratify=y_temp, random_state=RANDOM_STATE
    )

    # PREPROCESS
    imputer = SimpleImputer()
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    vt = VarianceThreshold(0.0005)
    X_train = vt.fit_transform(X_train)
    X_val   = vt.transform(X_val)
    X_test  = vt.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # MODELS
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=14, class_weight="balanced", n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05),
        "AdaBoost": AdaBoostClassifier(n_estimators=120),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=15, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }

    results = []
    final_probs = None
    final_threshold = None

    # ============================================================
    # KPCA LOOP
    # ============================================================
    for kernel in KERNELS:

        print(f"\nKernel: {kernel}")

        kpca = KernelPCA(
            n_components=KPCA_COMPONENTS,
            kernel=kernel,
            gamma=1 / X_train.shape[1]
        )

        idx = np.random.RandomState(42).choice(
            len(X_train),
            min(SUBSET_SIZE, len(X_train)),
            replace=False
        )

        kpca.fit(X_train[idx])

        Xtr = kpca.transform(X_train)
        Xv  = kpca.transform(X_val)
        Xt  = kpca.transform(X_test)

        for name, model in models.items():

            model.fit(Xtr, y_train)

            probs_val = model.predict_proba(Xv)[:,1]

            best_t, best_score = 0.5, -1

            for t in np.linspace(0.2, 0.8, 50):
                preds = (probs_val >= t).astype(int)
                mcc = matthews_corrcoef(y_val, preds)
                score = mcc - abs(t - 0.5)*0.05

                if score > best_score:
                    best_score, best_t = score, t

            probs = model.predict_proba(Xt)[:,1]
            preds = (probs >= best_t).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            cost = FP_COST*fp + FN_COST*fn

            results.append([
                kernel, name,
                accuracy_score(y_test, preds),
                f1_score(y_test, preds),
                roc_auc_score(y_test, probs),
                matthews_corrcoef(y_test, preds),
                cost
            ])

            final_probs = probs
            final_threshold = best_t

        gc.collect()

    # ============================================================
    # RESULTS
    # ============================================================
    df_res = pd.DataFrame(results, columns=[
        "kernel","model","accuracy","f1","auc","mcc","cost"
    ])

    df_res["score"] = (
        0.7*df_res["mcc"] + 0.2*df_res["auc"] + 0.1*df_res["f1"]
    )

    best = df_res.sort_values("score", ascending=False).iloc[0]

    print("\n🏆 BEST MODEL:")
    print(best)

    df_res.to_csv(f"{OUT_DIR}/results.csv", index=False)
    pd.DataFrame([best]).to_csv(f"{OUT_DIR}/best_model.csv", index=False)

    with open(f"{OUT_DIR}/threshold.txt", "w") as f:
        f.write(str(final_threshold))

    # ============================================================
    # DSI
    # ============================================================
    pca_cluster = PCA(n_components=10)
    Xc = pca_cluster.fit_transform(X_test)

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(Xc)

    cluster_mcc = []

    for cid in np.unique(clusters):

        idx = clusters == cid
        if idx.sum() < 30:
            continue

        y_c = y_test[idx]
        p_c = final_probs[idx]
        pred_c = (p_c >= final_threshold).astype(int)

        if len(np.unique(y_c)) < 2:
            continue

        cluster_mcc.append(matthews_corrcoef(y_c, pred_c))

    dsi = 1 - np.std(cluster_mcc) if len(cluster_mcc) > 1 else 0

    print("\nDSI:", dsi)

    with open(f"{OUT_DIR}/dsi.txt", "w") as f:
        f.write(str(dsi))

# ============================================================
# RUN
# ============================================================
for name, file in DATASETS.items():

    df = load_dataset(os.path.join(BASE_PATH, file), name)

    if name == "CICIDS":
        df = reduce_cicids(df)

    run_pipeline(df, name)

print("\n🎯 ALL DATASETS COMPLETED")