import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from src.data_prep import load_csv, build_feature_table

def train(
    raw_csv="data/raw/dataset.csv",
    smiles_col="smiles",
    label_col="label",
    use_morgan=False,
    random_state=42
):
    df = load_csv(raw_csv, smiles_col, label_col)
    feat_df = build_feature_table(df, use_morgan=use_morgan)
    X = feat_df.drop(columns=["label"])
    y = feat_df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    return clf, X.columns.tolist()

if __name__ == "__main__":
    train()  # expects data/raw/dataset.csv with columns: smiles,label
