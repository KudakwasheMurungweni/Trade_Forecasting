from src.data_loader import load_final_dataset
from src.preprocess import clean, get_feature_cols, get_target_col, scale, train_test_split_temporal
import numpy as np

df = load_final_dataset("import")
df = clean(df)
train, test = train_test_split_temporal(df, test_months=24)

feat_cols  = get_feature_cols()
target_col = get_target_col()

X_tr, y_tr, sx, sy = scale(train, feat_cols, target_col, fit=True)
X_te, y_te, _,  _  = scale(test,  feat_cols, target_col, sx, sy, fit=False)

print("Train X:", X_tr.shape, "| Train y:", y_tr.shape)
print("Test  X:", X_te.shape, "| Test  y:", y_te.shape)
print("Pipeline OK")
