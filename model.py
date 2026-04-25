
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

#to load dataset
CSV_PATH = "jamming-merged-gps-only.csv"  

df = pd.read_csv(CSV_PATH)
print(f"Dataset shape : {df.shape}")
print(f"Label counts  :\n{df['label'].value_counts()}\n")


# Drop non-feature columns (labl is more important )
DROP_COLS = ["timestamp", "time_utc_usec", "label"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df["label"].values

# Encode label to binary 
le = LabelEncoder()
y = le.fit_transform(y_raw)          # {benign:0, malicious:1}
print(f"Classes : {le.classes_}  →  {list(range(len(le.classes_)))}")

# Normalising features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


SEQ_LEN = 20      # winsow siwe 

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])   # label at the END of the window
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X_scaled, y, SEQ_LEN)
print(f"Sequence shape : {X_seq.shape}   Labels shape : {y_seq.shape}")


# 4. split 

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_seq, y_seq, test_size=0.20, random_state=42, stratify=y_seq)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"Train : {X_train.shape[0]}  Val : {X_val.shape[0]}  Test : {X_test.shape[0]}\n")

# 5.  imalanece

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw))
print(f"Class weights : {class_weights}\n")


# the model

n_features = X_seq.shape[2]

def build_model(seq_len, n_features):
    model = Sequential([
        #  Layer 1
        Bidirectional(
            LSTM(128, return_sequences=True),
            input_shape=(seq_len, n_features)
        ),
        BatchNormalization(),
        Dropout(0.3),

        # Layer 2
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        # Layer 3
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        # dense 
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")    # cuw the imput is binary
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model

model = build_model(SEQ_LEN, n_features)
model.summary()


# 7.  CALLBACKS

callbacks = [
    EarlyStopping(monitor="val_auc", patience=10,
                  restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1)
]

#train
EPOCHS     = 60
BATCH_SIZE = 64

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

#evaluatio,n
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\n── Test Set Evaluation ──")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")

#graph 
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("GPS Jamming Detection — LSTM Results", fontsize=15, fontweight="bold")

# — Training curves (loss) —
axes[0, 0].plot(history.history["loss"],     label="Train Loss")
axes[0, 0].plot(history.history["val_loss"], label="Val Loss")
axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].set_xlabel("Epoch")

# — Training curves (accuracy) —
axes[0, 1].plot(history.history["accuracy"],     label="Train Acc")
axes[0, 1].plot(history.history["val_accuracy"], label="Val Acc")
axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].set_xlabel("Epoch")

# — Training curves (AUC) —
axes[0, 2].plot(history.history["auc"],     label="Train AUC")
axes[0, 2].plot(history.history["val_auc"], label="Val AUC")
axes[0, 2].set_title("AUC"); axes[0, 2].legend(); axes[0, 2].set_xlabel("Epoch")

# — Confusion matrix —
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=axes[1, 0], colorbar=False)
axes[1, 0].set_title("Confusion Matrix")

# — ROC curve —
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score    = roc_auc_score(y_test, y_prob)
axes[1, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="darkorange")
axes[1, 1].plot([0, 1], [0, 1], "k--")
axes[1, 1].set_title("ROC Curve")
axes[1, 1].set_xlabel("FPR"); axes[1, 1].set_ylabel("TPR")
axes[1, 1].legend()

# — Prediction probability distribution —
axes[1, 2].hist(y_prob[y_test == 0], bins=40, alpha=0.6, label="Benign",    color="steelblue")
axes[1, 2].hist(y_prob[y_test == 1], bins=40, alpha=0.6, label="Malicious", color="tomato")
axes[1, 2].axvline(0.5, color="black", linestyle="--", label="Threshold=0.5")
axes[1, 2].set_title("Predicted Probabilities")
axes[1, 2].set_xlabel("P(malicious)"); axes[1, 2].legend()

plt.tight_layout()
plt.savefig("gps_jamming_lstm_results.png", dpi=150)
plt.show()
print("\nPlot saved → gps_jamming_lstm_results.png")

#save
model.save("gps_jamming_lstm_model.keras")
print("Model saved  → gps_jamming_lstm_model.keras")

# ─────────────────────────────────────────────
#  INFERENCE HELPER
# ─────────────────────────────────────────────
def predict_jamming(raw_window: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Predict GPS jamming for a single window.

    Parameters
    ----------
    raw_window : np.ndarray, shape (SEQ_LEN, n_features)
        Raw (unscaled) feature window.
    threshold  : float
        Decision threshold (default 0.5).

    Returns
    -------
    dict with keys: probability, label, is_jammed
    """
    assert raw_window.shape == (SEQ_LEN, n_features), \
        f"Expected shape ({SEQ_LEN}, {n_features}), got {raw_window.shape}"

    scaled = scaler.transform(raw_window)
    prob   = float(model.predict(scaled[np.newaxis], verbose=0)[0][0])
    label  = "malicious" if prob >= threshold else "benign"
    return {"probability": prob, "label": label, "is_jammed": prob >= threshold}


# demo
demo_raw  = scaler.inverse_transform(X_test[0])
demo_pred = predict_jamming(demo_raw)
print(f"\nDemo prediction: {demo_pred}  (true label: {le.classes_[y_test[0]]})")