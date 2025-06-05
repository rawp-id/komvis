
# FITUR EKSTRAKSI UNTUK LOGISTIC REGRESSION (OPSIONAL)
feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)  # ambil sebelum Dense

# Ambil semua data dari val_gen untuk fitur
X_features = []
y_true = []

val_gen.reset()
for i in range(len(val_gen)):
    x_batch, y_batch = val_gen[i]
    features = feature_extractor.predict(x_batch)
    X_features.append(features)
    y_true.append(y_batch)
    if i >= len(val_gen) - 1:
        break

X_features = np.vstack(X_features)
y_true = np.argmax(np.vstack(y_true), axis=1)

# KLASIFIKASI DENGAN LOGISTIC REGRESSION
clf = LogisticRegression(max_iter=1000)
clf.fit(X_features, y_true)
y_pred = clf.predict(X_features)

# METRIK EVALUASI
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys()))
