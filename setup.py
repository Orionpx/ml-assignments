import os
import pandas as pd
import numpy as np
import csv

base_dir = r"d:\Orion\aimlpbl\assignments"

# Assignments data
assignments = {}

assignments[1] = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# 1. Load and inspect the data
df = pd.read_csv("advertising.csv")
print(df.head())
print(df.info())
print(df.describe())

# Independent variables: TV, Radio, Newspaper | Dependent variable: Sales
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 2. Correlation matrix and heatmap
corr = df.corr()
print(corr)
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="gray_r", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 3. Scatter plot of each independent variable against Sales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, X.columns):
    ax.scatter(df[col], y, color="black", s=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Sales")
plt.tight_layout()
plt.show()

# 4. Pearson correlation and p-value (hypothesis test, H0: r = 0)
for col in X.columns:
    r, p = stats.pearsonr(df[col], y)
    print(f"{col}: r = {r:.3f}, p-value = {p:.4g}")

# 5. Simple linear regression for each predictor
for col in X.columns:
    res = stats.linregress(df[col], y)
    print(f"{col}: slope = {res.slope:.4f}, intercept = {res.intercept:.4f}, R2 = {res.rvalue**2:.4f}")

# 6. Multiple linear regression (all predictors together)
model = LinearRegression().fit(X, y)
print("Intercept:", round(model.intercept_, 4))
for name, coef in zip(X.columns, model.coef_):
    print(f"Coefficient of {name}: {coef:.4f}")
print("R-squared:", round(model.score(X, y), 4))

# 7. Check multicollinearity among independent variables
print(X.corr())
"""

assignments[2] = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load data and select the features used for clustering
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
X_scaled = StandardScaler().fit_transform(X)

# 2. Elbow method and silhouette score to choose K
wcss, sil = [], []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(K, wcss, "ko-")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of clusters K")
ax[0].set_ylabel("WCSS")
ax[1].plot(K, sil, "ko-")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("Number of clusters K")
ax[1].set_ylabel("Score")
plt.show()

# 3. Final K-Means model (K = 5)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
print("Silhouette score:", silhouette_score(X_scaled, df["KMeans_Cluster"]))
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=df["KMeans_Cluster"], cmap="tab10", s=40)
centers = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200, label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering (K = 5)")
plt.legend()
plt.show()

# 4. Cluster profile
print(df.groupby("KMeans_Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())

# 5. Hierarchical (agglomerative) clustering with Ward linkage
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()

agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
df["Hier_Cluster"] = agg.fit_predict(X_scaled)
print("Hierarchical silhouette:", silhouette_score(X_scaled, df["Hier_Cluster"]))
print(df["Hier_Cluster"].value_counts())
"""

assignments[3] = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 1. Load data and standardise
data = load_wine()
X, y = data.data, data.target
print("Original shape:", X.shape)
X_std = StandardScaler().fit_transform(X)

# 2. PCA from scratch using eigen decomposition
cov = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eigh(cov)
order = np.argsort(eig_vals)[::-1]
eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]
print("Explained variance ratio (manual):")
print(np.round(eig_vals / eig_vals.sum(), 4))

# 3. PCA using scikit-learn
pca = PCA().fit(X_std)
ratio = pca.explained_variance_ratio_
cum = np.cumsum(ratio)
print("Cumulative variance:", np.round(cum, 4))
n95 = np.argmax(cum >= 0.95) + 1
print("Components needed for 95% variance:", n95)

# 4. Scree plot and cumulative variance plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(1, 14), ratio, color="gray", edgecolor="black")
ax[0].set_title("Scree Plot")
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("Explained variance ratio")
ax[1].plot(range(1, 14), cum, "ko-")
ax[1].axhline(0.95, color="black", linestyle="--")
ax[1].set_title("Cumulative Explained Variance")
ax[1].set_xlabel("Number of components")
plt.show()

# 5. Reduce to 2 dimensions and visualise
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_std)
plt.figure(figsize=(7, 5))
for cls, marker in zip(range(3), ["o", "s", "^"]):
    plt.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], marker=marker, label=data.target_names[cls])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine dataset projected on 2 principal components")
plt.legend()
plt.show()

# 6. Feature loadings of the first two components
loadings = pd.DataFrame(pca2.components_.T, index=data.feature_names, columns=["PC1", "PC2"])
print(loadings.round(3))

# 7. Effect of PCA on classification accuracy
clf = LogisticRegression(max_iter=1000)
acc_full = cross_val_score(clf, X_std, y, cv=5).mean()
X_n95 = PCA(n_components=n95).fit_transform(X_std)
acc_pca = cross_val_score(clf, X_n95, y, cv=5).mean()
print("Accuracy with 13 features :", round(acc_full, 4))
print(f"Accuracy with {n95} components:", round(acc_pca, 4))
"""

assignments[4] = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target
print("Shape:", X.shape)
print("Classes:", data.target_names)

# 2. Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
print("Class priors:", nb.class_prior_)

# 4. Predict and evaluate
y_pred = nb.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap="Greys")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.show()

# 6. Posterior probabilities for the first 5 test samples
print(np.round(nb.predict_proba(X_test[:5]), 4))

# 7. 10-fold cross validation
scores = cross_val_score(GaussianNB(), X, y, cv=10)
print("CV accuracy:", round(scores.mean(), 4), "+/-", round(scores.std(), 4))

# 8. Multinomial Naive Bayes on a small text example
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["win money now", "free prize win", "meeting at noon", "project report attached", "claim free money", "lunch tomorrow"]
labels = [1, 1, 0, 0, 1, 0] # 1 = spam, 0 = not spam
vec = CountVectorizer()
mnb = MultinomialNB(alpha=1.0).fit(vec.fit_transform(texts), labels)
test = vec.transform(["free money prize", "report for the meeting"])
print("Spam prediction:", mnb.predict(test))
"""

assignments[5] = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target
print("Features:", iris.feature_names)
print("Classes :", iris.target_names)

# 2. Split and scale the features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Find the best K using 5-fold cross validation
k_values = range(1, 26)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores.append(cross_val_score(knn, X_train_s, y_train, cv=5).mean())

best_k = k_values[int(np.argmax(cv_scores))]
print("Best K:", best_k, "with CV accuracy:", round(max(cv_scores), 4))

plt.figure(figsize=(7, 4))
plt.plot(k_values, cv_scores, "ko-")
plt.xlabel("Value of K")
plt.ylabel("Cross-validated accuracy")
plt.title("Selecting K for KNN")
plt.grid(True)
plt.show()

# 4. Train the final model with the best K
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

# 5. Evaluation
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap="Greys")
plt.title("Confusion Matrix - KNN")
plt.show()

# 6. Predict the class of a new flower
new_sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
pred = knn.predict(new_sample)
print("Predicted class:", iris.target_names[pred[0]])
print("Class probabilities:", knn.predict_proba(new_sample))

# 7. Compare distance metrics
for metric in ["euclidean", "manhattan", "chebyshev"]:
    model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    model.fit(X_train_s, y_train)
    print(metric, "accuracy:", round(model.score(X_test_s, y_test), 4))
"""

assignments[6] = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Generate data: y = 4 + 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Linear regression using the Normal Equation
Xb_train = np.c_[np.ones((len(X_train), 1)), X_train]
Xb_test = np.c_[np.ones((len(X_test), 1)), X_test]
theta_ne = np.linalg.inv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train
print("Normal equation : intercept = %.4f, slope = %.4f" % (theta_ne[0, 0], theta_ne[1, 0]))

# 3. Linear regression using Gradient Descent (implemented from scratch)
def gradient_descent(Xb, y, lr=0.1, epochs=1000):
    m = len(y)
    theta = np.zeros((Xb.shape[1], 1))
    cost_history = []
    for _ in range(epochs):
        error = Xb @ theta - y
        gradient = (1 / m) * Xb.T @ error
        theta = theta - lr * gradient
        cost_history.append((1 / (2 * m)) * np.sum(error ** 2))
    return theta, cost_history

theta_gd, costs = gradient_descent(Xb_train, y_train)
print("Gradient descent : intercept = %.4f, slope = %.4f" % (theta_gd[0, 0], theta_gd[1, 0]))

# 4. Linear regression using scikit-learn
model = LinearRegression().fit(X_train, y_train)
print("Scikit-learn : intercept = %.4f, slope = %.4f" % (model.intercept_[0], model.coef_[0, 0]))

# 5. Predictions and evaluation on the test set
y_pred = Xb_test @ theta_gd
print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE :", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R2 :", round(r2_score(y_test, y_pred), 4))

# 6. Plot the regression line and the cost curve
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(X_train, y_train, color="black", s=15, label="Train")
ax[0].scatter(X_test, y_test, color="gray", s=25, label="Test")
line = np.array([[0], [2]])
ax[0].plot(line, theta_gd[0] + theta_gd[1] * line, "k-", label="Fitted line")
ax[0].set_xlabel("X")
ax[0].set_ylabel("y")
ax[0].set_title("Linear Regression Fit")
ax[0].legend()
ax[1].plot(costs, "k-")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Cost J")
ax[1].set_title("Gradient Descent Convergence")
plt.show()

# 7. Predict a new value
x_new = 1.5
print("Prediction for x = 1.5:", round(float(theta_gd[0, 0] + theta_gd[1, 0] * x_new), 4))
"""

assignments[7] = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Generate non-linear data
np.random.seed(1)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * np.random.randn(200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Train SVR with three different kernels
kernels = {
    "Linear": SVR(kernel="linear", C=100, epsilon=0.1),
    "Polynomial": SVR(kernel="poly", C=100, degree=3, epsilon=0.1),
    "RBF": SVR(kernel="rbf", C=100, gamma=0.5, epsilon=0.1),
}

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="gray", s=15, label="Training data")
for name, svr in kernels.items():
    svr.fit(X_train, y_train)
    pred = svr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:10s} RMSE = {rmse:.4f} R2 = {r2_score(y_test, pred):.4f} Support vectors = {len(svr.support_)}")
    plt.plot(X, svr.predict(X), label=name + " kernel")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Support Vector Regression with different kernels")
plt.legend()
plt.show()

# 3. Hyperparameter tuning with GridSearchCV (scaled inputs)
pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
param_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__gamma": [0.01, 0.1, 1],
    "svr__epsilon": [0.01, 0.1, 0.2],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)

# 4. Evaluate the best model
best = grid.best_estimator_
y_pred = best.predict(X_test)
print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE :", round(mean_squared_error(y_test, y_pred), 4))
print("R2 :", round(r2_score(y_test, y_pred), 4))

# 5. Plot the tuned model with the epsilon tube
eps = best.named_steps["svr"].epsilon
y_line = best.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="gray", s=15, label="Training data")
plt.plot(X, y_line, "k-", label="Tuned SVR (RBF)")
plt.plot(X, y_line + eps, "k--", label="Epsilon tube")
plt.plot(X, y_line - eps, "k--")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Tuned SVR with epsilon-insensitive tube")
plt.legend()
plt.show()
"""

assignments[8] = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create a small noisy dataset
np.random.seed(0)
X = np.sort(np.random.rand(40, 1) * 6, axis=0)
y = np.sin(X).ravel() + 0.3 * np.random.randn(40)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 2. High degree polynomial models: without and with regularization
degree = 12

def make_model(reg):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)

models = {
    "No regularization": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.01, max_iter=100000),
    "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000),
}

# 3. Train and compare train / test performance
X_plot = np.linspace(0, 6, 300).reshape(-1, 1)
plt.figure(figsize=(9, 6))
plt.scatter(X_train, y_train, color="black", label="Train data")
plt.scatter(X_test, y_test, color="gray", marker="s", label="Test data")
print(f"{'Model':20s}{'Train R2':>10s}{'Test R2':>10s}{'Test MSE':>10s}")
for name, reg in models.items():
    pipe = make_model(reg).fit(X_train, y_train)
    tr = r2_score(y_train, pipe.predict(X_train))
    te = r2_score(y_test, pipe.predict(X_test))
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"{name:20s}{tr:10.3f}{te:10.3f}{mse:10.3f}")
    plt.plot(X_plot, pipe.predict(X_plot), label=name)

plt.ylim(-2.5, 2.5)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Effect of regularization on a degree-12 polynomial")
plt.legend()
plt.show()

# 4. Compare the size of the coefficients
for name in ["No regularization", "Ridge (L2)", "Lasso (L1)"]:
    pipe = make_model(models[name]).fit(X_train, y_train)
    coef = pipe[-1].coef_
    print(f"{name:20s} max |coef| = {np.abs(coef).max():10.3f} zero coefficients = {(np.abs(coef) < 1e-8).sum()}")

# 5. Choose the best alpha using cross-validation
ridge_cv = make_model(RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(X_train, y_train)
lasso_cv = make_model(LassoCV(alphas=np.logspace(-4, 0, 50), cv=5, max_iter=100000)).fit(X_train, y_train)
print("Best Ridge alpha:", ridge_cv[-1].alpha_)
print("Best Lasso alpha:", lasso_cv[-1].alpha_)
print("Ridge CV test R2:", round(r2_score(y_test, ridge_cv.predict(X_test)), 3))
print("Lasso CV test R2:", round(r2_score(y_test, lasso_cv.predict(X_test)), 3))
"""

assignments[9] = """import cv2
from ultralytics import YOLO

# 1. Load the pretrained YOLOv8 model (trained on COCO dataset)
model = YOLO("yolov8n.pt")
targets = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]

# 2. Open the CCTV video and prepare the output writer
cap = cv2.VideoCapture("cctv.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
out = cv2.VideoWriter("cctv_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
frame_no = 0
total_counts = {}

# 3. Process the video frame by frame
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    results = model(frame, conf=0.4, verbose=False)[0]
    frame_counts = {}
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        if label not in targets:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame_counts[label] = frame_counts.get(label, 0) + 1
        
    # keep the maximum count seen per class and show counts on the frame
    for k, v in frame_counts.items():
        total_counts[k] = max(total_counts.get(k, 0), v)
    text = ", ".join(f"{k}: {v}" for k, v in frame_counts.items())
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow("CCTV Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 4. Release resources and print summary
cap.release()
out.release()
cv2.destroyAllWindows()
print("Frames processed:", frame_no)
print("Maximum objects seen in one frame:", total_counts)

# 5. Detection on a single image (optional)
res = model("cctv_frame.jpg", conf=0.4)
res[0].save("cctv_frame_detected.jpg")
"""

assignments[10] = """import os
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load a sample of the multimodal dataset
df = pd.read_csv("multimodal_train.tsv", sep="\\t")
df = df.dropna(subset=["clean_title", "image_url"])
# use small sample if less than 2000
df = df.sample(min(2000, len(df)), random_state=42).reset_index(drop=True)
os.makedirs("images", exist_ok=True)

def download(row):
    path = f"images/{row.id}.jpg"
    try:
        if not os.path.exists(path):
            r = requests.get(row.image_url, timeout=5)
            with open(path, "wb") as f:
                f.write(r.content)
            Image.open(path).convert("RGB")
        return path
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        return None

df["path"] = [download(r) for r in df.itertuples()]
df = df.dropna(subset=["path"]).reset_index(drop=True)

if len(df) == 0:
    print("No valid images downloaded. Exiting.")
    exit()

y = df["2_way_label"].values
print("Samples with valid image:", len(df), "| label counts:", np.bincount(y))

# 2. Text features using DistilBERT
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

def text_features(texts, bs=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            cls = bert(**enc).last_hidden_state[:, 0, :]
            feats.append(cls.cpu().numpy())
    return np.vstack(feats)

# 3. Image features using ResNet18 (last layer removed)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()
prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def image_features(paths, bs=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(device)
            feats.append(resnet(batch).cpu().numpy())
    return np.vstack(feats)

X_text = text_features(df["clean_title"].tolist())
X_img = image_features(df["path"].tolist())

# 4. Early fusion: concatenate text and image features
X_fused = np.hstack([X_text, X_img])
print("Text:", X_text.shape, "Image:", X_img.shape, "Fused:", X_fused.shape)

# 5. Train and compare text-only, image-only and multimodal classifiers
idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)

def run(X, name):
    sc = StandardScaler().fit(X[idx_train])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(X[idx_train]), y[idx_train])
    pred = clf.predict(sc.transform(X[idx_test]))
    print(f"{name:12s} Accuracy = {accuracy_score(y[idx_test], pred):.4f} F1 = {f1_score(y[idx_test], pred):.4f}")
    return pred

run(X_text, "Text only")
run(X_img, "Image only")
pred = run(X_fused, "Multimodal")

# 6. Detailed report of the multimodal model (see README for label meaning)
print(classification_report(y[idx_test], pred, target_names=["Label 0", "Label 1"]))
"""

# Create directories and files
for i in range(1, 11):
    dir_name = os.path.join(base_dir, f"Assignment_{i}")
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, f"assignment_{i}.py"), "w", encoding="utf-8") as f:
        f.write(assignments[i])
        
# Create datasets

# Assignment 1 - advertising.csv
with open(os.path.join(base_dir, "Assignment_1", "advertising.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["TV", "Radio", "Newspaper", "Sales"])
    writer.writerows([
        [230.1, 37.8, 69.2, 22.1],
        [44.5, 39.3, 45.1, 10.4],
        [17.2, 45.9, 69.3, 9.3],
        [151.5, 41.3, 58.5, 18.5],
        [180.8, 10.8, 58.4, 12.9],
    ])

# Assignment 2 - Mall_Customers.csv
with open(os.path.join(base_dir, "Assignment_2", "Mall_Customers.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])
    writer.writerows([
        [1, "Male", 19, 15, 39],
        [2, "Male", 21, 15, 81],
        [3, "Female", 20, 16, 6],
        [4, "Female", 23, 16, 77],
        [5, "Female", 31, 17, 40],
    ])

# Assignment 9 - cctv.mp4 and cctv_frame.jpg
open(os.path.join(base_dir, "Assignment_9", "cctv_frame.jpg"), "w").close()
open(os.path.join(base_dir, "Assignment_9", "cctv.mp4"), "w").close()

# Assignment 10 - multimodal_train.tsv
with open(os.path.join(base_dir, "Assignment_10", "multimodal_train.tsv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["id", "clean_title", "image_url", "2_way_label"])
    writer.writerows([
        ["dummy1", "dummy fake title", "https://via.placeholder.com/150", 1],
        ["dummy2", "dummy real title", "https://via.placeholder.com/150", 0],
    ])

print("All assignments setup successfully.")
