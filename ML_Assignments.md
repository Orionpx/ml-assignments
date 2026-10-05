# Assignment - 1

**AIM:** Identify the association between dependent and independent variables.

## Theory

### 1\. Dependent and Independent Variables

In any data analysis problem, the variables are divided into two groups. An independent variable (also called a predictor, feature or input, usually written as X) is a variable whose value is chosen or observed and is used to explain or predict something else. A dependent variable (also called the response, target or output, usually written as Y) is the variable whose value is expected to change when the independent variables change. For example, in an advertising dataset the money spent on TV, Radio and Newspaper advertisements are independent variables, while Sales is the dependent variable.

Association means that the values of two variables tend to change together in some systematic way. It can be positive (both increase together), negative (one increases while the other decreases) or absent. It can also be linear or non-linear. Identifying this association is the first step of every supervised learning task, because it tells us which features are useful for prediction.

### 2\. Measures of Association

Covariance measures how two variables vary together. Its value depends on the units of the variables, so it is difficult to interpret directly. Pearson's correlation coefficient (r) standardises covariance and always lies between -1 and +1:

_r = Σ(xi - x̄)(yi - ȳ) / √\[ Σ(xi - x̄)² · Σ(yi - ȳ)² \]_

- r close to +1 indicates a strong positive linear association.
- r close to -1 indicates a strong negative linear association.
- r close to 0 indicates little or no linear association.
- Spearman's rank correlation is used when the relationship is monotonic but not linear, or when the data has outliers.

### 3\. Regression Coefficient and Hypothesis Test

Regression gives a quantitative description of the association. For one predictor the model is Y = β0 + β1X + ε, where β0 is the intercept, β1 is the regression coefficient (slope) and ε is the error term. The slope β1 tells us how much Y changes on average for a one-unit increase in X, keeping other variables constant. For several predictors the model becomes Y = β0 + β1X1 + β2X2 + ... + βpXp + ε.

To check whether the association is statistically significant, a hypothesis test is applied. The null hypothesis H0 states that β1 = 0 (no association), and the alternative hypothesis H1 states that β1 ≠ 0. If the p-value is less than the significance level (usually 0.05), H0 is rejected and we conclude that the association is significant. The coefficient of determination R² gives the fraction of the variation in Y that is explained by the independent variables.

### 4\. Visual Tools and Cautions

Scatter plots show the shape of the relationship between two variables, and a correlation heatmap summarises the pairwise correlations of all variables at once. Strong correlation between two independent variables is called multicollinearity and makes regression coefficients unstable. It is also important to remember that correlation does not imply causation: two variables may be correlated only because of a third hidden variable or by pure chance (spurious correlation).

## Code

\# Dataset: <https://www.kaggle.com/datasets/ashydv/advertising-dataset>

\# Download advertising.csv and keep it in the working directory

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.linear_model import LinearRegression

\# 1. Load and inspect the data

df = pd.read_csv("advertising.csv")

print(df.head())

print(df.info())

print(df.describe())

\# Independent variables: TV, Radio, Newspaper | Dependent variable: Sales

X = df\[\["TV", "Radio", "Newspaper"\]\]

y = df\["Sales"\]

\# 2. Correlation matrix and heatmap

corr = df.corr()

print(corr)

plt.figure(figsize=(6, 5))

sns.heatmap(corr, annot=True, cmap="gray_r", fmt=".2f")

plt.title("Correlation Matrix")

plt.show()

\# 3. Scatter plot of each independent variable against Sales

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col in zip(axes, X.columns):

ax.scatter(df\[col\], y, color="black", s=12)

ax.set_xlabel(col)

ax.set_ylabel("Sales")

plt.tight_layout()

plt.show()

\# 4. Pearson correlation and p-value (hypothesis test, H0: r = 0)

for col in X.columns:

r, p = stats.pearsonr(df\[col\], y)

print(f"{col}: r = {r:.3f}, p-value = {p:.4g}")

\# 5. Simple linear regression for each predictor

for col in X.columns:

res = stats.linregress(df\[col\], y)

print(f"{col}: slope = {res.slope:.4f}, "

f"intercept = {res.intercept:.4f}, R2 = {res.rvalue\*\*2:.4f}")

\# 6. Multiple linear regression (all predictors together)

model = LinearRegression().fit(X, y)

print("Intercept:", round(model.intercept_, 4))

for name, coef in zip(X.columns, model.coef_):

print(f"Coefficient of {name}: {coef:.4f}")

print("R-squared:", round(model.score(X, y), 4))

\# 7. Check multicollinearity among independent variables

print(X.corr())

## Output

## Conclusion

The association between the independent variables (TV, Radio, Newspaper) and the dependent variable (Sales) was identified using correlation, scatter plots and regression. TV and Radio showed a strong and significant association with Sales, while Newspaper showed a weak one. The regression coefficients and R² quantify how strongly each variable influences Sales.

# Assignment - 2

**AIM:** Apply the clustering techniques using unsupervised learning.

## Theory

### 1\. Unsupervised Learning and Clustering

In unsupervised learning the data has no target labels and the algorithm must discover hidden structure on its own. Clustering is the most common unsupervised task: it groups the data points into clusters so that points in the same cluster are highly similar to each other and points in different clusters are dissimilar. Typical applications are customer segmentation, document grouping, image compression, anomaly detection and gene expression analysis.

### 2\. K-Means Clustering

K-Means is a centroid-based algorithm that divides the data into K clusters, where K is chosen in advance. The steps are:

- Choose K and initialise K centroids (K-Means++ is used to pick good starting centroids).
- Assign every data point to the nearest centroid using Euclidean distance.
- Recompute each centroid as the mean of all points assigned to it.
- Repeat the assignment and update steps until the centroids stop changing or the maximum number of iterations is reached.

The algorithm minimises the Within-Cluster Sum of Squares (WCSS), also called inertia: WCSS = Σk Σx∈Ck ||x - μk||², where μk is the centroid of cluster Ck.

### 3\. Choosing the Number of Clusters

The Elbow method plots WCSS against K; the point where the curve bends like an elbow suggests a suitable K, because adding more clusters beyond it gives only a small reduction in WCSS. The Silhouette score measures how well a point fits its own cluster compared with the nearest other cluster: s = (b - a) / max(a, b), where a is the mean distance to points in its own cluster and b is the mean distance to points in the nearest neighbouring cluster. The score lies between -1 and +1 and a higher value means better defined clusters.

### 4\. Hierarchical Clustering

Agglomerative hierarchical clustering starts with every point as its own cluster and repeatedly merges the two closest clusters until only one cluster remains. The merging history is shown as a tree called a dendrogram, and cutting the dendrogram at a chosen height gives the final clusters. Ward linkage merges the clusters that cause the smallest increase in total variance. Unlike K-Means, it does not need K in advance, but it is slower for large datasets.

### 5\. Practical Points

Clustering is distance based, therefore features must be scaled (for example with StandardScaler) before the algorithm is applied. K-Means works best for compact, roughly spherical clusters of similar size and is sensitive to outliers and to the random initialisation.

## Code

\# Dataset: <https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python>

\# Download Mall_Customers.csv and keep it in the working directory

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

\# 1. Load data and select the features used for clustering

df = pd.read_csv("Mall_Customers.csv")

print(df.head())

X = df\[\["Annual Income (k\$)", "Spending Score (1-100)"\]\].values

X_scaled = StandardScaler().fit_transform(X)

\# 2. Elbow method and silhouette score to choose K

wcss, sil = \[\], \[\]

K = range(2, 11)

for k in K:

km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)

wcss.append(km.inertia_)

sil.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax\[0\].plot(K, wcss, "ko-")

ax\[0\].set_title("Elbow Method")

ax\[0\].set_xlabel("Number of clusters K")

ax\[0\].set_ylabel("WCSS")

ax\[1\].plot(K, sil, "ko-")

ax\[1\].set_title("Silhouette Score")

ax\[1\].set_xlabel("Number of clusters K")

ax\[1\].set_ylabel("Score")

plt.show()

\# 3. Final K-Means model (K = 5)

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)

df\["KMeans_Cluster"\] = kmeans.fit_predict(X_scaled)

print("Silhouette score:", silhouette_score(X_scaled, df\["KMeans_Cluster"\]))

plt.figure(figsize=(7, 5))

plt.scatter(X\[:, 0\], X\[:, 1\], c=df\["KMeans_Cluster"\], cmap="tab10", s=40)

centers = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)

plt.scatter(centers\[:, 0\], centers\[:, 1\], c="black", marker="X", s=200,

label="Centroids")

plt.xlabel("Annual Income (k\$)")

plt.ylabel("Spending Score (1-100)")

plt.title("K-Means Clustering (K = 5)")

plt.legend()

plt.show()

\# 4. Cluster profile

print(df.groupby("KMeans_Cluster")\[\["Age", "Annual Income (k\$)",

"Spending Score (1-100)"\]\].mean())

\# 5. Hierarchical (agglomerative) clustering with Ward linkage

Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))

dendrogram(Z, truncate_mode="lastp", p=30)

plt.title("Dendrogram (Ward linkage)")

plt.xlabel("Data points")

plt.ylabel("Distance")

plt.show()

agg = AgglomerativeClustering(n_clusters=5, linkage="ward")

df\["Hier_Cluster"\] = agg.fit_predict(X_scaled)

print("Hierarchical silhouette:", silhouette_score(X_scaled, df\["Hier_Cluster"\]))

print(df\["Hier_Cluster"\].value_counts())

## Output

## Conclusion

K-Means and hierarchical clustering were applied on the customer data without using any labels. The elbow method and silhouette score suggested five natural groups of customers based on income and spending score. Clustering therefore helps to discover hidden segments in unlabelled data.

# Assignment - 3

**AIM:** Apply dimensionality reduction using PCA.

## Theory

### 1\. Need for Dimensionality Reduction

Real-world datasets often contain tens or hundreds of features, many of which are correlated or carry little information. A large number of features increases computation time, makes visualisation impossible and leads to the curse of dimensionality, where the data becomes sparse and models overfit. Dimensionality reduction transforms the data into a smaller number of features while keeping as much useful information as possible. It can be done by feature selection (keeping a subset of original features) or feature extraction (creating new features, as PCA does).

### 2\. Principal Component Analysis (PCA)

PCA is an unsupervised linear technique that converts the original correlated features into a new set of uncorrelated features called principal components. The first principal component is the direction along which the data has maximum variance, the second is the direction of maximum remaining variance that is orthogonal to the first, and so on. The components are ordered by the amount of variance they capture.

### 3\. Steps of PCA

- Standardise the data so that each feature has zero mean and unit variance.
- Compute the covariance matrix of the standardised data.
- Find the eigenvalues and eigenvectors of the covariance matrix. Each eigenvector is a principal component direction and its eigenvalue is the variance along that direction.
- Sort the eigenvectors in descending order of eigenvalues and select the top k.
- Project the data onto the selected eigenvectors: Z = X · W, where W is the matrix of the k chosen eigenvectors.

### 4\. Explained Variance

The explained variance ratio of a component is its eigenvalue divided by the sum of all eigenvalues. A scree plot shows this ratio for every component, and the cumulative explained variance plot is used to decide how many components to keep, commonly enough components to retain 95 percent of the total variance. In practice PCA is computed through Singular Value Decomposition (SVD), which is what scikit-learn uses.

### 5\. Advantages and Limitations

PCA removes multicollinearity, reduces noise and storage, speeds up learning algorithms and allows the data to be plotted in two or three dimensions. However, the new components are linear combinations of the original features and are therefore harder to interpret, PCA captures only linear relationships, and it is sensitive to the scale of features.

## Code

\# Dataset: built into scikit-learn (Wine dataset, 178 samples, 13 features)

\# Also available at <https://archive.ics.uci.edu/dataset/109/wine>

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

\# 1. Load data and standardise

data = load_wine()

X, y = data.data, data.target

print("Original shape:", X.shape)

X_std = StandardScaler().fit_transform(X)

\# 2. PCA from scratch using eigen decomposition

cov = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eigh(cov)

order = np.argsort(eig_vals)\[::-1\]

eig_vals, eig_vecs = eig_vals\[order\], eig_vecs\[:, order\]

print("Explained variance ratio (manual):")

print(np.round(eig_vals / eig_vals.sum(), 4))

\# 3. PCA using scikit-learn

pca = PCA().fit(X_std)

ratio = pca.explained_variance_ratio_

cum = np.cumsum(ratio)

print("Cumulative variance:", np.round(cum, 4))

n95 = np.argmax(cum >= 0.95) + 1

print("Components needed for 95% variance:", n95)

\# 4. Scree plot and cumulative variance plot

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax\[0\].bar(range(1, 14), ratio, color="gray", edgecolor="black")

ax\[0\].set_title("Scree Plot")

ax\[0\].set_xlabel("Principal component")

ax\[0\].set_ylabel("Explained variance ratio")

ax\[1\].plot(range(1, 14), cum, "ko-")

ax\[1\].axhline(0.95, color="black", linestyle="--")

ax\[1\].set_title("Cumulative Explained Variance")

ax\[1\].set_xlabel("Number of components")

plt.show()

\# 5. Reduce to 2 dimensions and visualise

pca2 = PCA(n_components=2)

X_pca = pca2.fit_transform(X_std)

plt.figure(figsize=(7, 5))

for cls, marker in zip(range(3), \["o", "s", "^"\]):

plt.scatter(X_pca\[y == cls, 0\], X_pca\[y == cls, 1\], marker=marker,

label=data.target_names\[cls\])

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("Wine dataset projected on 2 principal components")

plt.legend()

plt.show()

\# 6. Feature loadings of the first two components

loadings = pd.DataFrame(pca2.components_.T, index=data.feature_names,

columns=\["PC1", "PC2"\])

print(loadings.round(3))

\# 7. Effect of PCA on classification accuracy

clf = LogisticRegression(max_iter=1000)

acc_full = cross_val_score(clf, X_std, y, cv=5).mean()

X_n95 = PCA(n_components=n95).fit_transform(X_std)

acc_pca = cross_val_score(clf, X_n95, y, cv=5).mean()

print("Accuracy with 13 features :", round(acc_full, 4))

print(f"Accuracy with {n95} components:", round(acc_pca, 4))

## Output

## Conclusion

PCA was applied to the 13-feature Wine dataset after standardisation. A small number of principal components captured most of the variance, and the data could be visualised clearly in two dimensions. Classification accuracy remained almost the same with fewer features, which shows that PCA reduces dimensions without losing much information.

# Assignment - 4

**AIM:** Apply Naïve Bayes classification algorithms.

## Theory

### 1\. Bayes' Theorem

Naïve Bayes is a family of probabilistic classifiers based on Bayes' theorem, which gives the probability of a class C given the observed features X:

_P(C | X) = P(X | C) · P(C) / P(X)_

Here P(C) is the prior probability of the class, P(X | C) is the likelihood of the features given the class, P(X) is the evidence and P(C | X) is the posterior probability.

### 2\. The Naïve Assumption

Computing P(X | C) for many features is difficult, so the algorithm makes the naïve assumption that all features are conditionally independent given the class. The likelihood then becomes a simple product: P(X | C) = P(x1 | C) · P(x2 | C) · ... · P(xn | C). Although this assumption is rarely true in real data, the classifier works surprisingly well in practice, especially for text classification and spam filtering.

### 3\. Decision Rule (MAP)

Since P(X) is the same for every class, the predicted class is the one with the maximum posterior probability (Maximum A Posteriori estimate): ŷ = argmax over C of P(C) · Π P(xi | C). In practice the logarithm of the probabilities is used to avoid numerical underflow.

### 4\. Types of Naïve Bayes

- Gaussian Naïve Bayes: for continuous features, assuming each feature follows a normal distribution, P(xi | C) = (1 / √(2πσ²)) · exp(-(xi - μ)² / 2σ²), where μ and σ are estimated from the training data of that class.
- Multinomial Naïve Bayes: for count data such as word frequencies in documents.
- Bernoulli Naïve Bayes: for binary features such as the presence or absence of a word.

### 5\. Smoothing and Evaluation

If a feature value never occurs with a class in the training data, its probability becomes zero and wipes out the whole product. Laplace (additive) smoothing adds a small count to every feature to avoid this. The classifier is evaluated with accuracy, precision = TP / (TP + FP), recall = TP / (TP + FN) and F1-score = 2 · precision · recall / (precision + recall), together with the confusion matrix.

### 6\. Advantages and Limitations

Naïve Bayes is very fast, needs little training data, handles many features and is easy to implement. Its limitations are the independence assumption and the zero-frequency problem, and its probability estimates are often not well calibrated.

## Code

\# Dataset: built into scikit-learn (Breast Cancer Wisconsin, 569 samples)

\# Also available at <https://archive.ics.uci.edu/dataset/17>

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (accuracy_score, precision_score, recall_score,

f1_score, confusion_matrix,

classification_report, ConfusionMatrixDisplay)

\# 1. Load data

data = load_breast_cancer()

X, y = data.data, data.target

print("Shape:", X.shape)

print("Classes:", data.target_names)

\# 2. Train-test split (80:20)

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.2, random_state=42, stratify=y)

\# 3. Train Gaussian Naive Bayes

nb = GaussianNB()

nb.fit(X_train, y_train)

print("Class priors:", nb.class_prior_)

\# 4. Predict and evaluate

y_pred = nb.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))

print("Precision:", round(precision_score(y_test, y_pred), 4))

print("Recall :", round(recall_score(y_test, y_pred), 4))

print("F1-score :", round(f1_score(y_test, y_pred), 4))

print(classification_report(y_test, y_pred, target_names=data.target_names))

\# 5. Confusion matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap="Greys")

plt.title("Confusion Matrix - Gaussian Naive Bayes")

plt.show()

\# 6. Posterior probabilities for the first 5 test samples

print(np.round(nb.predict_proba(X_test\[:5\]), 4))

\# 7. 10-fold cross validation

scores = cross_val_score(GaussianNB(), X, y, cv=10)

print("CV accuracy:", round(scores.mean(), 4), "+/-", round(scores.std(), 4))

\# 8. Multinomial Naive Bayes on a small text example

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

texts = \["win money now", "free prize win", "meeting at noon",

"project report attached", "claim free money", "lunch tomorrow"\]

labels = \[1, 1, 0, 0, 1, 0\] # 1 = spam, 0 = not spam

vec = CountVectorizer()

mnb = MultinomialNB(alpha=1.0).fit(vec.fit_transform(texts), labels)

test = vec.transform(\["free money prize", "report for the meeting"\])

print("Spam prediction:", mnb.predict(test))

## Output

## Conclusion

The Gaussian Naïve Bayes classifier was implemented on the Breast Cancer dataset and achieved high accuracy, precision and recall. A Multinomial Naïve Bayes example was also shown for text data. This proves that Naïve Bayes is a simple and fast classifier even though it assumes feature independence.

# Assignment - 5

**AIM:** Apply the KNN algorithm for the classification.

## Theory

### 1\. K-Nearest Neighbours (KNN)

KNN is a simple supervised algorithm that can be used for both classification and regression. It is called a lazy or instance-based learner because it does not build a model during training; it simply stores the training data and does all the work at prediction time. It is also non-parametric, as it makes no assumption about the distribution of the data.

### 2\. Working of the Algorithm

- Choose the number of neighbours K.
- For a new data point, calculate its distance from every point in the training set.
- Select the K training points that are closest to it.
- For classification, take a majority vote of the classes of these K neighbours and assign the winning class to the new point. For regression, the average of the neighbours' values is taken.

### 3\. Distance Metrics

The notion of closeness is defined by a distance metric. For two points x and y with n features:

- Euclidean distance: d = √\[ Σ (xi - yi)² \]
- Manhattan distance: d = Σ |xi - yi|
- Minkowski distance: d = ( Σ |xi - yi|^p )^(1/p), which becomes Manhattan for p = 1 and Euclidean for p = 2.

### 4\. Choosing K and Preprocessing

A very small K (for example 1) makes the model sensitive to noise and leads to overfitting (high variance), while a very large K smooths the decision boundary too much and leads to underfitting (high bias). The best K is normally found by cross-validation, and an odd K avoids ties in binary problems. Because KNN depends on distances, all features must be scaled to the same range using standardisation or min-max scaling. Weighted KNN gives closer neighbours a larger vote than distant ones.

### 5\. Advantages and Limitations

KNN is easy to understand, needs no training time and adapts naturally to new data. However, prediction is slow and memory-hungry for large datasets, the performance falls in high dimensional data because of the curse of dimensionality, and it is sensitive to irrelevant features and to unscaled data.

## Code

\# Dataset: built into scikit-learn (Iris dataset, 150 samples, 4 features)

\# Also available at <https://archive.ics.uci.edu/dataset/53/iris>

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, confusion_matrix,

classification_report, ConfusionMatrixDisplay)

\# 1. Load data

iris = load_iris()

X, y = iris.data, iris.target

print("Features:", iris.feature_names)

print("Classes :", iris.target_names)

\# 2. Split and scale the features

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)

X_test_s = scaler.transform(X_test)

\# 3. Find the best K using 5-fold cross validation

k_values = range(1, 26)

cv_scores = \[\]

for k in k_values:

knn = KNeighborsClassifier(n_neighbors=k)

cv_scores.append(cross_val_score(knn, X_train_s, y_train, cv=5).mean())

best_k = k_values\[int(np.argmax(cv_scores))\]

print("Best K:", best_k, "with CV accuracy:", round(max(cv_scores), 4))

plt.figure(figsize=(7, 4))

plt.plot(k_values, cv_scores, "ko-")

plt.xlabel("Value of K")

plt.ylabel("Cross-validated accuracy")

plt.title("Selecting K for KNN")

plt.grid(True)

plt.show()

\# 4. Train the final model with the best K

knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")

knn.fit(X_train_s, y_train)

y_pred = knn.predict(X_test_s)

\# 5. Evaluation

print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))

print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap="Greys")

plt.title("Confusion Matrix - KNN")

plt.show()

\# 6. Predict the class of a new flower

new_sample = scaler.transform(\[\[5.1, 3.5, 1.4, 0.2\]\])

pred = knn.predict(new_sample)

print("Predicted class:", iris.target_names\[pred\[0\]\])

print("Class probabilities:", knn.predict_proba(new_sample))

\# 7. Compare distance metrics

for metric in \["euclidean", "manhattan", "chebyshev"\]:

model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)

model.fit(X_train_s, y_train)

print(metric, "accuracy:", round(model.score(X_test_s, y_test), 4))

## Output

## Conclusion

The KNN algorithm was applied to classify the Iris flowers after scaling the features. The best value of K was selected using cross-validation and the model achieved high accuracy on the test data. KNN is simple and effective, but its result depends strongly on K and on feature scaling.

# Assignment - 6

**AIM:** Implement Linear Regression Algorithm.

## Theory

### 1\. Linear Regression

Linear regression is a supervised learning algorithm used to predict a continuous numerical value. It models the relationship between the dependent variable y and one or more independent variables x by fitting a straight line (or a hyperplane in higher dimensions) through the data. For one feature the hypothesis is h(x) = θ0 + θ1x, and for n features it is h(x) = θ0 + θ1x1 + θ2x2 + ... + θnxn, or in matrix form h(X) = Xθ.

### 2\. Cost Function

The parameters θ are chosen so that the difference between the predicted and the actual values is as small as possible. The most common cost function is the Mean Squared Error:

_J(θ) = (1 / 2m) Σ ( h(xi) - yi )²_

where m is the number of training examples. The method of minimising this cost is called Ordinary Least Squares (OLS).

### 3\. Methods of Finding the Parameters

- Normal equation: a closed-form solution, θ = (XᵀX)⁻¹ Xᵀ y. It needs no learning rate but becomes slow when the number of features is very large.
- Gradient descent: an iterative method that starts with random values of θ and updates them repeatedly in the direction of the negative gradient, θj := θj - α · ∂J/∂θj, where α is the learning rate. For linear regression this becomes θj := θj - (α / m) Σ ( h(xi) - yi ) · xij.

### 4\. Assumptions of Linear Regression

- Linearity: the relationship between X and y is linear.
- Independence: the observations (and the residuals) are independent of each other.
- Homoscedasticity: the residuals have constant variance.
- Normality: the residuals are approximately normally distributed.
- No multicollinearity: the independent variables are not highly correlated.

### 5\. Evaluation Metrics

The model is evaluated using Mean Absolute Error (MAE = (1/m) Σ |y - ŷ|), Mean Squared Error (MSE = (1/m) Σ (y - ŷ)²), Root Mean Squared Error (RMSE = √MSE) and the coefficient of determination R² = 1 - (SSres / SStot), which tells the proportion of variance in y explained by the model. An R² close to 1 indicates a good fit.

## Code

\# Dataset: synthetic data generated in the code (no download required)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

\# 1. Generate data: y = 4 + 3x + noise

np.random.seed(42)

X = 2 \* np.random.rand(100, 1)

y = 4 + 3 \* X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.2, random_state=42)

\# 2. Linear regression using the Normal Equation

Xb_train = np.c_\[np.ones((len(X_train), 1)), X_train\]

Xb_test = np.c_\[np.ones((len(X_test), 1)), X_test\]

theta_ne = np.linalg.inv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train

print("Normal equation : intercept = %.4f, slope = %.4f"

% (theta_ne\[0, 0\], theta_ne\[1, 0\]))

\# 3. Linear regression using Gradient Descent (implemented from scratch)

def gradient_descent(Xb, y, lr=0.1, epochs=1000):

m = len(y)

theta = np.zeros((Xb.shape\[1\], 1))

cost_history = \[\]

for _ in range(epochs):

error = Xb @ theta - y

gradient = (1 / m) \* Xb.T @ error

theta = theta - lr \* gradient

cost_history.append((1 / (2 \* m)) \* np.sum(error \*\* 2))

return theta, cost_history

theta_gd, costs = gradient_descent(Xb_train, y_train)

print("Gradient descent : intercept = %.4f, slope = %.4f"

% (theta_gd\[0, 0\], theta_gd\[1, 0\]))

\# 4. Linear regression using scikit-learn

model = LinearRegression().fit(X_train, y_train)

print("Scikit-learn : intercept = %.4f, slope = %.4f"

% (model.intercept_\[0\], model.coef_\[0, 0\]))

\# 5. Predictions and evaluation on the test set

y_pred = Xb_test @ theta_gd

print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))

print("MSE :", round(mean_squared_error(y_test, y_pred), 4))

print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

print("R2 :", round(r2_score(y_test, y_pred), 4))

\# 6. Plot the regression line and the cost curve

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax\[0\].scatter(X_train, y_train, color="black", s=15, label="Train")

ax\[0\].scatter(X_test, y_test, color="gray", s=25, label="Test")

line = np.array(\[\[0\], \[2\]\])

ax\[0\].plot(line, theta_gd\[0\] + theta_gd\[1\] \* line, "k-", label="Fitted line")

ax\[0\].set_xlabel("X")

ax\[0\].set_ylabel("y")

ax\[0\].set_title("Linear Regression Fit")

ax\[0\].legend()

ax\[1\].plot(costs, "k-")

ax\[1\].set_xlabel("Iterations")

ax\[1\].set_ylabel("Cost J")

ax\[1\].set_title("Gradient Descent Convergence")

plt.show()

\# 7. Predict a new value

x_new = 1.5

print("Prediction for x = 1.5:", round(float(theta_gd\[0, 0\] + theta_gd\[1, 0\] \* x_new), 4))

## Output

## Conclusion

Linear regression was implemented using the normal equation, gradient descent from scratch and scikit-learn. All three methods gave almost the same intercept and slope, close to the true values 4 and 3. The low error values and high R² show that the model fits the data well.

# Assignment - 7

**AIM:** Implement SVM Algorithm for Regression.

## Theory

### 1\. Support Vector Regression (SVR)

Support Vector Machine is mostly known as a classifier, but the same idea can be used for regression and is then called Support Vector Regression. Instead of finding a hyperplane that separates classes with the maximum margin, SVR finds a function f(x) that deviates from the actual targets by at most a value ε for all training points and, at the same time, is as flat as possible.

### 2\. ε-Insensitive Loss and the Tube

SVR uses the ε-insensitive loss function: L(y, f(x)) = 0 if |y - f(x)| ≤ ε, otherwise |y - f(x)| - ε. Errors smaller than ε are ignored, so a tube of width 2ε is formed around the regression function. Only the points that lie on or outside the tube become support vectors and influence the model. The optimisation problem is:

_minimise (1/2)||w||² + C Σ (ξi + ξi\*)_

_subject to |yi - (w · xi + b)| ≤ ε + ξi, where ξi and ξi\* are slack variables that allow some points to fall outside the tube._

### 3\. Important Hyperparameters

- C: the regularisation parameter. A large C penalises points outside the tube heavily (low bias, high variance), and a small C gives a smoother and simpler function.
- ε (epsilon): the width of the insensitive tube. A larger ε gives fewer support vectors and a simpler model.
- kernel: the function that maps the data into a higher dimensional space so that non-linear relationships can be learnt.
- γ (gamma): the kernel coefficient for RBF and polynomial kernels, which controls how far the influence of a single training point reaches.

### 4\. Kernel Functions

- Linear kernel: K(x, x') = x · x'
- Polynomial kernel: K(x, x') = (γ x · x' + r)^d
- RBF (Gaussian) kernel: K(x, x') = exp( -γ ||x - x'||² )

The kernel trick calculates the dot product in the higher dimensional space without actually transforming the data, which keeps the computation efficient.

### 5\. Practical Points

SVR is sensitive to feature scale, so both the inputs and the target should be standardised. It works well for small and medium sized datasets and is robust to outliers because of the ε-insensitive loss, but it becomes slow for very large datasets and the hyperparameters must be tuned, usually with grid search and cross-validation.

## Code

\# Dataset: synthetic noisy sine wave generated in the code (no download)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

\# 1. Generate non-linear data

np.random.seed(1)

X = np.sort(5 \* np.random.rand(200, 1), axis=0)

y = np.sin(X).ravel() + 0.2 \* np.random.randn(200)

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.25, random_state=42)

\# 2. Train SVR with three different kernels

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

print(f"{name:10s} RMSE = {rmse:.4f} R2 = {r2_score(y_test, pred):.4f}"

f" Support vectors = {len(svr.support_)}")

plt.plot(X, svr.predict(X), label=name + " kernel")

plt.xlabel("X")

plt.ylabel("y")

plt.title("Support Vector Regression with different kernels")

plt.legend()

plt.show()

\# 3. Hyperparameter tuning with GridSearchCV (scaled inputs)

pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf"))

param_grid = {

"svr_\_C": \[0.1, 1, 10, 100\],

"svr_\_gamma": \[0.01, 0.1, 1\],

"svr_\_epsilon": \[0.01, 0.1, 0.2\],

}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error")

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

\# 4. Evaluate the best model

best = grid.best_estimator_

y_pred = best.predict(X_test)

print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))

print("MSE :", round(mean_squared_error(y_test, y_pred), 4))

print("R2 :", round(r2_score(y_test, y_pred), 4))

\# 5. Plot the tuned model with the epsilon tube

eps = best.named_steps\["svr"\].epsilon

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

## Output

## Conclusion

Support Vector Regression was implemented with linear, polynomial and RBF kernels on non-linear data. The RBF kernel captured the sine pattern much better than the linear kernel. Grid search was used to tune C, gamma and epsilon, which improved the R² and reduced the error of the model.

# Assignment - 8

**AIM:** Apply regularization to avoid overfitting.

## Theory

### 1\. Overfitting and Underfitting

A model overfits when it learns the training data too well, including its noise, so that it gives very low training error but a high error on unseen data. Overfitting is the result of high variance and usually occurs when the model is too complex (for example a high degree polynomial) or when the training data is small. Underfitting is the opposite case, where the model is too simple to capture the pattern and gives high error on both training and test data (high bias). A good model balances bias and variance.

### 2\. Regularization

Regularization is a technique that discourages model complexity by adding a penalty term on the size of the coefficients to the cost function. The large coefficients that are typical of an overfitted model are shrunk towards zero, which makes the learnt function smoother. The strength of the penalty is controlled by the hyperparameter λ (alpha in scikit-learn), where λ = 0 gives the ordinary model and a very large λ gives an underfitted model.

### 3\. Types of Regularization

- Ridge Regression (L2): J(θ) = MSE + λ Σ θj². It shrinks the coefficients but rarely makes them exactly zero, and works well when many features contribute a little.
- Lasso Regression (L1): J(θ) = MSE + λ Σ |θj|. It can shrink some coefficients exactly to zero, so it performs automatic feature selection.
- Elastic Net: J(θ) = MSE + λ1 Σ |θj| + λ2 Σ θj². It combines the L1 and L2 penalties and is useful when features are correlated.

### 4\. Choosing the Regularization Strength

The best value of λ is found by cross-validation (RidgeCV, LassoCV). Since the penalty depends on the magnitude of the coefficients, the features must be standardised before applying regularization.

### 5\. Other Techniques to Avoid Overfitting

- Collecting more training data or using data augmentation.
- Cross-validation and early stopping during training.
- Feature selection and dimensionality reduction.
- Dropout and weight decay in neural networks.
- Pruning in decision trees and ensemble methods such as Random Forest.

## Code

\# Dataset: synthetic noisy sine data generated in the code (no download)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.linear_model import RidgeCV, LassoCV

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

\# 1. Create a small noisy dataset

np.random.seed(0)

X = np.sort(np.random.rand(40, 1) \* 6, axis=0)

y = np.sin(X).ravel() + 0.3 \* np.random.randn(40)

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.3, random_state=1)

\# 2. High degree polynomial models: without and with regularization

degree = 12

def make_model(reg):

return make_pipeline(PolynomialFeatures(degree, include_bias=False),

StandardScaler(), reg)

models = {

"No regularization": LinearRegression(),

"Ridge (L2)": Ridge(alpha=1.0),

"Lasso (L1)": Lasso(alpha=0.01, max_iter=100000),

"Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000),

}

\# 3. Train and compare train / test performance

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

\# 4. Compare the size of the coefficients

for name in \["No regularization", "Ridge (L2)", "Lasso (L1)"\]:

pipe = make_model(models\[name\]).fit(X_train, y_train)

coef = pipe\[-1\].coef_

print(f"{name:20s} max |coef| = {np.abs(coef).max():10.3f} "

f"zero coefficients = {(np.abs(coef) < 1e-8).sum()}")

\# 5. Choose the best alpha using cross-validation

ridge_cv = make_model(RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(X_train, y_train)

lasso_cv = make_model(LassoCV(alphas=np.logspace(-4, 0, 50), cv=5,

max_iter=100000)).fit(X_train, y_train)

print("Best Ridge alpha:", ridge_cv\[-1\].alpha_)

print("Best Lasso alpha:", lasso_cv\[-1\].alpha_)

print("Ridge CV test R2:", round(r2_score(y_test, ridge_cv.predict(X_test)), 3))

print("Lasso CV test R2:", round(r2_score(y_test, lasso_cv.predict(X_test)), 3))

## Output

## Conclusion

A degree-12 polynomial overfitted the small dataset, giving a very high training score but poor test performance. Ridge, Lasso and Elastic Net regularization reduced the size of the coefficients and improved the test score. Lasso also set some coefficients to zero, so it helps in feature selection as well.

# Assignment - 9

**AIM:** Apply object detection from CCTV Footage.

## Theory

### 1\. Object Detection

Object detection is a computer vision task that finds all the objects of interest in an image or a video frame and, for each of them, gives the class label (what the object is) and a bounding box (where the object is). It is more difficult than image classification, which only assigns one label to the whole image. In CCTV surveillance it is used for counting people, detecting vehicles, intrusion detection, traffic monitoring and recognising suspicious behaviour.

### 2\. Deep Learning Detectors

- Two-stage detectors (R-CNN, Fast R-CNN, Faster R-CNN): first propose regions that may contain objects and then classify each region. They are accurate but slower.
- Single-stage detectors (YOLO, SSD, RetinaNet): predict the boxes and classes directly in one pass of the network, so they are fast enough for real-time video.

### 3\. YOLO (You Only Look Once)

YOLO divides the image into a grid, and every grid cell predicts a number of bounding boxes with their confidence scores and class probabilities. A convolutional neural network backbone extracts features from the frame and a detection head produces the predictions. Because many overlapping boxes are predicted for the same object, Non-Maximum Suppression (NMS) keeps only the box with the highest confidence and removes the others that overlap it strongly. In this assignment the pretrained YOLOv8 model, which is trained on the COCO dataset of 80 classes (person, car, bus, truck, bicycle and so on), is used.

### 4\. Evaluation Metrics

Intersection over Union (IoU) measures the overlap between the predicted box and the actual box: IoU = Area of overlap / Area of union. A prediction is counted as correct when the IoU is above a threshold such as 0.5. Precision, recall and the mean Average Precision (mAP) over all classes are the standard metrics for comparing detectors.

### 5\. Processing Pipeline for CCTV Video

- Read the video file or the camera stream frame by frame using OpenCV.
- Pass each frame to the detection model.
- Filter the detections by confidence threshold and by the required classes.
- Draw bounding boxes with labels, count the objects and save the processed video.

### 6\. Challenges in CCTV Footage

CCTV videos often have low resolution, poor night-time lighting, camera shake, small and distant objects, occlusion in crowds and the need to process frames in real time. These can be reduced by using a better model size, a lower confidence threshold, image enhancement and a GPU.

## Code

\# Install: pip install ultralytics opencv-python

\# Sample CCTV / traffic videos (download any .mp4 and name it cctv.mp4):

\# UA-DETRAC : <https://detrac-db.rit.albany.edu/>

\# MOT17 : <https://motchallenge.net/data/MOT17/>

\# Free clips: <https://www.pexels.com/search/videos/cctv/>

\# The pretrained model yolov8n.pt downloads automatically on first run.

import cv2

from ultralytics import YOLO

\# 1. Load the pretrained YOLOv8 model (trained on COCO dataset)

model = YOLO("yolov8n.pt")

targets = \["person", "car", "bus", "truck", "motorcycle", "bicycle"\]

\# 2. Open the CCTV video and prepare the output writer

cap = cv2.VideoCapture("cctv.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS) or 25

out = cv2.VideoWriter("cctv_output.mp4", cv2.VideoWriter_fourcc(\*"mp4v"),

fps, (width, height))

frame_no = 0

total_counts = {}

\# 3. Process the video frame by frame

while cap.isOpened():

ok, frame = cap.read()

if not ok:

break

frame_no += 1

results = model(frame, conf=0.4, verbose=False)\[0\]

frame_counts = {}

for box in results.boxes:

label = model.names\[int(box.cls\[0\])\]

if label not in targets:

continue

conf = float(box.conf\[0\])

x1, y1, x2, y2 = map(int, box.xyxy\[0\])

cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),

cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

frame_counts\[label\] = frame_counts.get(label, 0) + 1

\# keep the maximum count seen per class and show counts on the frame

for k, v in frame_counts.items():

total_counts\[k\] = max(total_counts.get(k, 0), v)

text = ", ".join(f"{k}: {v}" for k, v in frame_counts.items())

cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,

(0, 0, 255), 2)

out.write(frame)

cv2.imshow("CCTV Object Detection", frame)

if cv2.waitKey(1) & 0xFF == ord("q"):

break

\# 4. Release resources and print summary

cap.release()

out.release()

cv2.destroyAllWindows()

print("Frames processed:", frame_no)

print("Maximum objects seen in one frame:", total_counts)

\# 5. Detection on a single image (optional)

res = model("cctv_frame.jpg", conf=0.4)

res\[0\].save("cctv_frame_detected.jpg")

## Output

## Conclusion

Object detection was applied on CCTV footage using the pretrained YOLOv8 model and OpenCV. Persons and vehicles were detected in every frame with bounding boxes, labels and confidence scores, and the processed video was saved. The single-stage YOLO detector is fast and accurate enough for near real-time surveillance.

# Assignment - 10

**AIM:** Design a model to detect the Fake News from Multimodal data.

## Theory

### 1\. Fake News and Multimodal Data

Fake news is false or misleading information that is presented as real news, and it spreads rapidly on social media. Modern posts normally contain more than one type of data, called modalities, such as the text of the headline or caption and an attached image. Multimodal learning combines these different modalities in one model. Text-only or image-only detectors miss important clues, for example a real photograph used with a false caption, or a manipulated image that goes with a convincing text, so using both modalities gives better detection.

### 2\. Feature Extraction from Each Modality

- Text: the headline is converted to a vector using a pretrained language model such as BERT or DistilBERT, which gives a contextual embedding (768 numbers) of the sentence. Simpler options are TF-IDF and word embeddings.
- Image: a pretrained Convolutional Neural Network such as ResNet or VGG, trained on ImageNet, is used as a feature extractor by removing its last classification layer, giving a 512 or 2048 dimensional vector for every image.

### 3\. Fusion Strategies

- Early (feature-level) fusion: the text and image vectors are concatenated into one vector and given to a single classifier. This is the approach used in this assignment.
- Late (decision-level) fusion: separate classifiers are trained for text and image and their predicted probabilities are combined by averaging or voting.
- Hybrid fusion and attention-based fusion: cross-modal attention learns which words and which image regions are related to each other.

### 4\. Proposed Model

The pipeline is: input post (text and image) → DistilBERT text encoder and ResNet18 image encoder → concatenation of the two feature vectors → classifier (Logistic Regression or a fully connected network) → output label real or fake. The text-only, image-only and multimodal models are trained separately and compared to show the benefit of combining the modalities.

### 5\. Dataset and Evaluation

The Fakeddit dataset is a large multimodal fake news dataset collected from Reddit. Each sample has a title, an image URL and labels for 2-way, 3-way and 6-way classification. The model is evaluated using accuracy, precision, recall, F1-score and the confusion matrix. Challenges include class imbalance, images that are missing or deleted, sarcasm and satire in text, and the high cost of training large models.

## Code

\# Install: pip install torch torchvision transformers scikit-learn pandas pillow requests

\# Dataset (Fakeddit, text + image URLs): <https://github.com/entitize/Fakeddit>

\# Download multimodal_train.tsv (link given in the repository README)

\# Text-only fake news data (optional): <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>

import os

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

\# 1. Load a sample of the multimodal dataset

df = pd.read_csv("multimodal_train.tsv", sep="\\t")

df = df.dropna(subset=\["clean_title", "image_url"\])

df = df.sample(2000, random_state=42).reset_index(drop=True)

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

df\["path"\] = \[download(r) for r in df.itertuples()\]

df = df.dropna(subset=\["path"\]).reset_index(drop=True)

y = df\["2_way_label"\].values

print("Samples with valid image:", len(df), "| label counts:", np.bincount(y))

\# 2. Text features using DistilBERT

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

def text_features(texts, bs=32):

feats = \[\]

with torch.no_grad():

for i in range(0, len(texts), bs):

enc = tok(texts\[i:i + bs\], padding=True, truncation=True,

max_length=64, return_tensors="pt").to(device)

cls = bert(\*\*enc).last_hidden_state\[:, 0, :\]

feats.append(cls.cpu().numpy())

return np.vstack(feats)

\# 3. Image features using ResNet18 (last layer removed)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

resnet.fc = torch.nn.Identity()

resnet = resnet.to(device).eval()

prep = transforms.Compose(\[

transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),

transforms.Normalize(\[0.485, 0.456, 0.406\], \[0.229, 0.224, 0.225\])\])

def image_features(paths, bs=32):

feats = \[\]

with torch.no_grad():

for i in range(0, len(paths), bs):

batch = torch.stack(\[prep(Image.open(p).convert("RGB"))

for p in paths\[i:i + bs\]\]).to(device)

feats.append(resnet(batch).cpu().numpy())

return np.vstack(feats)

X_text = text_features(df\["clean_title"\].tolist())

X_img = image_features(df\["path"\].tolist())

\# 4. Early fusion: concatenate text and image features

X_fused = np.hstack(\[X_text, X_img\])

print("Text:", X_text.shape, "Image:", X_img.shape, "Fused:", X_fused.shape)

\# 5. Train and compare text-only, image-only and multimodal classifiers

idx_train, idx_test = train_test_split(

np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)

def run(X, name):

sc = StandardScaler().fit(X\[idx_train\])

clf = LogisticRegression(max_iter=2000)

clf.fit(sc.transform(X\[idx_train\]), y\[idx_train\])

pred = clf.predict(sc.transform(X\[idx_test\]))

print(f"{name:12s} Accuracy = {accuracy_score(y\[idx_test\], pred):.4f} "

f"F1 = {f1_score(y\[idx_test\], pred):.4f}")

return pred

run(X_text, "Text only")

run(X_img, "Image only")

pred = run(X_fused, "Multimodal")

\# 6. Detailed report of the multimodal model (see README for label meaning)

print(classification_report(y\[idx_test\], pred, target_names=\["Label 0", "Label 1"\]))

## Output

## Conclusion

A multimodal fake news detection model was designed by extracting text features with DistilBERT and image features with ResNet18 and fusing them before classification. The multimodal model performed better than the text-only and image-only models. This shows that combining text and image information gives a more reliable detection of fake news.