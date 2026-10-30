import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = {
    "Student": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics": [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms": [13, 15, 9, 17, 10, 14],
}
df = pd.DataFrame(data).set_index("Student")
print(df)

print("\nShape:", df.shape)
print(df.describe())

mean = df.mean()
std = df.std()
Z = (df - mean) / std
print("\nStandardized data:")
print(Z.round(3))

corr = np.corrcoef(Z.values, rowvar=False)
print("\nCorrelation matrix:")
print(pd.DataFrame(corr, index=df.columns, columns=df.columns).round(3))

eigenvalues, eigenvectors = np.linalg.eig(corr)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("\nEigenvalues:", eigenvalues.round(4))
print("Eigenvectors (columns):")
print(eigenvectors.round(4))

W = eigenvectors[:, :2]

scores_scratch = Z.values @ W
print("\nCoordinates (scratch):")
print(pd.DataFrame(scores_scratch, index=df.index, columns=["PC1", "PC2"]).round(4))

ratio_scratch = eigenvalues / eigenvalues.sum()
cum_scratch = np.cumsum(ratio_scratch)
print("\nExplained variance ratio:", ratio_scratch.round(4))
print("Cumulative:", cum_scratch.round(4))

pca = PCA(n_components=2)

scores_sk = pca.fit_transform(Z.values)
print("\nCoordinates (sklearn):")
print(pd.DataFrame(scores_sk, index=df.index, columns=["PC1", "PC2"]).round(4))

ratio_sk = pca.explained_variance_ratio_
cum_sk = np.cumsum(ratio_sk)
print("\nExplained variance ratio:", ratio_sk.round(4))
print("Cumulative:", cum_sk.round(4))

print("\nScratch coordinates - sklearn coordinates:")
print((scores_scratch - scores_sk).round(4))
print("Scratch + sklearn coordinates:")
print((scores_scratch + scores_sk).round(4))

print("\nEigenvalues scratch:", eigenvalues[:2].round(4))
print("Eigenvalues sklearn:", pca.explained_variance_.round(4))
print("Ratio scratch:", ratio_scratch[:2].round(4))
print("Ratio sklearn:", ratio_sk.round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, scores, title in zip(axes, [scores_scratch, scores_sk],
                             ["PCA from scratch", "PCA scikit-learn"]):
    ax.scatter(scores[:, 0], scores[:, 1])
    for i, name in enumerate(df.index):
        ax.annotate(name, (scores[i, 0], scores[i, 1]),
                    xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("pca_students.png", dpi=150)
plt.show()

print("\nVariance retained by PC1+PC2: %.2f %%" % (cum_scratch[1] * 100))
