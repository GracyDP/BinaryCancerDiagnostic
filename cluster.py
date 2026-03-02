import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv("dataset/clean_data.csv")

df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

X = df.drop(columns=["id", "diagnosis"], errors="ignore")
y = df["diagnosis"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Varianza spiegata PCA:", pca.explained_variance_ratio_.sum())

kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_pca)

# Valutazione (NON supervisionata)
ari = adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index: {ari:.3f}")

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=clusters,
    palette="Set1",
    alpha=0.7
)
plt.title("K-Means Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.tight_layout()

plt.tight_layout()
plt.savefig("dataset/kmeans_pca.png", dpi=300, bbox_inches="tight")

plt.show()


# Confronto cluster vs diagnosis
ct = pd.crosstab(clusters, y, rownames=["Cluster"], colnames=["Diagnosis"])
print("\nContingenza Cluster vs Diagnosis")
print(ct)




from mpl_toolkits.mplot3d import Axes3D

#PCA a 3 componenti principali
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

#grafico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#colori per le classi (benigno vs maligno)
ax.scatter(
    X_pca_3d[y == 0, 0], X_pca_3d[y == 0, 1], X_pca_3d[y == 0, 2], 
    c='b', label='Benigno', alpha=0.6, edgecolors='w', s=80
)
ax.scatter(
    X_pca_3d[y == 1, 0], X_pca_3d[y == 1, 1], X_pca_3d[y == 1, 2], 
    c='r', label='Maligno', alpha=0.6, edgecolors='w', s=80
)

# assi
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

#  legenda
ax.set_title('Proiezione 3D delle prime 3 componenti principali (Benigno vs Maligno)')
ax.legend()

# Salvataggio della figura
plt.tight_layout()
plt.savefig("dataset/pca_3d.png", dpi=300, bbox_inches="tight")
plt.show()
