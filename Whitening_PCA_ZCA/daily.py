import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
mu = [0, 0]
sigma = [[5, 4], [4, 5]]
n = 1000
x = np.random.multivariate_normal(mu, sigma, size=n).T

set1 = np.argsort(np.linalg.norm(x, axis=0))[-20:]
set2 = list(set(range(n)) - set(set1))

fig, ax = plt.subplots()
ax.scatter(x[0, set1], x[1, set1], s=20, c="red", alpha=0.2)
ax.scatter(x[0, set2], x[1, set2], s=20, alpha=0.2)
ax.set_aspect("equal")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Original")
# plt.show()
print(np.corrcoef(x)[0, 1])
# 0.8020186259500502

sigma = np.cov(x)
# [4.07635286 5.19821674]]
evals, evecs = np.linalg.eigh(sigma)
# [1.00594615 9.16185704]
# [[-0.71694925  0.69712536]
#  [ 0.69712536  0.71694925]]

# PCA
z = np.diag(evals ** (-1 / 2)) @ evecs.T @ x
fig, ax = plt.subplots()
ax.scatter(z[0, set1], z[1, set1], s=20, c="red", alpha=0.2)
ax.scatter(z[0, set2], z[1, set2], s=20, alpha=0.2)
ax.set_aspect("equal")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("PCA")

# ZCA
z = evecs @ np.diag(evals**(-1/2)) @ evecs.T @ x
fig, ax = plt.subplots()
ax.scatter(z[0, set1], z[1, set1], s=20, c="red", alpha=0.2)
ax.scatter(z[0, set2], z[1, set2], s=20, alpha=0.2)
ax.set_aspect("equal")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("ZCA")
plt.show()