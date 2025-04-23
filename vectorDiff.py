import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

model = api.load("glove-wiki-gigaword-300")
pca = PCA(n_components=3)

societyList = [
    "king", "queen", "peasant", "serf", "nobleman", "soldier", "clergyman", "priest", "merchant", "blacksmith",
    "emperor", "empress", "duke", "duchess", "prince", "princess", "baron", "baroness", "lord", "lady"
]



societyListVectors = np.array([model[word] for word in societyList if word in model])


# do PCA on the society vectors
pca.fit(societyListVectors)

# getting the PCA and the direction vector to connect the two points
kingPCA = pca.transform([model["king"]])[0]
peasantPCA = pca.transform([model["peasant"]])[0]
royaltyDirection = peasantPCA - kingPCA


# transform the society words into the PCA space
societyListVectorsSimplified = pca.transform(societyListVectors)

# function that projects point onto line defined by direction and origin (kingPCA)
def projectOntoLine(point, origin, direction):
    direction = direction / np.linalg.norm(direction)
    projectionScalar = np.dot(point - origin, direction)
    projection = origin + projectionScalar * direction
    return projection

# get all the projected points into an array
projected_points = np.array([projectOntoLine(point, kingPCA, royaltyDirection) for point in societyListVectorsSimplified])

# creating the plot
fig = plt.figure(figsize=(10,10))
plot = fig.add_subplot(111,projection='3d')

# plot the points of the kingPCA and the peasant PCA
plot.scatter(kingPCA[0], kingPCA[1], kingPCA[2], color='gold', s=100, label="King")
plot.scatter(peasantPCA[0], peasantPCA[1], peasantPCA[2], color='brown', s=100, label="Peasant")

# plot the royalty axis
plot.plot([kingPCA[0], peasantPCA[0]], [kingPCA[1], peasantPCA[1]], [kingPCA[2], peasantPCA[2]], color='blue', label='royalty axis')

# plot the society words' projections onto the royalty axis
plot.scatter(societyListVectorsSimplified[:,0], societyListVectorsSimplified[:,1], societyListVectorsSimplified[:,2], 
             color = 'red', label = 'society words')

# make orthogonal lines from point to its projection
for i in range(len(societyListVectorsSimplified)):
    plot.plot([societyListVectorsSimplified[i,0], projected_points[i,0]],
              [societyListVectorsSimplified[i,1], projected_points[i,1]],
              [societyListVectorsSimplified[i,2], projected_points[i,2]],
              color = 'grey', linestyle = '--')

# label the points
for i in range(len(societyListVectorsSimplified)):
    plot.text(societyListVectorsSimplified[i,0], societyListVectorsSimplified[i,1], societyListVectorsSimplified[i,2], societyList[i])


plot.set_xlabel("Comp 1")
plot.set_ylabel("Comp 2")
plot.set_zlabel("Comp 3")

plt.legend()
plt.show()