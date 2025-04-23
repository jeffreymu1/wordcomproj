### Demo to plot different categories on the same dimensional grid and whether this affects encoding ###


import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# load words from glove model
model = api.load("glove-wiki-gigaword-300")

societyList = [
    "king", "queen", "peasant", "serf", "nobleman", "soldier", "clergyman", "priest", "merchant", "blacksmith",
    "emperor", "empress", "duke", "duchess", "prince", "princess", "baron", "baroness", "lord", "lady",
    "farmer", "laborer", "miller", "carpenter", "mason", "weaver", "tailor", "potter", "innkeeper", "scribe",
    "knight", "warrior", "general", "archer", "pikeman", "guard", "squire", "crusader", "mercenary", "cavalry",
    "bishop", "monk"
]

foodList = [
    "bread", "cheese", "wine", "beer", "milk", "butter", "meat", "fish", "fruit", "vegetable",
    "grain", "honey", "salt", "spices", "tea", "coffee", "chocolate", "pastry", "olive_oil", "rice",
    "nuts", "sugar", "corn", "egg", "yogurt", "pasta", "beans", "lentils", "tofu", "mushrooms"
]

animalList = [
    "lion", "tiger", "elephant", "giraffe", "zebra", "wolf", "fox", "bear", "deer", "rabbit",
    "eagle", "hawk", "owl", "snake", "crocodile", "turtle", "shark", "whale", "dolphin", "octopus",
    "horse", "cow", "pig", "sheep", "goat", "chicken", "duck", "dog", "cat", "mouse"
]

# makes an array of the 300D vector encodings for all the words in the societyList
societyListDimensionStorage = np.array([model[word] for word in societyList if word in model])
foodListDimensionStorage = np.array([model[word] for word in foodList if word in model])
animalListDimensionStorage = np.array([model[word] for word in animalList if word in model])

# compress the dimensionality of all the words into three dimensions
pca = PCA(n_components=3)
reducedRoyalVectors = pca.fit_transform(societyListDimensionStorage)
reducedFoodVectors = pca.fit_transform(foodListDimensionStorage)
reducedAnimalVectors = pca.fit_transform(animalListDimensionStorage)

# 5x5, 1 by 1 in one plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111,projection='3d')

## royal
# plot all the points in reducedRoyalVectors in the PCA space, iterating through the list
ax.scatter(reducedRoyalVectors[:,0], reducedRoyalVectors[:,1], reducedRoyalVectors[:,2], color="green", label="Society words")

# show the text for each point
for i in range(len(reducedRoyalVectors)):
    ax.text(reducedRoyalVectors[i, 0], reducedRoyalVectors[i, 1], reducedRoyalVectors[i, 2], societyList[i])


## food
# plot all the points in reducedFoodVectors in the PCA space, iterating through the list
ax.scatter(reducedFoodVectors[:,0], reducedFoodVectors[:,1], reducedFoodVectors[:,2], color="red", label="Food words")

# show the text for each point
for i in range(len(reducedFoodVectors)):
    ax.text(reducedFoodVectors[i, 0], reducedFoodVectors[i, 1], reducedFoodVectors[i, 2], foodList[i])


## animals
# plot all the points in reducedAnimalVectors in the PCA space, iterating through the list
ax.scatter(reducedAnimalVectors[:,0], reducedAnimalVectors[:,1], reducedAnimalVectors[:,2], color="blue", label="Animal words")

# show the text for each point
for i in range(len(reducedAnimalVectors)):
    ax.text(reducedAnimalVectors[i, 0], reducedAnimalVectors[i, 1], reducedAnimalVectors[i, 2], animalList[i])







ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

plt.legend()
plt.show()