import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec

corpus = api.load("glove-wiki-gigaword-300")
pca = PCA(n_components=3)

royalty = [
    "king", "queen", "prince", "princess", "emperor", "empress"
]

animal = [
    "elephant", "bear", "deer", "mouse", "germ"
]

bridge = [
    "lionheart", "stallion", "hawk", "dragon", "beast"
]

# royalty = [
#     "king", "queen", "prince", "princess", "emperor", "empress",
#     "peasant", "beggar", "merchant", "townsfolk",
#     "duke", "duchess", "baron", "baroness", "count", "countess", "lord", "lady",
#     "squire", "knight", "viceroy", "archduke", "archduchess", "monarch", "sovereign",
#     "noble", "councilor", "serf", "courtesan", "vassal"
# ]

# animal = [
#     "lion", "tiger", "wolf", "eagle", "elephant", "bear", "deer", "mouse", "germ",
#     "horse", "primeape", "monkey",
#     "zebra", "giraffe", "panda", "koala", "kangaroo", "rabbit", "squirrel", "otter",
#     "fox", "leopard", "jaguar", "hippopotamus", "crocodile", "whale", "dolphin",
#     "shark", "octopus", "platypus", "owl", "falcon", "parrot"
# ]

#establish a baseline, query gensim for all the words and sample a subset

# bridge = [
#     "lionheart", "stallion", "hawk", "dragon", "beast",
#     "griffin", "phoenix", "wyvern", "pegasus", "unicorn", "cherub", "cerberus",
#     "minotaur", "chimera", "troll", "ogre", "satyr", "centaur", "golem", "giant"
# ]


full_list = royalty + animal + bridge
full_list_vectors = np.array([corpus[word] for word in full_list if word in corpus])


# fitting the pca space to the list of all the vectors to find similarity
pca.fit(full_list_vectors)


# specific to the hierarchy analysis
low = pca.transform([corpus["germ"]])[0]
high = pca.transform([corpus["emperor"]])[0]
hierarchy_pca_dir_vector = high - low


# specific to physical power? not sure this works
low_p = pca.transform([corpus["mouse"]])[0]
high_p = pca.transform([corpus["dragon"]])[0]
power_pca_dir_vector = high_p - low_p


# specific to intelligence? not sure this works
low_i = pca.transform([corpus["deer"]])[0]
high_i = pca.transform([corpus["empress"]])[0]
intel_pca_dir_vector = high_i - low_i


# function to calculate the projection of the point onto the line
def project(point, origin, direction):
    direction /= np.linalg.norm(direction)
    calc = np.dot(point, direction)
    return origin + calc * direction
    
fig = plt.figure(figsize=(10,10))
plot = fig.add_subplot(111, projection = '3d')


# transform all the data points
full_simplified = pca.transform(full_list_vectors)


# get the array of the points projected onto the line
points_projected_onto_line = np.array([project(vec, high, hierarchy_pca_dir_vector) for vec in full_simplified])

points_projected_onto_line_p = np.array([project(vec, high_p, power_pca_dir_vector) for vec in full_simplified])

points_projected_onto_line_i = np.array([project(vec, high_i, intel_pca_dir_vector) for vec in full_simplified])


# plot all the points in the pca space
plot.scatter(full_simplified[:,0], full_simplified[:,1], full_simplified[:,2], color = 'black', label = 'all words')


# plot the distance between the point on the line to its actual location
for i in range(len(points_projected_onto_line)):
    plot.plot([points_projected_onto_line[i,0], full_simplified[i,0]],
              [points_projected_onto_line[i,1], full_simplified[i,1]],
              [points_projected_onto_line[i,2], full_simplified[i,2]], 
              color = "grey", linestyle = "--")
    
for i in range(len(points_projected_onto_line_p)):
    plot.plot([points_projected_onto_line_p[i,0], full_simplified[i,0]],
              [points_projected_onto_line_p[i,1], full_simplified[i,1]],
              [points_projected_onto_line_p[i,2], full_simplified[i,2]], 
              color = "#768982", linestyle = "--")
    
for i in range(len(points_projected_onto_line_i)):
    plot.plot([points_projected_onto_line_i[i,0], full_simplified[i,0]],
              [points_projected_onto_line_i[i,1], full_simplified[i,1]],
              [points_projected_onto_line_i[i,2], full_simplified[i,2]], 
              color = "#b5c2bd", linestyle = "--")

# show the text for all the words
for i in range(len(full_simplified)):
    plot.text(full_simplified[i,0], full_simplified[i,1], full_simplified[i,2], full_list[i])


# plot the lines associated
plot.plot([low[0],high[0]],[low[1],high[1]],[low[2],high[2]], color = 'purple', label = 'hierarchy line')

plot.plot([low_p[0],high_p[0]],[low_p[1],high_p[1]],[low_p[2],high_p[2]], color = 'red', label = 'power line')

plot.plot([low_i[0],high_i[0]],[low_i[1],high_i[1]],[low_i[2],high_i[2]], color = 'pink', label = 'intelligence line')


# finally plot the end points
plot.scatter(low[0], low[1], low[2], color='#594a17', s=100, label="germ")
plot.scatter(high[0], high[1], high[2], color='#e6be36', s=100, label="emperor")

plot.scatter(low_p[0], low_p[1], low_p[2], color='#1bcf8d', s=100, label="mouse")
plot.scatter(high_p[0], high_p[1], high_p[2], color='#087c52', s=100, label="dragon")

plot.scatter(low_i[0], low_i[1], low_i[2], color='#3f3587', s=100, label="deer")
plot.scatter(high_i[0], high_i[1], high_i[2], color='#796cd3', s=100, label="empress")


# plot the line on the pca space
plot.set_xlabel("comp1")
plot.set_ylabel("comp2")
plot.set_zlabel("comp3")

plot.legend()
plt.show()

print(pca.explained_variance_ratio_)

