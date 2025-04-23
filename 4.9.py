import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec

# get all the words from glove

corpus = api.load("glove-wiki-gigaword-300")
vocab_list = list(corpus.key_to_index.keys())

expanded_vectors = np.array([corpus[word] for word in vocab_list])

# fitting the pca space to the list of all the vectors to find similarity
pca = PCA(n_components=3)


### defining the groupings transformed in the PCA space

small = [
    "pebble", "mouse", "atom", "seed", "spark", "ant", "droplet", "atom", "miniature", "tiny", "small"
]

medium = [
    "cat", "book", "lamp", "dog", "basket", "kid", "guitar", "fox", "average", "medium"
]

large = [
    "airplane", "boulder", "whale", "mountain", "skyscraper", "titan", "mammoth", "massive", "jumbo", "large"
]

full_list = small + medium + large
full_list_vectors = np.array([corpus[word] for word in full_list if word in corpus])
pca.fit(expanded_vectors)

# go through all the vectors in particular group and get avg of the group
sum_s = np.zeros(300)
for word in small:
    sum_s += corpus[word]

sum_l = np.zeros(300)
for word in large:
    sum_l += corpus[word]

sum_m = np.zeros(300)
for word in medium:
    sum_m += corpus[word]

large_avg = sum_l/len(large)
small_avg = sum_s/len(small)
mid_avg = sum_m/len(medium)

large_avg_pca = pca.transform([large_avg])[0]
small_avg_pca = pca.transform([small_avg])[0]
mid_avg_pca = pca.transform([mid_avg])[0]

## defining the direction vector created by the groupings of points
size_dir_vector = large_avg_pca - small_avg_pca


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
points_projected_onto_line = np.array([project(vec, small_avg_pca, size_dir_vector) for vec in full_simplified])

projections_1d = [np.dot(p - small_avg_pca, size_dir_vector) / np.linalg.norm(size_dir_vector) 
                  for p in points_projected_onto_line]

# pair words with their projection values
word_projections = list(zip(full_list, projections_1d))

# from smallest to largest
word_projections_sorted = sorted(word_projections, key=lambda x: x[1])

print("Words in order along the size line:")
for word, _ in word_projections_sorted:
    print(word)


# plot all the points in the pca space
plot.scatter(full_simplified[:,0], full_simplified[:,1], full_simplified[:,2], color = 'black', label = 'all words')


# plot the distance between the point on the line to its actual location
for i in range(len(points_projected_onto_line)):
    plot.plot([points_projected_onto_line[i,0], full_simplified[i,0]],
              [points_projected_onto_line[i,1], full_simplified[i,1]],
              [points_projected_onto_line[i,2], full_simplified[i,2]], 
              color = "grey", linestyle = "--")

# show the text for all the words
for i in range(len(full_simplified)):
    plot.text(full_simplified[i,0], full_simplified[i,1], full_simplified[i,2], full_list[i])


# plot the line associated
plot.plot([small_avg_pca[0],large_avg_pca[0]],[small_avg_pca[1],large_avg_pca[1]],[small_avg_pca[2],large_avg_pca[2]], color = 'purple', label = 'size line')


# finally plot the end points
plot.scatter(small_avg_pca[0], small_avg_pca[1], small_avg_pca[2], color='#594a17', s=100, label="smallavg")
plot.scatter(large_avg_pca[0], large_avg_pca[1], large_avg_pca[2], color='#e6be36', s=100, label="largeavg")

# plot the line on the pca space
plot.set_xlabel("Component 1")
plot.set_ylabel("Component 2")
plot.set_zlabel("Component 3")

print(pca.explained_variance_ratio_)

plot.legend()
plt.show()