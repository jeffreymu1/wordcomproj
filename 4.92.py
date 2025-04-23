import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec

# get all the words from glove

corpus = api.load("glove-wiki-gigaword-300")
vocab_list = list(corpus.key_to_index.keys())

expanded_vectors = np.array([corpus[word] for word in vocab_list if word in corpus])

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

small_avg = np.mean([corpus[word] for word in small if word in corpus], axis=0) 
medium_avg = np.mean([corpus[word] for word in medium if word in corpus], axis=0)
large_avg = np.mean([corpus[word] for word in large if word in corpus], axis=0)

direction = large_avg - small_avg
direction /= np.linalg.norm(direction)

def group_proj(words, origin, direction):
    projections = []
    for word in words:
        if word in corpus:
            vec = corpus[word]
            dotted = np.dot(vec - origin, direction)
            projections.append(dotted)
    return np.mean(projections), np.std(projections)

small_score, small_std = group_proj(small, small_avg, direction)
medium_score, mid_std = group_proj(medium, small_avg, direction)
large_score, large_std = group_proj(large, small_avg, direction)

group_scores = [small_score, medium_score, large_score]
group_errors = [small_std, mid_std, large_std]

group_names = ['small', 'medium', 'large']
group_scores = [small_score, medium_score, large_score]
plt.figure(figsize=(8, 5))
bars = plt.bar(group_names, group_scores, yerr=group_errors, color=['#a6cee3', '#1f78b4', '#b2df8a'], capsize=5)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}', ha='center')

plt.ylabel("avg size score")
plt.show()