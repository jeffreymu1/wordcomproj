import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # loads from pretrained word vector database, grand et al used glove
# model = api.load("glove-wiki-gigaword-300")

# size_axis = model["big"] - model["small"]
# intelligence_axis = model["smart"] - model["foolish"]
# strength_axis = model["strong"] - model["weak"]

# ##normalize
# size_axis /= np.linalg.norm(size_axis)
# intelligence_axis /= np.linalg.norm(intelligence_axis)
# strength_axis /= np.linalg.norm(strength_axis)

# # project onto an axis
# def project_word(word, axis):
#     if word in model:
#         word_vector = model[word]
#         return np.dot(word_vector, axis)
#     else:
#         return None
    
# words = ["mouse", "cat", "dog", "horse", "elephant", "dolphin", "alligator", "tiger", "whale"]

# # Compute projections
# projections = {word: (
#     project_word(word, size_axis),
#     project_word(word, intelligence_axis),
#     project_word(word, strength_axis)
# ) for word in words if word in model}

# # Print results
# print("\nProjections onto semantic axes:")
# for word, values in projections.items():
#     print(f"{word}: Size={values[0]:.4f}, Danger={values[1]:.4f}, Intelligence={values[2]:.4f}")

# coords = np.array(list(projections.values()))
# labels = list(projections.keys())

# #plot
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')

# #scater
# ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker='o')

# # Annotate each point
# for i, label in enumerate(labels):
#     ax.text(coords[i, 0], coords[i, 1], coords[i, 2], label)

# # Axis labels
# ax.set_xlabel('Size')
# ax.set_ylabel('Danger')
# ax.set_zlabel('Intelligence')
# plt.title('Semantic Projection of Word Embeddings')

# plt.show()

###########################
#            2            #
###########################

model = api.load("glove-wiki-gigaword-300")

size_axis = model["large"] - model["small"]
point_large = model["large"]
point_small = model["small"]

animals = [
    "mouse", "cat", "dog", "horse", "elephant", "dolphin", "alligator", "tiger", "whale",
    "lion", "zebra", "giraffe", "kangaroo", "panda", "gorilla", "rabbit", "wolf", "bear", "fox",
    "cheetah", "leopard", "camel", "otter", "beaver"]



animal_vectors = np.array([model[animal] for animal in animals if animal in model])


pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(animal_vectors)

scaler = MinMaxScaler()
reduced_vectors = scaler.fit_transform(reduced_vectors)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], color="blue", label="animals")


size_axis_pca = pca.transform([size_axis])[0]
large_point_pca = pca.transform([point_large])[0]
small_point_pca = pca.transform([point_small])[0]

size_line_extension = 50
size_line_start = -size_line_extension
size_line_end = size_line_extension
size_line = np.linspace(size_line_start, size_line_end, num=100)
size_axis_3D = np.outer(size_line, size_axis_pca) 

projected_large = (np.dot(large_point_pca, size_axis_pca) / np.dot(size_axis_pca, size_axis_pca)) * size_axis_pca
projected_small = (np.dot(small_point_pca, size_axis_pca) / np.dot(size_axis_pca, size_axis_pca)) * size_axis_pca


ax.plot(size_axis_3D[:, 0], size_axis_3D[:, 1], size_axis_3D[:, 2], color="red", label="Size Axis")
ax.scatter(projected_large[0], projected_large[1], projected_large[2], color="green", marker="o", label="Large")
ax.scatter(projected_small[0], projected_small[1], projected_small[2], color="purple", marker="o", label="Small")


for i, animal_vector in enumerate(reduced_vectors):
    projection_point = (np.dot(animal_vector, size_axis_pca) / np.dot(size_axis_pca, size_axis_pca)) * size_axis_pca * 20
    
    ax.plot([animal_vector[0], projection_point[0]],
            [animal_vector[1], projection_point[1]],
            [animal_vector[2], projection_point[2]],
            color="lightblue", linestyle="dotted")

    ax.text(animal_vector[0], animal_vector[1], animal_vector[2], animals[i])

plt.plot(model["large"], 2, marker='o', linestyle='none')



ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.legend()

plt.show()





###########################
#          Notes          #
###########################

# final conclusion doesn't work for some reason for small and large points. not sure why this is