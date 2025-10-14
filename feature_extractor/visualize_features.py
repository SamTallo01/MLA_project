import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the feature dictionary from the pickle file
with open("extracted_features/KimiaNet_NDPI_Features.pickle", "rb") as f:
    features_dict = pickle.load(f)

# Now you can extract features and patch names from the loaded dictionary
features = np.array(list(features_dict.values()))
patch_names = list(features_dict.keys())

# Dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(features_2d[:,0], features_2d[:,1], s=5)
plt.title('t-SNE of KimiaNet extracted features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()



from PIL import Image
import matplotlib.pyplot as plt

# Load and display a sample patch image by patch name
patch_name = patch_names[0]  # example patch
patch_path = f"./patch_images/{patch_name}.png"  # your image storage path

patch_img = Image.open(patch_path)
plt.imshow(patch_img)
plt.title(f"Patch: {patch_name}")
plt.axis('off')
plt.show()


from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

similarity_matrix = cosine_similarity(features)
plt.figure(figsize=(10,8))
sns.heatmap(similarity_matrix, cmap='viridis')
plt.title('Similarity between patch features')
plt.show()
