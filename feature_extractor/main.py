import openslide
from PIL import Image

slide = openslide.OpenSlide("C:\Users\reem_\Desktop\polito\sem10\Machine learning in applications\project\dataset/M-1.ndpi")
print("Slide dimensions:", slide.dimensions)

# Extract a tile at level 0 (highest resolution)
tile = slide.read_region((x, y), level=0, size=(224,224))  # x, y coordinates
tile = tile.convert("RGB")  # Removes alpha channel

tile.save("test_tile.png")
print("Tile saved as test_tile.png")


from kimianet_features import KimiaNetFeatureExtractor

# Instantiate the extractor (choose 'cuda' if you have a GPU available)
extractor = KimiaNetFeatureExtractor(device="cuda")
print("KimiaNet model loaded.")

# Path to your test image tile
tile_path = "test_tile.png"  # Change to the actual image tile you have

# Extract features
features = extractor.extract_tile_features(tile_path)
print("KimiaNet features shape:", features.shape)
print("KimiaNet feature vector (first 5 elements):", features[:5])


