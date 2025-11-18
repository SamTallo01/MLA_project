import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# Output feature vector dimension for ResNet50 after AdaptiveAvgPool is 2048
OUTPUT_DIM = 2048

def get_feature_extractor(device="cuda"):
    """
    Loads a pre-trained ResNet-50 model (ImageNet weights) and converts it 
    into a feature extractor by removing the final classification layer.
    
    Returns:
        model (nn.Module): The feature extraction model.
        
        NOTE: The image transformation (transforms) is omitted here, as it
              is handled by the calling function in extract_fv.py.
    """
    
    # Load the official pre-trained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Remove the final fully connected (fc) layer, keeping AdaptiveAvgPool2d
    feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
    
    # Move model to device and set to evaluation mode
    feature_extractor = feature_extractor.to(device).eval()
    
    # The transform definition is intentionally removed from here.
    
    return feature_extractor 

# Expose the expected feature vector dimension
def get_output_dim():
    return OUTPUT_DIM

# if __name__ == '__main__':
#     # Simple test of the extractor
#     model = get_feature_extractor(device="cuda" if torch.cuda.is_available() else "cpu")
#     print(f"ResNet-50 Feature Extractor loaded successfully. Output dim: {get_output_dim()}")
    
#     # In a real use case, you would define the transform here:
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])