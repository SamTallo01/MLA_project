# Here we keep track of global settings for the project
# For example batch size, learning rate, etc.

# Always import this file as follows:
# from settings import Settings as S

class Settings:

    # Dataset folder
    data_path = "data/"
    
    # Model save folder
    model_save_path = "models/"
    
    backbone = 'resnet50'  # or 'kimia' for custom model
    embed_dim = 1024 # Dimension of the final embedding vector and the imput to attention
    pretrained = True  # Use pretrained weights for backbone if available
    gated = True  # Use gated attention
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10