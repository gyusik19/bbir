import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cosine_similarity_loss(feature1, feature2):
    """
    Compute the cosine similarity loss between two feature tensors.
    
    Args:
    - feature1 (tensor): Feature tensor of shape [batch_size, channels, height, width].
    - feature2 (tensor): Feature tensor of shape [batch_size, channels, height, width].
    - target (str): Either 'similar' or 'dissimilar'. If 'similar', the loss tries to make the features similar.
    
    Returns:
    - loss (tensor): Scalar tensor representing the loss.
    """
    # Flatten the spatial dimensions
    feature1 = feature1.flatten(start_dim=1)  # [batch_size, 2048 * 49]
    feature2 = feature2.flatten(start_dim=1)  # [batch_size, 2048 * 49]
    # feature1 = feature1.view(feature1.size(0), feature1.size(1), -1)  # [batch_size, 2048, 49]
    # feature2 = feature2.view(feature2.size(0), feature2.size(1), -1)  # [batch_size, 2048, 49]
    feature1 = F.normalize(feature1, p=2, dim=1)  # [batch_size, 2048 * 49]
    feature2 = F.normalize(feature2, p=2, dim=1)  # [batch_size, 2048 * 49]
    # Compute cosine similarity for each spatial location
    cosine_sims = F.cosine_similarity(feature1, feature2, dim=1)  # [batch_size]
    
    # Compute the mean cosine similarity across the batch
    loss = 1 - cosine_sims.mean()
    
    return loss