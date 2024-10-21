import torch

def kmeans(data, K, max_iters=100):
    # Step 1: Initialize centers (K-means++ or random)
    centers = data[torch.randperm(data.size(0))[:K]]
    
    for _ in range(max_iters):
        # Step 2: Assign data points to nearest centers
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)
        
        # Step 3: Update centers
        new_centers = torch.stack([data[labels == k].mean(dim=0) for k in range(K)])
        
        # Check for convergence
        try:
            if torch.all(new_centers == centers):
                break
        except RuntimeError: pass
            
        centers = new_centers
    
    return centers
