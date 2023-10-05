import torch
import numpy as np
import time

class BatchKNearestNeighbor:
    def __init__(self, batch_size=10000, device="cuda:0", mask=False):
        self.data = None
        self.device = device
        self.batch_size = batch_size
        self.mask = mask

    def fit(self, data):
        # Convert to FP16 while transferring to the GPU
        self.data = torch.tensor(data, dtype=torch.float16).to(self.device)

    def cosine_distance_masked(self, query, data_batch):
        if self.mask:
            non_zero_indices = torch.where(query != 0)[0]
            masked_query = query[non_zero_indices]
            masked_data_matrix = data_batch[:, non_zero_indices]
        else:
            masked_query = query
            masked_data_matrix = data_batch

        dot_products = masked_query @ masked_data_matrix.T
        query_norm = torch.linalg.norm(masked_query)
        data_norms = torch.linalg.norm(masked_data_matrix, dim=1)

        cosine_similarities = dot_products / (query_norm * data_norms)
        cosine_distances = 1.0 - cosine_similarities

        return cosine_distances

    def predict(self, queries, k=1):
        # Convert queries to FP16 before transferring to GPU
        queries_tensor = torch.tensor(queries, dtype=torch.float16).to(self.device)
        all_neighbors = []
        all_distances_flat = []

        with torch.no_grad():
            for idx, query in enumerate(queries_tensor):
                all_distances = []
                for i in range(0, len(self.data), self.batch_size):
                    data_batch = self.data[i:i+self.batch_size]
                    distances_batch = self.cosine_distance_masked(query, data_batch)
                    all_distances.append(distances_batch)

                distances = torch.cat(all_distances)
                sorted_indices = torch.argsort(distances)[:k]

                all_neighbors.append(sorted_indices.cpu().numpy())
                all_distances_flat.append(distances[sorted_indices].cpu().numpy())

        return all_neighbors, all_distances_flat