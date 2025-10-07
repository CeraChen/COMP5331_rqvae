import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, vector_num, vector_dim, beta):
        super().__init__()
        self.vector_dim = vector_dim
        self.vector_num = vector_num
        self.beta = beta
        
        self.vectors = nn.Embedding(vector_num, vector_dim)
        self.vectors.weight.data.uniform_(-1/vector_num, 1/vector_num) # to K-means
        
    def forward(self, x):
        ## Compute the L2 distance
        # x: batch, vector_dim
        # weight: vector_num, vector_dim
        distances = (torch.sum(x**2, dim=1, keepdim=True) 
                    + torch.sum(self.vectors.weight**2, dim=1)
                    - 2 * torch.matmul(x, self.vectors.weight.t()))
        # distances: batch, vector_num
        
        ## Get the vector
        vector_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(vector_indices, self.vector_num).float()
        quantized = torch.matmul(encodings, self.vectors.weight)
        # quantized = quantized.view(input_shape)
        
        
        cur_loss = {}
        ## Compute the quant loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        quant_loss = q_latent_loss + self.beta * e_latent_loss
        
        ## Get the quantized outputs
        quantized = x + (quantized - x).detach()
        
        ## Compute the diverse loss
        counts = torch.sum(encodings, dim=0)
        N = x.shape[0]
        util_loss = torch.sum(torch.abs(counts - N/self.vector_num))/N
        compact_loss = 2*torch.mean(torch.pdist(self.vectors.weight, p=2))
        # div_loss = util_loss + compact_loss
        
        cur_loss["quantization"] = quant_loss
        cur_loss["utilization"] = util_loss
        cur_loss["compactness"] = compact_loss
        
        return quantized, cur_loss, vector_indices
    
    
    
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_num, vector_num, vector_dim, beta):
        super().__init__()
        self.codebook_num = codebook_num
        self.quantizers = nn.ModuleList([
            VectorQuantizer(vector_num, vector_dim, beta) 
            for _ in range(codebook_num)
        ])
        self.loss_items = ["quantization", "utilization", "compactness"]
        
    def forward(self, x):
        quantized = 0
        residual = x
        all_indices = []
        total_loss = {k: 0.0 for k in self.loss_items}
        
        for _, quantizer in enumerate(self.quantizers):
            cur_quantized, cur_loss, indices = quantizer(residual)
            
            quantized += cur_quantized
            residual = x - quantized
            
            all_indices.append(indices)
            total_loss = {k: total_loss[k]+cur_loss[k] for k in total_loss.keys()}
            
        return quantized, total_loss, all_indices


