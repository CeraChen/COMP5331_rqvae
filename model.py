import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder
import decoder
import qunatizer


class RQVAE(nn.Module):
    def __init__(self, 
                embedding_dim, 
                vae_hidden_dims=[128, 512, 1024],
                vector_dim=64, 
                vector_num=64, 
                codebook_num=3, 
                beta=0.5, 
                ):
        super().__init__()
        
        self.encoder = encoder.Encoder(embedding_dim, vae_hidden_dims, vector_dim)
        self.decoder = decoder.Decoder(vector_dim, vae_hidden_dims, embedding_dim)
        self.quantizer = qunatizer.ResidualVectorQuantizer(codebook_num, vector_num, vector_dim, beta)
        self._initialize_weights()
        self.recon_criterion = nn.MSELoss()
    
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    
    def forward(self, x):
        features = self.encoder(x)
        quantized, cur_loss, all_indices = self.quantizer(features)
        
        recon_x = self.decoder(quantized)
        recon_loss = self.recon_criterion(recon_x, x)
        cur_loss["reconstruction"] = recon_loss
        
        return quantized, cur_loss, all_indices



    