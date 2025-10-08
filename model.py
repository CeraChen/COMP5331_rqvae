import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder
import decoder
import quantizer


class RQVAE(nn.Module):
    def __init__(self, 
                embedding_dim, 
                vae_hidden_dims,
                vector_dim, 
                vector_num, 
                codebook_num, 
                commitment_weight,
                random_state 
                ):
        super().__init__()
        
        ## Encoder will rearrange the hidden dims decreasingly (downsampling)
        ## while Decoder will rearrange them increasingly (upsampling)
        self.encoder = encoder.Encoder(embedding_dim, vae_hidden_dims, vector_dim)
        self.decoder = decoder.Decoder(vector_dim, vae_hidden_dims, embedding_dim)
        self.quantizer = quantizer.ResidualVectorQuantizer(codebook_num, vector_num, vector_dim, commitment_weight)
        self._initialize_weights()
        self.recon_criterion = nn.MSELoss()
        self.random_state = random_state
    
    
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

    
    def initialize(self, x):
        features = self.encoder(x)
        self.quantizer.initialize(features, self.random_state)
    