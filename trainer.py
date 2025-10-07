import torch
import torch.nn as nn
import torch.optim as optim
import model

## Configs
BATCH_SIZE = 128
EPOCH_NUM = 50

CODEBOOL_NUM = 3
VECTOR_NUM = 64
VECTOR_DIM = 64
VAE_HIDDEN_DIMS = [128, 512, 1024]

QUANT_WEIGHT = 1.0
DIV_WEIGHT = 0.25
COMMITMENT_WEIGHT = 0.5
RANDOM_STATE = 43

LOSS_TERMS = [
    "reconstruction",
    "quantization",
    "utilization",
    "compactness"
]
LOSS_WEIGHTS = {
    "reconstruction": 1.0,
    "quantization": QUANT_WEIGHT,
    "utilization": DIV_WEIGHT,
    "compactness": DIV_WEIGHT
}

LR = 1e-5
INPUT_DIM_MAP = {
    "category": (301, 1),
    "region": (1000, 1),
    "temporal": (24, 10),
    "collaborative": (6592, 10)
    # (total_dim, k_hot)
}
# embedding is concatenated from the above dimensions, e.g., 301+1000+24+6592 = 7917
EMBEDDING_DIM = sum([i[0] for i in INPUT_DIM_MAP.values()])




# Initialize RQVAE model and optimizer
device = 'cpu'
rqvae_model = model.RQVAE(
    embedding_dim=EMBEDDING_DIM,
    vae_hidden_dims=VAE_HIDDEN_DIMS,
    vector_dim=VECTOR_DIM, 
    vector_num=VECTOR_NUM, 
    codebook_num=CODEBOOL_NUM, 
    commitment_weight=COMMITMENT_WEIGHT,
    random_state=RANDOM_STATE
    ).to(device)

optimizer = optim.Adam(
            rqvae_model.parameters(),
            lr=LR,
            # betas=(0.9, 0.98),
            # eps=1e-9
        )




## For randomly initializing inputs
def vectorized_k_hot(batch_size, num_classes, k):
    indices = torch.argsort(torch.rand(batch_size, num_classes), dim=1)
    k_hot = torch.zeros(batch_size, num_classes)
    k_hot.scatter_(1, indices[:, :k], 1.0)
    return k_hot


## Synthesize inputs
batches = []
for i in range(10):
    x = []
    for dim_name, dim_item in INPUT_DIM_MAP.items():
        dim, k = dim_item
        k_hot = vectorized_k_hot(BATCH_SIZE, dim, k)
        # print(k_hot.shape)
        x.append(k_hot)
    x = torch.cat(x, dim=1)
    # print("Synthesized inputs (batch x dims):", x.shape)
    batches.append(x)



## Training
rqvae_model.train()
for epoch in range(EPOCH_NUM):
    print("Start epoch", epoch)
    total_loss_dict = {k: 0.0 for k in LOSS_TERMS}    
    
    ## TODO: add dataloader
    for step, batch in enumerate(batches):
        if epoch == 0 and step == 0:
            # Skip training but initialize quantizer via k-means with the first batch
            rqvae_model.initialize(x)
            continue 
        
        x = x.to(device)
        optimizer.zero_grad()
        quantized, step_loss_dict, all_indices = rqvae_model(x)
        
        loss = sum([step_loss_dict[k]*LOSS_WEIGHTS[k] for k in LOSS_TERMS])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rqvae_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # print("Step {} loss:".format(step), round(float(loss), 2))
        # print(", ".join(["{}: {}".format(k, round(float(step_loss_dict[k]), 2)) for k in LOSS_TERMS]))
        total_loss_dict = {k: total_loss_dict[k]+step_loss_dict[k] for k in LOSS_TERMS}
    
    total_loss = sum([total_loss_dict[k]*LOSS_WEIGHTS[k] for k in LOSS_TERMS])
    print("Epoch {} loss:".format(epoch), round(float(total_loss), 2))
    print(", ".join(["{}: {}".format(k, round(float(total_loss_dict[k]), 2)) for k in LOSS_TERMS]))