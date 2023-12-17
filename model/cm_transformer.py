import torch

from compressive_transformer_pytorch import CompressiveTransformer

model = CompressiveTransformer(
    num_tokens = 20000,
    emb_dim = 128,                 # embedding dimensions, embedding factorization from Albert paper
    dim = 512,
    depth = 12,
    seq_len = 1024,
    mem_len = 1024,                # memory length
    cmem_len = 1024 // 4,          # compressed memory buffer length
    cmem_ratio = 4,                # compressed memory ratio, 4 was recommended in paper
    reconstruction_loss_weight = 1,# weight to place on compressed memory reconstruction loss
    attn_dropout = 0.1,            # dropout post-attention
    ff_dropout = 0.1,              # dropout in feedforward
    attn_layer_dropout = 0.1,      # dropout for attention layer output
    gru_gated_residual = True,     # whether to gate the residual intersection, from 'Stabilizing Transformer for RL' paper
    mogrify_gru = False,           # experimental feature that adds a mogrifier for the update and residual before gating by the GRU
    memory_layers = range(6, 13),  # specify which layers to use long-range memory, from 'Do Transformers Need LR Memory' paper
    ff_glu = True                  # use GLU variant for feedforward
)

inputs = torch.randint(0, 256, (1, 2048))
masks = torch.ones_like(inputs).bool()

segments = inputs.reshape(1, -1, 1024).transpose(0, 1)
masks = masks.reshape(1, -1, 1024).transpose(0, 1)

logits, memories, aux_loss = model(segments[0], mask = masks[0])
logits,        _, aux_loss = model(segments[1], mask = masks[1], memories = memories)

# memories is a named tuple that contains the memory (mem) and the compressed memory (cmem)