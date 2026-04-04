import torch
import torch.nn as nn
# import torch.nn.functional as F
# from einops import repeat

# from networks_attn import T_Transformer
class CLIP_TXT_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIP)
        # print(model)

        # self.learned_global_token = nn.Parameter(torch.randn(512))

        # # self.project_in = nn.Linear(clip.dim_latent, dim) if clip.dim_latent != dim else nn.Identity()

        # self.t_transformer = T_Transformer(
        #     dim = 512,
        #     depth = 1,
        # )
        for name, param in self.named_parameters():
            if name not in ['learned_global_token','project_in','t_transformer']:
                param.requires_grad = False

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        # copied from clip.model.encode_text
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return sent_emb, x
    
    # def forward_no(self, text):
    #     # copied from clip.model.encode_text
    #     x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    #     x = x + self.positional_embedding.type(self.dtype)
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x).type(self.dtype)
    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     text_encodings = x
    #     mask = (text_encodings != 0.).any(dim = -1)

    #     # text_encodings = self.project_in(text_encodings)

    #     mask_with_global = F.pad(mask, (1, 0), value = True)

    #     batch = text_encodings.shape[0]
    #     global_tokens = repeat(self.learned_global_token, 'd -> b d', b = batch)

    #     # change 
    #     text_encodings = torch.cat([global_tokens.unsqueeze(0), text_encodings], dim=0)
    #     text_encodings = self.transformer(text_encodings, mask=mask_with_global)
    #     global_tokens = text_encodings[0]
    #     text_encodings = text_encodings[1:]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     # sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    #     return global_tokens, text_encodings