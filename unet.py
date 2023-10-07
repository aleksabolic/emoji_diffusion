import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

from labml_helpers.module import Module
from transformers import BartTokenizer, BartModel

class Swish(Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):

        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb
    
class SentenceBlock(Module):
    def __init__(self, output_h: int, output_w: int, output_channels: int, embedding_dim: int = 1024):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        if embedding_dim != output_h * output_w:
            self.sentence_emb = nn.Linear(embedding_dim, output_h * output_w)
        else:
            self.sentence_emb = nn.Identity()
        self.sentence_act = Swish()
        self.sentence_conv = nn.Conv2d(1, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, sentence_embedding: torch.Tensor):
        """
        * `sentence_embedding` has shape `[batch_size, embedding_dim]`
        """
        # Reshape sentence embedding to shape `[batch_size, output_channels, output_h, output_w]`
        sentence_embedding = self.sentence_emb(sentence_embedding)
        sentence_embedding = self.sentence_act(sentence_embedding)
        sentence_embedding = sentence_embedding.view(sentence_embedding.shape[0], 1, self.output_h, self.output_w)
        sentence_embedding = self.sentence_conv(sentence_embedding)
        return sentence_embedding



class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        super().__init__()

        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor, sentence_embedding: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Multiply by the sentence embedding
        if sentence_embedding.shape[1] == h.shape[1]:
            h *= sentence_embedding
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, sentence_embedding: Optional[torch.Tensor] = None):
  
        _ = t
        _ = sentence_embedding
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, sentence_embedding: torch.Tensor):
        x = self.res(x, t, sentence_embedding)
        x = self.attn(x)
        return x


class UpBlock(Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, sentence_embedding: torch.Tensor):
        x = self.res(x, t, sentence_embedding)
        x = self.attn(x)
        return x


class MiddleBlock(Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, sentence_embedding: torch.Tensor):
        x = self.res1(x, t, sentence_embedding)
        x = self.attn(x)
        x = self.res2(x, t, sentence_embedding)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, sentence_embedding: torch.Tensor):
        _ = t
        _ = sentence_embedding
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        self.in_channels = 0
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, sentence_embedding: torch.Tensor):

        _ = t
        _ = sentence_embedding
        return self.conv(x)


class UNet(Module):

    def __init__(self, image_channels: int = 3, image_shape: Tuple[int, ...] = (64, 64), n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2,
                 embedding_dim: int = 1024,
                 device = 'cuda:0'):
  
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        #self.embedding_dim = self.text_encoder.config.d_model
        self.embedding_dim = embedding_dim

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # Height and width of the feature map
        output_h, output_w = image_shape

        temp_h, temp_w = output_h, output_w
        # Define all sentence blocks
        self.sentence_encoders = {}
        num_channels = n_channels
        for i in range(n_resolutions):           
            # Add new sentence encoder to the list of sentence encoders
            num_channels *= ch_mults[i]
            self.sentence_encoders[(temp_h, temp_w)] = SentenceBlock(temp_h, temp_w, num_channels, embedding_dim=self.embedding_dim).to(device)
            temp_h //= 2
            temp_w //= 2
            

        # First half of U-Net - decreasing resolution
        down = []
        self.down_shapes = []
        
        # Number of channels
        out_channels = in_channels = n_channels
        
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i] 
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                print(f"DownBlock: {in_channels} -> {out_channels} height: {output_h} width: {output_w}")
                
                in_channels = out_channels
                self.down_shapes.append((output_h, output_w))
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
                self.down_shapes.append((output_h, output_w))
                output_h //= 2
                output_w //= 2

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4)

        # Second half of U-Net - increasing resolution
        up = []
        self.up_shapes = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                self.up_shapes.append((output_h, output_w))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            self.up_shapes.append((output_h, output_w))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))
                self.up_shapes.append((output_h, output_w))
                output_h *= 2
                output_w *= 2

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, sentence_embedding: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        * `c` has shape `[batch_size, seq_len]`
        """

        t = self.time_emb(t)
        x = self.image_proj(x)

        # Get sentence embedding for each resolution
        sentence_embeddings = {}
        for shape, sentence_encoder in self.sentence_encoders.items():
            sentence_embeddings[shape] = sentence_encoder(sentence_embedding)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for i, m in enumerate(self.down):
            x = m(x, t, sentence_embeddings[self.down_shapes[i]])
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t, sentence_embeddings[self.down_shapes[-1]])

        # Second half of U-Net
        for idx, m in enumerate(self.up):
            if isinstance(m, Upsample):
                x = m(x, t, sentence_embeddings[self.up_shapes[idx]]) # doesnt matter what sentence embedding we use here
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t, sentence_embeddings[self.up_shapes[idx]])

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))