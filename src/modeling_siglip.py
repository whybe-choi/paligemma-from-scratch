from typing import Optional, Tuple

import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,  # 임베딩 벡터의 사이즈
        intermediate_size=3072,  # FFN의 중간 레이어 사이즈
        num_hidden_layers=12,  # 트랜스포머 블록의 개수
        num_attention_heads=12,  # 어텐션 헤드의 개수
        num_channels=3,  # 입력 이미지의 채널 수(RGB)
        image_size=224,  # 입력 이미지의 크기 -> 이미지 입력 시 해당 사이즈로 resize
        patch_size=16,  # 패치의 크기 -> 입력된 이미지는 (patch_size x patch_size)로 분할
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,  # 출력할 이미지 토큰의 개수
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # This indicates no padding is added
        )

        # 전체 이미지에서 생성되는 패치의 개수 (예: 224x224 이미지에서 16x16 패치 -> 14x14=196개 패치)
        self.num_patches = (config.image_size // config.patch_size) ** 2
        # 각 패치가 어느 위치에 해당하는지를 나타내야 하므로 패치의 수만큼의 위치가 필요함
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 위치 인덱스를 미리 생성하여 버퍼로 등록
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # (batch_size, channels, height, width)
        _, _, height, width = pixel_values.shape
        # convolve the 'patch_size' kernel over the image, with no overlapping patches since
        # the output of the convolution will have shape (batch_size, embed_dim, num_patches_h, num_patches_w)
        # whe  re num_patches_h = height // patch_size and num_patches_w = width // patch_size
        patch_embed = self.patch_embedding(pixel_values)

        # (batch_size, embed_dim, num_patches_h, num_patches_w) -> (batch_size, embed_dim, num_patches)
        # where num_patches = num_patches_h * num_patches_w
        embeddings = patch_embed.flatten(2)
        # (batch_size, embed_dim, num_patches) -> (batch_size, num_patches, embed_dim)
        embeddings = embeddings.transpose(1, 2)
        # add position embeddings to each patch. each positional encoding is a vector of size (embed_dim)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # (batch_size, num_patches, embed_dim)
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (batch_size, n_patches, hidden_size) -> (batch_size, n_patches, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        # hidden_states: (batch_size, n_patches, intermediate_size)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # (batch_size, n_patches, intermediate_size) -> (batch_size, n_patches, hidden_size)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: (batch_size, n_patches, embed_dim)
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: (batch_size, n_patches, embed_dim)
        query_states = self.q_proj(hidden_states)
        # key_states: (batch_size, n_patches, embed_dim)
        key_states = self.k_proj(hidden_states)
        # value_states: (batch_size, n_patches, embed_dim)
        value_states = self.v_proj(hidden_states)
        # query_states: (batch_size, n_patches, num_heads, head_dim) -> (batch_size, num_heads, n_patches, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights : (batch_size, num_heads, num_patches, num_patches)
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is {attn_weights.size()}"
            )

        # apply softmax row-wise. attn_weights: (batch_size, num_heads, num_patches, num_patches)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # apply dropout only during training
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        # multiply the attention weights by value states. attn_output: (batch_size, num_heads, num_patches, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}"
            )

        # (batch_size, num_heads, n_patches, head_dim) -> (batch_size, n_patches, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch_size, n_patches, num_heads, head_dim) -> (batch_size, n_patches, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # (batch_size, n_patches, embed_dim) -> (batch_size, n_patches, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: (batch_size, n_patches, embed_dim)
        residual = hidden_states
        # (batch_size, n_patches, embed_dim) -> (batch_size, n_patches, embed_dim)
        hidden_states = self.layer_norm1(hidden_states)
        # (batch_size, n_patches, embed_dim) -> (batch_size, n_patches, embed_dim)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # (batch_size, n_patches, embed_dim)
        hidden_states = hidden_states + residual
        # residual: (batch_size, n_patches, embed_dim)
        residual = hidden_states
        # (batch_size, n_patches, embed_dim) -> (batch_size, n_patches, embed_dim)
        hidden_states = self.layer_norm2(hidden_states)
        # (batch_size, n_patches, embed_dim) -> (batch_size, n_patches, embed_dim)
        hidden_states = self.mlp(hidden_states)
        # (batch_size, n_patches, embed_dim)
        hidden_states = hidden_states + residual

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds: (batch_size, n_patches, embed_dim)
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            # (batch_size, n_patches, embed_dim) -> (batch_size, n_patches, embed_dim)
            hidden_states = encoder_layer(hidden_states=hidden_states)

        return hidden_states  # (batch_size, n_patches, embed_dim)


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (batch_size, channels, height, width) -> (batch_size, num_patches, embed_dim)
        hidden_states = self.embeddings(pixel_values=pixel_values)
        # 패치 임베딩과 위치 인코딩을 더한 후 트랜스포머 인코더에 통과
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state  # (batch_size, num_patches, embed_dim)


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # pixel_values: (batch_size, channels, height, width)
        # (batch_size, channels, height, width) -> (batch_size, num_patches, embed_dim)
        return self.vision_model(pixel_values=pixel_values)
