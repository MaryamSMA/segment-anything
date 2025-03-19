# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from tqdm import tqdm
import pennylane as qml

from .common import LayerNorm2d, MLPBlock

# ----------------------
# Quantum Module for Attention with Embedding Type Option
# ----------------------
import pennylane as qml
import torch
from torch import nn

class QuantumAttentionModule(nn.Module):
    def __init__(self, input_dim: int, n_qubits: int = 8, embedding_type: str = "amplitude"):
        """
        Quantum attention module using parameterized quantum circuits for feature extraction.
        This version uses:
        - Amplitude embedding for full feature representation
        - Parametrized RY, RX gates for learnable transformations
        - CNOT gates for entanglement
        - qml.probs() for full amplitude extraction
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.embedding_type = embedding_type

        self.reducer = nn.Linear(input_dim, 2 ** n_qubits)  # Match classical input to quantum qubit space
        
        #self.reducer = nn.Linear(input_dim, self.n_qubits)  # trial: Reduce to exactly 8 features
        #self.reducer = nn.Linear(input_dim, orig_shape[-1])  # trial: Ensures the same final dimension


        self.q_weights = nn.Parameter(torch.randn(n_qubits, 3,dtype=torch.float32))  # Three learnable params per qubit (RX, RY, RZ)
        
        self.dev = qml.device("lightning.qubit", wires=n_qubits)  # Use lightning.qubit for efficiency
        
        # **NEW: Projection layer to downsample quantum output**
        self.downsampler = nn.Linear(2 ** n_qubits, input_dim)  # Reduce from 256 → 64


    def quantum_circuit(self, x):
        """Quantum feature extraction circuit with entanglement and learnable parameters."""
        # 1. Amplitude Embedding
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True) #pad_with=0.0,

        # 2. Apply Trainable Parametrized Gates (RX, RY, RZ)
        for i in range(self.n_qubits):
            qml.RX(self.q_weights[i, 0], wires=i)
            qml.RY(self.q_weights[i, 1], wires=i)
            qml.RZ(self.q_weights[i, 2], wires=i)

        # 3. Apply Entanglement (CNOTs in a Ring Topology)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])  # Connect last to first (cyclic entanglement)

        # 4. Use qml.probs() to extract full probability vector (2^n_qubits outputs)
        return qml.probs(wires=range(self.n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit on reduced input and return extracted features."""
        x_reduced = self.reducer(x).float()  # Reduce classical input to 2^n_qubits
        qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
        #outputs = torch.stack([torch.tensor(qnode(sample)) for sample in x_reduced])
        outputs = torch.stack([qnode(sample).clone().detach() for sample in x_reduced])

        
        # 🔹 Ensure outputs match PyTorch's expected float32
        outputs = outputs.to(x.device).float()
        
        # 🔹 **Apply new downsampler to reduce 256 → 64**
        outputs = self.downsampler(outputs)  

        return outputs

# ----------------------
# Patch Embedding Module
# ----------------------
class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        return x

# ----------------------
# Attention Module with Optional Quantum Enhancement
# ----------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        use_quantum: bool = False,
        n_qubits: int = 4,
        embedding_type: str = "rotation"
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
        self.use_quantum = use_quantum
        if self.use_quantum:
            self.quantum_module = QuantumAttentionModule(head_dim, n_qubits=n_qubits, embedding_type=embedding_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        if self.use_quantum:
            orig_shape = q.shape  # Save original shape before quantum processing
            q_flat = q.reshape(-1, orig_shape[-1])  # Flatten before passing to quantum

            print("[DEBUG] Original q shape:", q.shape)
            print("[DEBUG] q_flat shape (before quantum):", q_flat.shape)

            q_transformed = self.quantum_module(q_flat)  # Apply Quantum Processing

            print("[DEBUG] q_transformed shape:", q_transformed.shape)
            print("[DEBUG] Expected orig_shape:", orig_shape)

            q = q_transformed.view(orig_shape)  # Reshape back

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

# ----------------------
# Transformer Block
# ----------------------
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_quantum: bool = False,
        n_qubits: int = 4,
        embedding_type: str = "rotation"
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_quantum=use_quantum,
            n_qubits=n_qubits,
            embedding_type=embedding_type
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

# ----------------------
# Modified ViT Encoder with Quantum-Enhanced Attention
# ----------------------
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        use_quantum: bool = False,
        n_qubits: int = 4,
        embedding_type: str = "rotation"
    ) -> None:
        """
        Args:
            use_quantum: If True, each transformer block’s attention uses quantum processing.
            n_qubits: Number of qubits to use in the quantum module.
            embedding_type: Which embedding to use in the quantum branch ("rotation" or "amplitude").
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_quantum=use_quantum,
                n_qubits=n_qubits,
                embedding_type=embedding_type
            )
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        print("Number of transformer blocks:", len(self.blocks))
        for i, blk in enumerate(self.blocks):
            print("Processing transformer block:", i + 1)
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

# ----------------------
# Helper functions for window partitioning and relative positional encoding
# ----------------------
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                                        size=max_rel_dist, mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int], k_size: Tuple[int, int]) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn

class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x
