# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
import pennylane as qml

from .common import LayerNorm2d, MLPBlock

# ----------------------
# Quantum Module for Attention with Embedding Type Option
# ----------------------
class QuantumAttentionModule(nn.Module):
    def __init__(self, input_dim: int, n_qubits: int = 4, embedding_type: str = "rotation"):
        """
        Processes an input feature vector of dimension input_dim by reducing it
        to n_qubits, applying a quantum circuit with either rotation‐based or amplitude embedding,
        and then expanding it back.
        
        Args:
            input_dim: Dimension of the input token (e.g., head dimension).
            n_qubits: Number of qubits to use.
            embedding_type: Which embedding method to use. Options are:
                "rotation" (default) – use RY/RZ rotations,
                "amplitude" – use amplitude embedding.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.embedding_type = embedding_type
        print("[QuantumAttentionModule] Using embedding type:", self.embedding_type)
        #self.reducer = nn.Linear(input_dim, n_qubits)
        if self.embedding_type == "amplitude":
            self.reducer = nn.Linear(input_dim, 2 ** n_qubits)
        else:
            self.reducer = nn.Linear(input_dim, n_qubits)
        self.q_weights = nn.Parameter(torch.randn(n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.expander = nn.Linear(n_qubits, input_dim)
        # Flag to print normalization debug info only once per forward pass.
        self._printed_norm_debug = False

    def quantum_circuit(self, x, weights):
        if self.embedding_type == "rotation":
            # Rotation-based embedding: apply RY and RZ gates.
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            for i in range(self.n_qubits):
                qml.RZ(weights[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        elif self.embedding_type == "amplitude":
            # Amplitude embedding expects a normalized vector of length 2^n.
            required_length = 2 ** self.n_qubits
            if x.shape[0] < required_length:
                padding = required_length - x.shape[0]
                x = torch.cat([x, torch.zeros(padding, dtype=x.dtype, device=x.device)], dim=0)
            
            # ------------------ ADDED: Begin simple smoothing enhancement ------------------
            # Instead of using a gradient filter (Sobel), we apply a simple 1D average pooling 
            # to create a smoothed version of the vector. This can help reduce noise and capture 
            # prominent transitions without the artifacts of a gradient filter.
            x_unsqueezed = x.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, required_length]
            kernel_size = 3  # You can adjust this kernel size as needed
            smoothed = F.avg_pool1d(x_unsqueezed, kernel_size=kernel_size, stride=1, padding=1)
            smoothed = smoothed.squeeze()  # shape: [required_length]
            # Combine the original vector with its smoothed version using a weighted sum.
            alpha = 0.5  # ADJUST: Experiment with this weight to control the influence of smoothing.
            x = x + alpha * smoothed
            # ------------------ ADDED: End simple smoothing enhancement ------------------
            
            # Normalize the vector (using L2 norm).
            norm = x.norm(p=2)
            if not self._printed_norm_debug:
                print("Before normalization, L2 norm:", norm.item(),
                      "and stats:", x.min().item(), x.max().item(), x.mean().item())
                self._printed_norm_debug = True
            if norm > 0:
                x_normalized = x / norm
            else:
                x_normalized = x
            qml.AmplitudeEmbedding(x_normalized, wires=range(self.n_qubits), normalize=False)
        else:
            raise ValueError(f"Unsupported embedding_type: {self.embedding_type}")
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reset the normalization debug flag for this forward pass.
        self._printed_norm_debug = False

        print("[QuantumAttentionModule] Input image shape:", x.shape) # Debug: original image dimensions
        x_reduced = self.reducer(x).float()
        print("[QuantumAttentionModule] Reduced tensor shape:", x_reduced.shape) # Expect (B, H_patch, W_patch, embed_dim)

        # (Optional) Print summary stats of x_reduced if needed:
        print("[QuantumAttentionModule] Reduced tensor stats: min =", x_reduced.min().item(),
              "max =", x_reduced.max().item(), "mean =", x_reduced.mean().item())

        outputs = []
        qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
        for idx, sample in enumerate(x_reduced):
            sample = sample.float()
            # Optionally, print sample stats for the first sample.
            if idx == 0:
                print("[QuantumAttentionModule] First sample before quantum circuit:", sample)
            q_out = qnode(sample, self.q_weights.float())
            if idx == 0:
                quantum_out = torch.tensor(q_out)
                print("[QuantumAttentionModule] quantum_out shape (first sample):", quantum_out.shape)
                # Optionally, print quantum_out values for debugging.
                print("[QuantumAttentionModule] quantum_out values (first sample):", quantum_out)
            outputs.append(torch.stack(q_out))
        quantum_features = torch.stack(outputs)

        # Optionally, print shape and stats of quantum_features
        print("[QuantumAttentionModule] Quantum features shape before expansion:", quantum_features.shape)

        expanded = self.expander(quantum_features.float())
        print("[QuantumAttentionModule] Expanded tensor shape:", expanded.shape)
        return expanded

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
            orig_shape = q.shape
            q_flat = q.reshape(-1, orig_shape[-1])
            q_transformed = self.quantum_module(q_flat)
            q = q_transformed.view(orig_shape)
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
