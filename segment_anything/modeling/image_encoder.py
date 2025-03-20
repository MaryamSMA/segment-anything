# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

# ------- Bring in from your local 'common.py' -------
from .common import LayerNorm2d, MLPBlock

# ------- PennyLane Imports -------
import pennylane as qml


###############################################################################
#                         Quantum Circuit with GPU Device
###############################################################################
class QuantumCircuitAnsatz(nn.Module):
    """
    Trainable quantum ansatz:
      - Expects input dimension = 2^n_qubits (already L2-normalized).
      - Applies amplitude embedding, then N layers of single-qubit rotations + ring CNOT.
      - Returns the full statevector (complex).
    """

    def __init__(self, n_qubits: int, n_layers: int = 2, use_lightning_gpu: bool = False):
        """
        Args:
          n_qubits (int): number of qubits
          n_layers (int): number of repeated rotation + entangling layers
          use_lightning_gpu (bool): If True, try 'lightning.gpu' device. Otherwise 'lightning.qubit'.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # We'll store [n_layers, n_qubits, 3] rotation angles (RX, RY, RZ)
        init_shape = (n_layers, n_qubits, 3)
        self.params = nn.Parameter(0.01 * torch.randn(init_shape))

        # Choose device name
        dev_name = "lightning.gpu" if use_lightning_gpu else "lightning.qubit"

        print(f"[QuantumCircuitAnsatz] Using PennyLane device='{dev_name}' with n_qubits={n_qubits}")
        try:
            self.dev = qml.device(dev_name, wires=n_qubits)
        except Exception as e:
            print(f"[WARNING] Could not initialize '{dev_name}'. Falling back to 'default.qubit'. Error: {e}")
            self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build a QNode referencing self.forward_circuit
        self.qnode = qml.QNode(self.forward_circuit, self.dev, interface="torch")

    def forward_circuit(self, inputs):
        """
        The QNode circuit: 
          1) amplitude-encode 'inputs' 
          2) apply param gates 
          3) ring entangle 
          4) return full statevector
        """
        # 1) amplitude embed
        qml.templates.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=False)

        # 2) repeated layers
        for layer_idx in range(self.n_layers):
            for qubit_idx in range(self.n_qubits):
                rx_angle = self.params[layer_idx, qubit_idx, 0]
                ry_angle = self.params[layer_idx, qubit_idx, 1]
                rz_angle = self.params[layer_idx, qubit_idx, 2]
                qml.RX(rx_angle, wires=qubit_idx)
                qml.RY(ry_angle, wires=qubit_idx)
                qml.RZ(rz_angle, wires=qubit_idx)

            # ring entangling
            for qubit_idx in range(self.n_qubits):
                next_qubit = (qubit_idx + 1) % self.n_qubits
                qml.CNOT(wires=[qubit_idx, next_qubit])

        return qml.state()

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        """
        Forward method:
          x_1d: shape (2^n_qubits,) already L2-normalized
        Returns complex statevector of shape (2^n_qubits,).
        """
        state = self.qnode(x_1d)
        return state


def apply_learnable_quantum_ansatz(
    patch_tensor: torch.Tensor,  # [batch, embed_dim]
    q_model: QuantumCircuitAnsatz
) -> torch.Tensor:
    """
    For each patch in 'patch_tensor', run amplitude encoding + trainable ansatz.
    Returns the real part of the final state (dim=2^n_qubits).
    """
    B, dim = patch_tensor.shape
    required_dim = 2 ** q_model.n_qubits

    if dim != required_dim:
        print(f"[Quantum WARNING] embed_dim={dim} != 2^n_qubits={required_dim}. Returning original patch_tensor.")
        return patch_tensor

    # allocate output
    out = torch.zeros_like(patch_tensor, dtype=patch_tensor.dtype, device=patch_tensor.device)

    # debug prints
    print(f"[DEBUG] apply_learnable_quantum_ansatz: B={B}, patch_dim={dim}, n_qubits={q_model.n_qubits}")
    # We won't spam every patch, but let's show the first patch's first few values:
    print(f"         Example first patch input (first 8 values): {patch_tensor[0,:8]}")

    for i in range(B):
        vec = patch_tensor[i]
        norm = torch.norm(vec)
        if norm > 1e-9:
            vec = vec / norm  # amplitude embedding requires normalized input

        state = q_model(vec)        # shape (dim,) complex
        out[i] = state.real         # discard imaginary part for dimension match

    # print final sample
    print(f"         Example first patch output (first 8 values): {out[0,:8]}")
    return out


###############################################################################
#                                ImageEncoderViT
###############################################################################
class ImageEncoderViT(nn.Module):
    """
    A SAM-like ViT encoder with an *optional* quantum patch-embedding.
    If use_quantum=True, each patch is amplitude-encoded with 'n_qubits' => embed_dim=2^n_qubits.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
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

        # quantum args
        use_quantum: bool = False,
        n_qubits: int = 8,
        n_layers_q: int = 2,
        use_lightning_gpu: bool = False,
    ) -> None:
        """
        Args:
          - If use_quantum=True, embed_dim MUST = 2^n_qubits for amplitude encoding
          - n_layers_q: # of repeated layers in the quantum circuit
          - use_lightning_gpu: If True, tries to use the 'lightning.gpu' device
        """
        super().__init__()
        self.img_size = img_size
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.n_layers_q = n_layers_q
        self.use_lightning_gpu = use_lightning_gpu

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # Quantum model if requested
        if self.use_quantum:
            self.q_model = QuantumCircuitAnsatz(
                n_qubits=n_qubits,
                n_layers=n_layers_q,
                use_lightning_gpu=use_lightning_gpu
            )
        else:
            self.q_model = None

        # Optional absolute position embedding
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            pe_h = img_size // patch_size
            pe_w = img_size // patch_size
            self.pos_embed = nn.Parameter(torch.zeros(1, pe_h, pe_w, embed_dim))

        # Create the ViT blocks (classical)
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
            )
            self.blocks.append(block)

        # Neck
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) Patch embed => [B, Hp, Wp, C]
        2) If quantum, run amplitude embedding on each patch => new [B, Hp, Wp, C]
        3) Add pos_embed
        4) Pass ViT blocks
        5) Neck -> final [B, out_chans, Hp, Wp]
        """
        # Step 1: classical patch embedding
        x = self.patch_embed(x)  # => [B, Hp, Wp, embed_dim]
        B, Hp, Wp, C = x.shape
        print(f"[DEBUG] After patch_embed: {x.shape} (B={B}, Hp={Hp}, Wp={Wp}, C={C})")

        # Step 2: quantum on each patch embedding
        if self.use_quantum and self.q_model is not None:
            x_2d = x.view(B * Hp * Wp, C)
            x_2d_q = apply_learnable_quantum_ansatz(x_2d, self.q_model)
            x = x_2d_q.view(B, Hp, Wp, C)
            print(f"[DEBUG] After quantum step: {x.shape}")

        # Step 3: pos embed if used
        if self.pos_embed is not None:
            x = x + self.pos_embed
            print(f"[DEBUG] After pos_embed addition: {x.shape}")

        # Step 4: pass through blocks
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            # debug info
            # print(f"[DEBUG] After block {idx+1}: {x.shape}")

        # Step 5: neck => reshape to [B, C, Hp, Wp], run conv
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        print(f"[DEBUG] After neck: {x.shape}")
        return x


###############################################################################
#                         Supporting Classes
###############################################################################
class PatchEmbed(nn.Module):
    """
    Standard patch embedding with a conv2d that lumps patches into channels.
    """
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, in_chans, H, W]
        Output: [B, H_p, W_p, embed_dim]
        """
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # => [B, H_p, W_p, embed_dim]
        return x


class Block(nn.Module):
    """Transformer block with optional window attention (classical)."""
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
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => [B, H, W, C]
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


class Attention(nn.Module):
    """Classical multi-head attention with optional relative pos."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Need input_size if using relative pos."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape  # x => [B, H, W, C]
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B*self.num_heads, H*W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x_out = (attn @ v).view(B, self.num_heads, H, W, -1)
        x_out = x_out.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x_out = self.proj(x_out)
        return x_out


###############################################################################
#           Utility fns for window partition & relative positional encoding
###############################################################################
def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp//window_size, window_size, Wp//window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows, window_size: int, pad_hw: Tuple[int,int], hw: Tuple[int,int]):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp//window_size, Wp//window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, Hp, Wp, -1)
    if Hp>H or Wp>W:
        x = x[:,:H,:W,:].contiguous()
    return x


def get_rel_pos(q_size:int, k_size:int, rel_pos:torch.Tensor)->torch.Tensor:
    max_rel_dist = 2*max(q_size,k_size)-1
    if rel_pos.shape[0]!=max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0,2,1),
            size=max_rel_dist,mode="linear"
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1,0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:,None]*max(k_size/q_size,1.0)
    k_coords = torch.arange(k_size)[None,:]*max(q_size/k_size,1.0)
    relative_coords = (q_coords - k_coords) + (k_size-1)*max(q_size/k_size,1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn:torch.Tensor,q:torch.Tensor,rel_pos_h:torch.Tensor,rel_pos_w:torch.Tensor,q_size:Tuple[int,int],k_size:Tuple[int,int]):
    q_h,q_w = q_size
    k_h,k_w = k_size
    Rh = get_rel_pos(q_h,k_h,rel_pos_h)
    Rw = get_rel_pos(q_w,k_w,rel_pos_w)
    B,_,dim = q.shape
    r_q = q.reshape(B,q_h,q_w,dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk",r_q,Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk",r_q,Rw)
    attn = attn.view(B,q_h,q_w,k_h,k_w) + rel_h[:,:,:, :,None] + rel_w[:,:,:,None,:]
    attn = attn.view(B,q_h*q_w,k_h*k_w)
    return attn