import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

# Local imports from SAM's common
from .common import LayerNorm2d, MLPBlock

import pennylane as qml

###############################################################################
#                             Partial Quantum Module
###############################################################################
class PartialQuantumModule(nn.Module):
    """
    A "bottleneck" or "side branch" module that:
      - Takes in [B, H, W, embed_dim] (1280 for vit_h)
      - Splits out 'quantum_dim' channels (must be 2^n_qubits)
      - Runs a quantum circuit on that subset
      - Replaces that subset in the original tensor
      => Output shape remains [B, H, W, embed_dim]
    """
    def __init__(self, quantum_dim=256, n_qubits=8, n_layers=1, device_name="lightning.qubit"):
        super().__init__()
        self.quantum_dim = quantum_dim  # e.g. 256 => 2^8
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # If you want a trainable quantum circuit:
        # Store rotation angles as a parameter: [n_layers, n_qubits, 3]
        init_shape = (n_layers, n_qubits, 3)
        self.params = nn.Parameter(0.01 * torch.randn(init_shape))

        # Create a PennyLane device
        try:
            self.dev = qml.device(device_name, wires=n_qubits)
            print(f"[PartialQuantumModule] Using device={device_name}, n_qubits={n_qubits}")
        except Exception as e:
            print(f"[WARNING] Could not init device '{device_name}', fallback to default.qubit. Error: {e}")
            self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build the QNode
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

    def circuit(self, inputs):
        """
        1) Amplitude-embed 'inputs' (dimension=2^n_qubits)
        2) Apply 'n_layers' rotation + ring entanglement
        3) Return statevector (here, qml.probs for real output)
        """
        # Set normalize=True to let PennyLane handle any norm or zero-vector issues
        qml.templates.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)

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

        return qml.probs(wires=range(self.n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [B, H, W, C].
        We'll slice out the first 'quantum_dim' channels => shape [B, H, W, quantum_dim].
        Flatten => [B*H*W, quantum_dim], run amplitude encoding in a loop => replace back.
        """
        print(f"[DEBUG] PartialQuantumModule.forward called. Input shape = {x.shape}")

        B, H, W, C = x.shape

        # Optionally keep the amplitude-encoding check:
        required_dim = 2 ** self.n_qubits
        assert self.quantum_dim == required_dim, "quantum_dim must match 2^n_qubits for amplitude."

        # 1) Slice out the quantum subset: x_sub => shape [B, H, W, quantum_dim]
        x_sub = x[..., : self.quantum_dim]
        # The remainder => shape [B, H, W, C - quantum_dim]
        x_rem = x[..., self.quantum_dim :]

        print(f"[DEBUG] x_sub shape = {x_sub.shape}, x_rem shape = {x_rem.shape}")

        # 2) Flatten x_sub => [B*H*W, quantum_dim]
        BHW = B * H * W
        x_sub_2d = x_sub.reshape(BHW, self.quantum_dim)

        # 3) For each row => run amplitude encoding + quantum circuit
        out_sub_2d = torch.zeros_like(x_sub_2d, device=x_sub_2d.device, dtype=x_sub_2d.dtype)

        for i in range(BHW):
            vec = x_sub_2d[i]
            # We'll skip the manual norm-check:
            # norm = torch.norm(vec)
            # if norm > 1e-9:
            #    vec = vec / norm

            # Because we set normalize=True above, PennyLane auto-normalizes
            state = self.qnode(vec)
            out_sub_2d[i] = state.real

        # 4) Reshape back => [B, H, W, quantum_dim]
        out_sub = out_sub_2d.view(B, H, W, self.quantum_dim)

        # 5) Re-combine => [B, H, W, C]
        x_new = torch.cat([out_sub, x_rem], dim=-1)
        return x_new

###############################################################################
#                          ImageEncoderViT (with partial Q)
###############################################################################
class ImageEncoderViT(nn.Module):
    """
    By default, we keep embed_dim=1280 for vit_h, or 768 for vit_b, etc.
    We can insert a partial quantum module that handles quantum_dim channels.
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1280,  # for vit_h
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (7, 15, 23, 31),

        # partial quantum injection
        use_partial_quantum: bool = True,
        quantum_dim: int = 256,        # must be 2^n_qubits
        n_qubits: int = 8,
        n_layers_q: int = 1,
        quantum_device: str = "lightning.qubit"
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_partial_quantum = use_partial_quantum
        self.quantum_dim = quantum_dim
        self.n_qubits = n_qubits
        self.n_layers_q = n_layers_q
        self.quantum_device = quantum_device
        self.patch_size = patch_size

        # Debug prints: confirm partial quantum or classical
        if self.use_partial_quantum:
            fraction = 100.0 * self.quantum_dim / self.embed_dim
            print(f"[DEBUG] PartialQuantum: Enabled => quantum_dim={self.quantum_dim} / embed_dim={self.embed_dim}")
            print(f"[DEBUG] That's {fraction:.2f}% of channels.")
            print(f"[DEBUG] n_qubits={self.n_qubits} => 2^{self.n_qubits}={2**self.n_qubits}")
        else:
            print("[DEBUG] PartialQuantum: OFF (classical).")

        # patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # partial quantum module (side branch)
        if self.use_partial_quantum:
            self.partial_q = PartialQuantumModule(
                quantum_dim=self.quantum_dim,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers_q,
                device_name=self.quantum_device
            )
        else:
            self.partial_q = None

        # absolute position embedding
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            pe_h = img_size // patch_size
            pe_w = img_size // patch_size
            self.pos_embed = nn.Parameter(torch.zeros(1, pe_h, pe_w, embed_dim))

        # Build the blocks
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
        # 1) Patch embed => [B, Hp, Wp, embed_dim]
        x = self.patch_embed(x)
        B, Hp, Wp, C = x.shape

        # 2) partial quantum injection if enabled
        if self.partial_q is not None:
            x = self.partial_q(x)  # still [B, Hp, Wp, embed_dim]

        # 3) add absolute pos embed
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # 4) pass blocks
        for blk in self.blocks:
            x = blk(x)

        # 5) neck => [B, out_chans, Hp, Wp]
        x = x.permute(0,3,1,2)
        x = self.neck(x)
        return x

###############################################################################
#                                PatchEmbed
###############################################################################
class PatchEmbed(nn.Module):
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
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x

###############################################################################
#                                Block / Attention
###############################################################################
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
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

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

def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size:int, k_size:int, rel_pos:torch.Tensor)->torch.Tensor:
    max_rel_dist = 2 * max(q_size, k_size) - 1
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear"
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor,
                           rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int], k_size: Tuple[int, int]) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn
