from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from xnav.module.projector.q_former import QFormerProjector


# class PointEncoder(nn.Module):
# 	"""
# 	Point encoder that projects a 3D pose (x, y, z, yaw) into a fixed-length
# 	token sequence using a small MLP followed by a Q-Former style projector.

# 	Contract:
# 	- Input points: (B, 4) where 4 == (x, y, z, yaw)
# 	- Two-layer MLP: (B, 4) -> (B, d)
# 	- Repeat embedding: (B, d) -> (B, L, d)
# 	- QFormer projector: (B, L, d_in=d) -> (B, L, d_out=d)
# 	- Output tokens: (B, L, d)

# 	Parameters
# 	----------
# 	embed_dim : int
# 		Final token dimension d.
# 	num_tokens : int
# 		Number of output tokens L.
# 	qformer_heads : int
# 		Number of attention heads for Q-Former (d_model must be divisible by heads).
# 	qformer_layers : int
# 		Number of stacked Q-Former layers.
# 	qformer_hidden_dim : int | None
# 		Internal Q-Former latent size; if None, defaults to embed_dim.
# 	mlp_hidden_multiplier : int
# 		Width multiplier for hidden layer in the two-layer MLP; hidden = multiplier * embed_dim.
# 	dropout : float
# 		Dropout probability for Q-Former attention/MLP.
# 	use_query_self_attn : bool
# 		Whether to include query self-attention inside Q-Former layers.
# 	layer_norm : bool
# 		Whether to use LayerNorm inside Q-Former.
# 	use_x_positional_encoding : bool
# 		Whether to add sinusoidal positional encoding to projector K/V.
# 	use_query_positional_encoding : bool
# 		Whether to add learnable positional encoding to projector queries.
# 	"""

# 	def __init__(
# 		self,
# 		embed_dim: int = 128,
# 		num_tokens: int = 16,
# 		qformer_heads: int = 8,
# 		qformer_layers: int = 2,
# 		qformer_hidden_dim: int | None = None,
# 		mlp_hidden_multiplier: int = 2,
# 		dropout: float = 0.0,
# 		use_query_self_attn: bool = True,
# 		layer_norm: bool = True,
# 		use_x_positional_encoding: bool = True,
# 		use_query_positional_encoding: bool = True,
# 	) -> None:
# 		super().__init__()

# 		self.embed_dim = embed_dim
# 		self.num_tokens = num_tokens

# 		# Two-layer MLP: 4 -> hidden -> d
# 		hidden = mlp_hidden_multiplier * embed_dim
# 		self.mlp = nn.Sequential(
# 			nn.Linear(4, hidden),
# 			nn.SiLU(),
# 			nn.Linear(hidden, embed_dim),
# 		)

# 		# Q-Former projector
# 		d_model = qformer_hidden_dim if qformer_hidden_dim is not None else embed_dim
# 		self.projector = QFormerProjector(
# 			d_model=d_model,
# 			num_queries=num_tokens,            # we want exactly L output tokens
# 			num_heads=qformer_heads,
# 			num_layers=qformer_layers,
# 			dropout=dropout,
# 			layer_norm=layer_norm,
# 			use_query_self_attn=use_query_self_attn,
# 			use_x_positional_encoding=use_x_positional_encoding,
# 			use_query_positional_encoding=use_query_positional_encoding,
# 			input_dim=embed_dim,                # input tokens to projector are (B, L, d)
# 			output_dim=embed_dim,               # final output tokens are (B, L, d)
# 		)

# 	def emb_points(self, points: torch.Tensor) -> torch.Tensor:
# 		"""
# 		Embed points into a sequence of tokens.

# 		Args:
# 			points: Tensor of shape (B, 4)

# 		Returns:
# 			Tensor of shape (B, L, d)
# 		"""
# 		if points.dim() != 2 or points.size(-1) != 4:
# 			raise ValueError(f"points must have shape (B, 4); got {tuple(points.shape)}")

# 		B = points.size(0)
# 		d = self.embed_dim
# 		L = self.num_tokens

# 		# Two-layer MLP to (B, d)
# 		emb = self.mlp(points)  # (B, d)

# 		# Repeat to (B, L, d)
# 		x = emb.unsqueeze(1).expand(B, L, d).contiguous()

# 		# Q-Former projector to produce (B, L, d)
# 		y = self.projector(x)  # (B, L, d)
# 		return y

# 	def forward(self, points: torch.Tensor) -> torch.Tensor:
# 		return self.emb_points(points)


# Revised simpler version without Q-Former projector
# We think the Q-Former is overkill for just encoding 4D points into tokens
class PointEncoder(nn.Module):
	"""
	Point encoder that projects a 3D pose (x, y, z, yaw) into a fixed-length
	token sequence using a small MLP followed by a Q-Former style projector.

	Contract:
	- Input points: (B, 4) where 4 == (x, y, z, yaw)
	- Two-layer MLP: (B, 4) -> (B, d)
	- Repeat embedding: (B, d) -> (B, L, d)
	- QFormer projector: (B, L, d_in=d) -> (B, L, d_out=d)
	- Output tokens: (B, L, d)

	Parameters
	----------
	embed_dim : int
		Final token dimension d.
	num_tokens : int
		Number of output tokens L.
	qformer_heads : int
		Number of attention heads for Q-Former (d_model must be divisible by heads).
	qformer_layers : int
		Number of stacked Q-Former layers.
	qformer_hidden_dim : int | None
		Internal Q-Former latent size; if None, defaults to embed_dim.
	mlp_hidden_multiplier : int
		Width multiplier for hidden layer in the two-layer MLP; hidden = multiplier * embed_dim.
	dropout : float
		Dropout probability for Q-Former attention/MLP.
	use_query_self_attn : bool
		Whether to include query self-attention inside Q-Former layers.
	layer_norm : bool
		Whether to use LayerNorm inside Q-Former.
	use_x_positional_encoding : bool
		Whether to add sinusoidal positional encoding to projector K/V.
	use_query_positional_encoding : bool
		Whether to add learnable positional encoding to projector queries.
	"""

	def __init__(
		self,
		embed_dim: int = 128,
		num_tokens: int = 16,
		qformer_heads: int = 8,
		qformer_layers: int = 2,
		qformer_hidden_dim: int | None = None,
		mlp_hidden_multiplier: int = 2,
		dropout: float = 0.0,
		use_query_self_attn: bool = True,
		layer_norm: bool = True,
		use_x_positional_encoding: bool = True,
		use_query_positional_encoding: bool = True,
	) -> None:
		super().__init__()

		self.embed_dim = embed_dim
		self.num_tokens = num_tokens

		# Two-layer MLP: 4 -> hidden -> d
		hidden = mlp_hidden_multiplier * embed_dim
		self.mlp = nn.Sequential(
			nn.Linear(4, hidden),
			nn.SiLU(),
			nn.Linear(hidden, embed_dim),
		)

	def emb_points(self, points: torch.Tensor) -> torch.Tensor:
		"""
		Embed points into a sequence of tokens.

		Args:
			points: Tensor of shape (B, 4)

		Returns:
			Tensor of shape (B, L, d)
		"""
		if points.dim() != 2 or points.size(-1) != 4:
			raise ValueError(f"points must have shape (B, 4); got {tuple(points.shape)}")

		B = points.size(0)
		d = self.embed_dim
		L = self.num_tokens

		# Two-layer MLP to (B, d)
		emb = self.mlp(points)  # (B, d)

		# Repeat to (B, L, d)
		x = emb.unsqueeze(1).expand(B, L, d).contiguous()

		return x

	def forward(self, points: torch.Tensor) -> torch.Tensor:
		return self.emb_points(points)


if __name__ == "__main__":
	# Simple sanity test
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(0)

	B = 3
	L = 10
	d = 64
	heads = 8
	layers = 2
	q_dim = 64  # can be different from d; must be divisible by heads

	encoder = PointEncoder(
		embed_dim=d,
		num_tokens=L,
		qformer_heads=heads,
		qformer_layers=layers,
		qformer_hidden_dim=q_dim,
		dropout=0.1,
		use_query_self_attn=True,
	).to(device).eval()

	pts = torch.randn(B, 4, device=device)
	out = encoder(pts)
	print(f"points: {pts.shape} -> tokens: {out.shape} (expect: {(B, L, d)})")

