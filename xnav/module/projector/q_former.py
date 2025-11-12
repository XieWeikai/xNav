"""Q-Former style projector.

This module implements a lightweight projector that uses a set of learnable
queries to attend over an input sequence of token embeddings (cross-attention),
producing a fixed number of output query embeddings.

Reference idea: Q-Former (BLIP-2). Here we provide a compact, self-contained
version that fits the following contract:

- Learnable queries: shape (n, d)
- Input tokens:     shape (b, L, d)
- Output:           shape (b, n, d)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from xnav.module.action_head.action_encoder import SinusoidalPositionalEncoding


# Reuse the shared SinusoidalPositionalEncoding from the action encoder module.


class _QFormerLayer(nn.Module):
	"""One Transformer-like layer for Q-Former.

	Each layer performs (optionally) self-attention over queries, then cross-attention
	from queries to the encoder tokens (K,V), followed by a feedforward MLP, each
	wrapped with residual connections and LayerNorm (pre-norm style).

	Parameters
	----------
	d_model : int
		Embedding dimension ``d``.
	num_heads : int
		Number of attention heads.
	dropout : float
		Dropout probability for attention and MLP.
	layer_norm : bool
		Whether to use LayerNorm; if False, Identity is used.
	use_query_self_attn : bool
		Whether to include a self-attention block over the queries.
	"""

	def __init__(
		self,
		d_model: int,
		num_heads: int,
		dropout: float,
		layer_norm: bool,
		use_query_self_attn: bool,
	) -> None:
		super().__init__()
		self.use_query_self_attn = use_query_self_attn

		# LayerNorms (pre-norm)
		self.ln_q_sa = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
		self.ln_q_ca = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
		self.ln_q_ff = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

		# Query self-attention (over n queries)
		if use_query_self_attn:
			self.self_attn = nn.MultiheadAttention(
				embed_dim=d_model,
				num_heads=num_heads,
				dropout=dropout,
				batch_first=True,
			)
		else:
			self.self_attn = None

		# Cross-attention (queries attend over encoder tokens K,V)
		self.cross_attn = nn.MultiheadAttention(
			embed_dim=d_model,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=True,
		)

		# Feedforward (MLP)
		hidden = 4 * d_model
		self.mlp = nn.Sequential(
			nn.Linear(d_model, hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, d_model),
			nn.Dropout(dropout),
		)

	def forward(
		self,
		q: torch.Tensor,
		k: torch.Tensor,
		v: torch.Tensor,
		query_pos: Optional[torch.Tensor] = None,
		attn_mask: Optional[torch.Tensor] = None,
		key_padding_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""Apply one Q-Former layer.

		Parameters
		----------
		q : torch.Tensor
			Query embeddings of shape ``(b, n, d)``.
		k : torch.Tensor
			Key embeddings of shape ``(b, L, d)``.
		v : torch.Tensor
			Value embeddings of shape ``(b, L, d)``.
		query_pos : Optional[torch.Tensor], optional
			Optional positional embedding for queries of shape ``(1, n, d)`` to be
			added to ``q``.
		attn_mask : Optional[torch.Tensor], optional
			Attention mask for cross-attention. Supported shapes follow
			``nn.MultiheadAttention`` with ``batch_first=True``:
			- ``(n, L)`` for 2D masks (broadcast across batch and heads), or
			- ``(b * num_heads, n, L)`` for per-batch, per-head masks.
		key_padding_mask : Optional[torch.Tensor], optional
			Boolean mask of shape ``(b, L)`` where True marks padding positions
			in keys/values.

		Returns
		-------
		torch.Tensor
			Updated query embeddings of shape ``(b, n, d)``.
		"""
		# Positional embedding on queries, if provided.
		if query_pos is not None:
			q = q + query_pos  # (b, n, d)

		# 1) (Optional) Query self-attention
		if self.self_attn is not None:
			q_sa_in = self.ln_q_sa(q)  # (b, n, d)
			q_sa_out, _ = self.self_attn(
				query=q_sa_in,  # (b, n, d)
				key=q_sa_in,    # (b, n, d)
				value=q_sa_in,  # (b, n, d)
				need_weights=False,
			)
			q = q + q_sa_out  # (b, n, d)

		# 2) Cross-attention: queries attend over encoder tokens
		q_ca_in = self.ln_q_ca(q)  # (b, n, d)
		q_ca_out, _ = self.cross_attn(
			query=q_ca_in,  # (b, n, d)
			key=k,          # (b, L, d)
			value=v,        # (b, L, d)
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			need_weights=False,
		)
		q = q + q_ca_out  # (b, n, d)

		# 3) Feedforward with residual
		q_ff_in = self.ln_q_ff(q)  # (b, n, d)
		q_ff_out = self.mlp(q_ff_in)  # (b, n, d)
		q = q + q_ff_out  # (b, n, d)

		return q


class QFormerProjector(nn.Module):
	"""Q-Former-style projector with learnable queries and cross-attention.

	This module maintains ``num_queries`` learnable query vectors of hidden
	dimension ``d_model`` (Q-Former latent size). It performs cross-attention
	from the queries to an input token sequence and returns a fixed number of
	output query embeddings. Crucially, it supports mismatched dimensions
	between input tokens, Q-Former latent size, and output embeddings via
	lightweight linear projections.

	Contract (symbols):
	- d_in:    input token dim
	- d_model: Q-Former latent dim (query/key/value attn dim)
	- d_out:   output token dim

	Shapes:
	- Learnable queries: (n, d_model)
	- Input tokens:      (b, L, d_in)
	- Output:            (b, n, d_out)

	Parameters
	----------
	d_model : int
		Q-Former latent dimension for queries/attention.
	num_queries : int
		Number of learnable queries ``n``.
	num_heads : int
		Number of attention heads used in attention (``d_model`` must be divisible by ``num_heads``).
	num_layers : int, optional
		Number of stacked Q-Former layers, by default 4.
	dropout : float, optional
		Dropout probability applied in attention and MLP, by default 0.0.
	layer_norm : bool, optional
		If True, apply LayerNorm in a pre-norm style, by default True.
	use_query_self_attn : bool, optional
		Whether to include self-attention over the queries, by default True.
	use_x_positional_encoding : bool, optional
		Whether to add sinusoidal positional encoding to K,V, by default True.
	use_query_positional_encoding : bool, optional
		Whether to add learnable positional encoding to queries, by default True.
	input_dim : Optional[int], optional
		If provided and different from ``d_model``, input tokens of shape ``(b, L, input_dim)``
		will be projected to ``d_model`` internally for attention. If ``None``,
		defaults to ``d_model``.
	output_dim : Optional[int], optional
		If provided and different from ``d_model``, final query outputs are projected
		to ``output_dim``. If ``None``, defaults to ``d_model``.

	Notes
	-----
	- Uses ``nn.MultiheadAttention`` with ``batch_first=True`` so tensors follow
	  the conventional (b, seq, d) layout.
	- The module expects to be moved to the same device/dtype as the inputs before forward.
	"""

	def __init__(
		self,
		d_model: int,
		num_queries: int,
		num_heads: int,
		num_layers: int = 4,
		dropout: float = 0.0,
		layer_norm: bool = True,
		use_query_self_attn: bool = True,
		use_x_positional_encoding: bool = True,
		use_query_positional_encoding: bool = True,
		input_dim: Optional[int] = None,
		output_dim: Optional[int] = None,
	) -> None:
		super().__init__()

		if d_model % num_heads != 0:
			raise ValueError(
				f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
			)

		# Core dims
		self.d_model = d_model  # latent dim for queries/attn
		self.num_queries = num_queries
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.dropout = dropout
		self.use_layer_norm = layer_norm
		self.use_query_self_attn = use_query_self_attn
		self.use_x_positional_encoding = use_x_positional_encoding
		self.use_query_positional_encoding = use_query_positional_encoding

		# Handle input/output dims
		self.d_in = input_dim if input_dim is not None else d_model
		self.d_out = output_dim if output_dim is not None else d_model

		# Learnable queries: (n, d)
		self.queries = nn.Parameter(torch.randn(num_queries, d_model))
		# Optional learnable positional embedding for queries: (n, d)
		self.query_pos = (
			nn.Parameter(torch.randn(1, num_queries, d_model))
			if use_query_positional_encoding
			else None
		)

		# Shared sinusoidal positional encoding module for encoder tokens (K/V)
		self.x_pos_encoding = (
			SinusoidalPositionalEncoding(d_model)
			if use_x_positional_encoding
			else None
		)

		# Pre-projection norm for input tokens (in dim)
		self.ln_x_in = nn.LayerNorm(self.d_in) if layer_norm else nn.Identity()

		# Projections to obtain K and V from input x: (b, L, d_in) -> (b, L, d_model)
		self.key_proj = nn.Linear(self.d_in, d_model)
		self.value_proj = nn.Linear(self.d_in, d_model)

		# Stack of Q-Former layers
		self.layers = nn.ModuleList(
			[
				_QFormerLayer(
					d_model=d_model,
					num_heads=num_heads,
					dropout=dropout,
					layer_norm=layer_norm,
					use_query_self_attn=use_query_self_attn,
				)
				for _ in range(num_layers)
			]
		)

		# Optional output projection: (b, n, d_model) -> (b, n, d_out)
		self.output_proj = (
			nn.Identity() if self.d_out == d_model else nn.Linear(d_model, self.d_out)
		)

	def forward(
		self,
		x: torch.Tensor,
		attn_mask: Optional[torch.Tensor] = None,
		key_padding_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""Project an input sequence into a fixed number of query embeddings via stacked layers.

		Parameters
		----------
		x : torch.Tensor
			Input token embeddings of shape ``(b, L, d)``.
		attn_mask : Optional[torch.Tensor], optional
			Optional attention mask for cross-attention. Supported shapes follow
			``nn.MultiheadAttention`` with ``batch_first=True``:
			- ``(n, L)`` broadcast across batch and heads, where ``n`` is the
			  number of queries and ``L`` is the input sequence length, or
			- ``(b * num_heads, n, L)`` for per-batch, per-head masks.
		key_padding_mask : Optional[torch.Tensor], optional
			Boolean mask of shape ``(b, L)`` where True indicates padding tokens
			in the key/value should be ignored.

		Returns
		-------
		torch.Tensor
			Output query embeddings of shape ``(b, n, d)``.

		Notes
		-----
		- Shape flow (per forward):
		  1) Input x: (b, L, d)
		  2) K = key_proj(LN(x)), V = value_proj(LN(x)) -> (b, L, d)
		  3) Positional encoding on K and V (optional): K += pe_x, V += pe_x -> (b, L, d)
		  4) Replicated queries: q = expand(queries) -> (b, n, d)
		  5) For each layer: [Self-Attn (opt) -> Cross-Attn -> MLP] -> (b, n, d)
		- The module must be on the same device and dtype as ``x``. A runtime
		  check enforces this.
		"""
		if x.dim() != 3:
			raise ValueError(f"x must be 3D (b, L, d), got shape {tuple(x.shape)}")

		b, L, d_in = x.shape
		if d_in != self.d_in:
			raise ValueError(
				f"Input dim d_in ({d_in}) must equal configured input_dim ({self.d_in})."
			)

		# Ensure device/dtype agreement; users should move the module via .to()
		if self.queries.device != x.device or self.queries.dtype != x.dtype:
			raise RuntimeError(
				"Device/dtype mismatch between input and module parameters. "
				"Move the projector to the same device/dtype as the input, e.g., "
				"projector = projector.to(x.device)."
			)

		# Prepare K/V projections for encoder tokens x.
		x_kv_in = self.ln_x_in(x)  # (b, L, d_in)
		k = self.key_proj(x_kv_in)    # (b, L, d_model)
		v = self.value_proj(x_kv_in)  # (b, L, d_model)
		# Optionally add positional encoding to K and V using the shared module.
		if self.x_pos_encoding is not None:
			# Build integer position indices per sequence: (B, L)
			timesteps = torch.arange(L, device=x.device).unsqueeze(0).expand(b, -1)  # (b, L)
			pe_x = self.x_pos_encoding(timesteps).to(dtype=x.dtype)  # (b, L, d_model)
			k = k + pe_x  # (b, L, d)
			v = v + pe_x  # (b, L, d)

		# Replicate learnable queries for the batch: (n, d) -> (b, n, d)
		q = self.queries.unsqueeze(0).expand(b, -1, -1)

		# Prepare query positional embedding if enabled.
		query_pos = self.query_pos if self.query_pos is not None else None  # (1, n, d) or None

		# Pass through stacked layers
		for layer in self.layers:
			q = layer(
				q=q,
				k=k,
				v=v,
				query_pos=query_pos,
				attn_mask=attn_mask,
				key_padding_mask=key_padding_mask,
			)  # (b, n, d)

		# Optional output projection
		y = self.output_proj(q)  # (b, n, d_out)
		return y


if __name__ == "__main__":
	# Minimal sanity checks (matched and mismatched dims)
	torch.manual_seed(0)

	# Case 1: matched dims (legacy behavior)
	b, L, d = 2, 16, 64
	n = 8
	h = 8
	proj1 = QFormerProjector(
		d_model=d,
		num_queries=n,
		num_heads=h,
		num_layers=3,
		dropout=0.1,
		layer_norm=True,
		use_query_self_attn=True,
		use_x_positional_encoding=True,
		use_query_positional_encoding=True,
	)
	x1 = torch.randn(b, L, d)
	y1 = proj1(x1)
	print("[matched] input:", x1.shape, "output:", y1.shape)  # (b, n, d)

	# Case 2: mismatched dims with input projection and output projection
	d_in, d_model, d_out = 48, 96, 128
	h2 = 8  # d_model % h2 == 0
	proj2 = QFormerProjector(
		d_model=d_model,
		num_queries=n,
		num_heads=h2,
		num_layers=2,
		dropout=0.0,
		layer_norm=True,
		use_query_self_attn=True,
		use_x_positional_encoding=True,
		use_query_positional_encoding=True,
		input_dim=d_in,
		output_dim=d_out,
	)
	x2 = torch.randn(b, L, d_in)
	y2 = proj2(x2)
	print("[mismatched] input:", x2.shape, "latent d=", d_model, "output:", y2.shape)  # (b, n, d_out)
