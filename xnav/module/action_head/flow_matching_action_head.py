# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 Weikai Xie.
# SPDX-License-Identifier: Apache-2.0
#
# This file is based on code from NVIDIA CORPORATION & AFFILIATES, originally
# licensed under the Apache License, Version 2.0, with modifications by
# Weikai Xie made in 2025.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from xnav.module.action_head import (
    SinusoidalPositionalEncoding,
    swish,
)

from xnav.module.action_head.cross_attention_dit import (
    DiT,
    SelfAttentionTransformer,
    DiTConfig,
    VLSelfAttentionConfig,
)


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """Action head configuration.

    IMPORTANT: Original code set `diffusion_model_cfg=None` and then blindly did
    `DiT(**config.diffusion_model_cfg)`, causing a TypeError when no user-supplied
    dict was provided. We supply a safe default here aligned with DiT signature.
    """
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: DiTConfig | None = field(
        default=None, metadata={"help": "Diffusion model configuration object."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    max_state_dim: int = field(default=None, metadata={"help": "Max state dimension."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: VLSelfAttentionConfig | None = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Assign all provided kwargs onto self (legacy behavior)
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Provide defaults for critical numeric fields if omitted
        if getattr(self, 'action_dim', None) is None:
            self.action_dim = 32
        if getattr(self, 'action_horizon', None) is None:
            self.action_horizon = 16
        if getattr(self, 'max_state_dim', None) is None:
            self.max_state_dim = 64
        if getattr(self, 'backbone_embedding_dim', None) is None:
            self.backbone_embedding_dim = 1536
        if getattr(self, 'hidden_size', None) is None:
            self.hidden_size = 1024
        # Coerce dicts -> plain dicts for JSON-serializable storage (keep backward compatibility)
        dm_cfg = getattr(self, 'diffusion_model_cfg', None)
        if isinstance(dm_cfg, DiTConfig):
            dm_dict = dm_cfg.to_kwargs()
        elif isinstance(dm_cfg, dict):
            dm_dict = dict(dm_cfg)
        else:
            # Provide a reasonable default aligned with previous dict default
            dm_dict = {
                "attention_head_dim": 48,
                "cross_attention_dim": self.backbone_embedding_dim,
                "dropout": 0.1,
                "final_dropout": True,
                "interleave_self_attention": False,
                "norm_type": "ada_norm",
                "num_attention_heads": 24,
                "num_layers": 8,
                "output_dim": self.hidden_size,
                "positional_embeddings": "sinusoidal",
            }
        # Remove non-serializable entries like torch dtypes if present
        if "compute_dtype" in dm_dict:
            dm_dict.pop("compute_dtype", None)
        self.diffusion_model_cfg = dm_dict

        # vl_self_attention_cfg -> store as plain dict too
        vl_cfg = getattr(self, 'vl_self_attention_cfg', None)
        if isinstance(vl_cfg, VLSelfAttentionConfig):
            vl_dict = vl_cfg.to_kwargs()
        elif isinstance(vl_cfg, dict):
            vl_dict = dict(vl_cfg)
        elif getattr(self, 'use_vlln', True) and vl_cfg is None:
            num_heads = 32
            head_dim = max(1, self.backbone_embedding_dim // num_heads)
            if num_heads * head_dim != self.backbone_embedding_dim:
                head_dim = 48
                num_heads = max(1, self.backbone_embedding_dim // head_dim)
            vl_dict = {
                "num_attention_heads": num_heads,
                "attention_head_dim": head_dim,
                "dropout": 0.1,
                "final_dropout": True,
                "num_layers": 4,
                "positional_embeddings": None,
                "output_dim": self.backbone_embedding_dim,
            }
        else:
            vl_dict = None
        if isinstance(vl_dict, dict) and "compute_dtype" in vl_dict:
            vl_dict.pop("compute_dtype", None)
        self.vl_self_attention_cfg = vl_dict


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        # Instantiate diffusion model with typed config built from stored dict
        if isinstance(config.diffusion_model_cfg, dict):
            dm_obj = DiTConfig(**config.diffusion_model_cfg)
        else:
            dm_obj = config.diffusion_model_cfg
        self.model = DiT(**dm_obj.to_kwargs())
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        # self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        # nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**VLSelfAttentionConfig(**config.vl_self_attention_cfg).to_kwargs())
            if config.use_vlln and config.vl_self_attention_cfg is not None
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_condition(self, condition: BatchFeature) -> BatchFeature:
        nav_embeddings = condition.nav_embeddings
        nav_embeddings = self.vlln(nav_embeddings)
        nav_embeddings = self.vl_self_attention(nav_embeddings)
        condition.nav_embeddings = nav_embeddings
        return condition

    def forward(self, condition: BatchFeature, observation: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        condition = self.process_condition(condition)

        if self.config.expand_batch is not None:
            for k, v in condition.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                condition[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

            for k, v in observation.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                observation[k] = expanded

        # Get vision and language embeddings.
        vl_embs = condition.nav_embeddings
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vision_tokens = observation.vision_tokens
        sa_embs = torch.cat((state_features, vision_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            # encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        # print(f"pred_action.shape: {pred_actions.shape}")
        # print(f"velocity.shape: {velocity.shape}")
        # print(f"action_mask.shape: {action_mask.shape}")
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        # print(f"Predicted actions: {pred_actions}")
        # print(f"Target actions: {velocity}")
        # print(f"Action mask: {action_mask}")
        # print(f"Action head loss: {loss.item()}")
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, condition: BatchFeature, observation: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        condition = self.process_condition(condition)

        # Get vision and language embeddings.
        vl_embs = condition.nav_embeddings
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            
            vision_tokens = observation.vision_tokens
            sa_embs = torch.cat((state_features, vision_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 从你提供的 JSON 中提取 action_head_cfg 字典
    action_head_cfg_dict = {
        "action_dim": 32,
        "action_horizon": 16,
        "add_pos_embed": True,
        "backbone_embedding_dim": 2048,
        "diffusion_model_cfg": {
            "attention_head_dim": 48,
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_attention_heads": 32,
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None
        },
        "hidden_size": 1024,
        "input_embedding_dim": 1536,
        "max_action_dim": 32,       # 注意：这个 key 不在 FlowmatchingActionHeadConfig 的定义中，但会被 **kwargs 捕获
        "max_state_dim": 64,
        "model_dtype": "float32",
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "num_timestep_buckets": 1000,
        "tune_diffusion_model": True,
        "tune_projector": True,
        "use_vlln": True,
        "vl_self_attention_cfg": {
            "attention_head_dim": 64,
            "dropout": 0.2,
            "final_dropout": True,
            "num_attention_heads": 32,
            "num_layers": 4,
            "positional_embeddings": None
        }
    }

    # 2. 使用字典解包 (**) 来实例化配置对象
    action_head_config = FlowmatchingActionHeadConfig(**action_head_cfg_dict)
    
    # 验证一个关键值是否已正确设置
    print(f"Configured action_horizon: {action_head_config.action_horizon}")
    print(f"Configured max_state_dim: {action_head_config.max_state_dim}")
    print(f"Configured backbone_embedding_dim: {action_head_config.backbone_embedding_dim}")

    # 3. 实例化模型并移动到设备
    action_head = FlowmatchingActionHead(action_head_config).to(DEVICE)
    action_head.eval()
    # 打印模型结构（可选，但有助于调试）
    # print(action_head)

    # 4. 根据 *新配置* 创建 dummy inputs
    batch_size = 2
    vl_seq_len = 20  # 这个值在 config 中没有，保持任意值
    
    # 从新的 config 对象中获取维度
    action_horizon = action_head_config.action_horizon       # 将变为 16
    action_dim = action_head_config.action_dim             # 将变为 32
    state_dim = action_head_config.max_state_dim           # 将变为 64
    vl_emb_dim = action_head_config.backbone_embedding_dim   # 将变为 2048

    # embodiment_id 仍然使用默认的 max_num_embodiments (32)，因为 config 中未指定
    embodiment_id = torch.randint(0, action_head_config.max_num_embodiments, (batch_size,), device=DEVICE)

    # 使用模型参数的 dtype
    # 注意：你的 config 中 "model_dtype": "float32" 和 "torch_dtype": "bfloat16"
    # 这里的 action_head.dtype 会继承自 nn.Module 的默认值（float32）
    # 除非你在实例化时显式传递了 dtype，否则这里会是 float32
    dtype = action_head.dtype
    print(f"\nModel Dtype: {dtype}")

    # 5. 创建形状匹配新 config 的张量
    nav_embeddings = torch.randn(batch_size, vl_seq_len, vl_emb_dim, device=DEVICE, dtype=dtype)
    state_horizon = 1 # 原始代码硬编码为 1
    state = torch.randn(batch_size, state_horizon, state_dim, device=DEVICE, dtype=dtype)
    action = torch.randn(batch_size, action_horizon, action_dim, device=DEVICE, dtype=dtype)

    print(f"Input nav_embeddings shape: {nav_embeddings.shape}")
    print(f"Input state shape: {state.shape}")
    print(f"Input action shape: {action.shape}")
    
    observation_vision_tokens = torch.randn(
        batch_size,
        action_head_config.num_target_vision_tokens,
        action_head_config.input_embedding_dim,
        device=DEVICE,
        dtype=dtype
    )

    # 6. 运行 get_action
    with torch.no_grad():
        out = action_head.get_action(
            condition=BatchFeature(
                data={
                    "nav_embeddings": nav_embeddings,
                }
            ),
            observation=BatchFeature(
                data={
                    "vision_tokens": observation_vision_tokens,
                }
            ),
            action_input=BatchFeature(
                data={
                    "embodiment_id": embodiment_id,
                    "state": state,
                    "action": action, # 'action' 在 get_action 中仅用作形状参考，这里传入无妨
                }
            ),
        )
    
    print("\n--- Output ---")
    print(out)
    print(f"Predicted action shape: {out['action_pred'].shape}")
    
    # test forward pass
    action_input_for_forward = BatchFeature(
        data={
            "embodiment_id": embodiment_id,
            "state": state,
            "action": action,
            "action_mask": torch.ones(batch_size, action_horizon, action_dim, device=DEVICE, dtype=dtype),
        }
    )
    out_forward = action_head(
        condition=BatchFeature(
            data={
                "nav_embeddings": nav_embeddings,
            }
        ),
        observation=BatchFeature(
            data={
                "vision_tokens": observation_vision_tokens,
            }
        ),
        action_input=action_input_for_forward,
    )
    print(f"Forward pass loss: {out_forward['loss']}")
