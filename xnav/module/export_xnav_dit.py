import torch

from xnav.module.action_head.flow_matching_action_head import (
    FlowmatchingActionHeadConfig,
)
from xnav.module.action_head.cross_attention_dit import (
    DiTConfig,
    VLSelfAttentionConfig,
)
from xnav.module.xnav_dit import (
    XNavDitConfig,
    XNavDitForGoalConditioning,
)


import torch
import torch.nn as nn


def count_parameters(module: nn.Module):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params
    }


def main():
    # Provided full reference config (from original action head source)
    ref_cfg = {
        "action_dim": 32,
        "action_head_cfg": {
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
                "positional_embeddings": None,
            },
            "hidden_size": 1024,
            "input_embedding_dim": 1536,
            "max_action_dim": 32,
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
                "positional_embeddings": None,
            },
        },
        # Top-level (legacy source) fields not directly used for FlowmatchingActionHead:
        "action_horizon": 16,
        "architectures": ["GR00T_N1_5"],
        "attn_implementation": None,
        "backbone_cfg": {
            "eagle_path": "NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops",
            "load_bf16": False,
            "project_to_dim": None,
            "reproject_vision": False,
            "select_layer": 12,
            "tune_llm": False,
            "tune_visual": True,
            "use_flash_attention": True,
        },
        "hidden_size": 2048,
        "model_dtype": "float32",
        "model_type": "gr00t_n1_5",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.3",
    }

    ah_raw = ref_cfg["action_head_cfg"]
    dit_raw = ah_raw["diffusion_model_cfg"]
    vl_raw = ah_raw["vl_self_attention_cfg"]

    # Build typed nested configs
    dit_cfg = DiTConfig(
        num_attention_heads=dit_raw["num_attention_heads"],
        attention_head_dim=dit_raw["attention_head_dim"],
        output_dim=dit_raw["output_dim"],
        num_layers=dit_raw["num_layers"],
        dropout=dit_raw["dropout"],
        attention_bias=True,  # original code path assumes bias True
        activation_fn="gelu-approximate",  # keep default from our DiT impl
        num_embeds_ada_norm=1000,
        upcast_attention=False,
        norm_type=dit_raw["norm_type"],
        norm_elementwise_affine=False,
        norm_eps=1e-5,
        max_num_positional_embeddings=512,
        compute_dtype=torch.float32,
        final_dropout=dit_raw["final_dropout"],
        positional_embeddings=dit_raw["positional_embeddings"],
        interleave_self_attention=dit_raw["interleave_self_attention"],
        cross_attention_dim=dit_raw["cross_attention_dim"],
    )

    vl_cfg = VLSelfAttentionConfig(
        num_attention_heads=vl_raw["num_attention_heads"],
        attention_head_dim=vl_raw["attention_head_dim"],
        output_dim=ah_raw["backbone_embedding_dim"],  # ensure output matches backbone embedding dim
        num_layers=vl_raw["num_layers"],
        dropout=vl_raw["dropout"],
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        upcast_attention=False,
        max_num_positional_embeddings=512,
        compute_dtype=torch.float32,
        final_dropout=vl_raw["final_dropout"],
        positional_embeddings=vl_raw["positional_embeddings"],
        interleave_self_attention=False,
    )

    action_head_config = FlowmatchingActionHeadConfig(
        action_dim=ah_raw["action_dim"],
        action_horizon=ah_raw["action_horizon"],
        add_pos_embed=ah_raw["add_pos_embed"],
        backbone_embedding_dim=ah_raw["backbone_embedding_dim"],
        diffusion_model_cfg=dit_cfg,
        hidden_size=ah_raw["hidden_size"],
        input_embedding_dim=ah_raw["input_embedding_dim"],
        max_state_dim=ah_raw["max_state_dim"],
        model_dtype=ah_raw["model_dtype"],
        noise_beta_alpha=ah_raw["noise_beta_alpha"],
        noise_beta_beta=ah_raw["noise_beta_beta"],
        noise_s=ah_raw["noise_s"],
        num_inference_timesteps=ah_raw["num_inference_timesteps"],
        num_target_vision_tokens=ah_raw["num_target_vision_tokens"],
        num_timestep_buckets=ah_raw["num_timestep_buckets"],
        tune_diffusion_model=ah_raw["tune_diffusion_model"],
        tune_projector=ah_raw["tune_projector"],
        use_vlln=ah_raw["use_vlln"],
        vl_self_attention_cfg=vl_cfg,
    )

    # Build top-level XNav config to wrap the action head and goal-conditioning model
    xnav_config = XNavDitConfig(
        action_head_cfg=action_head_config,
        num_vision_tokens=action_head_config.num_target_vision_tokens,
        num_condition_tokens=action_head_config.num_target_vision_tokens,
        dinov2_encoder_name='vits',
        q_former_layers=1,
        q_former_heads=32,
        q_former_dropout=0.2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XNavDitForGoalConditioning(xnav_config).to(device).eval()

    print("Initialized XNavDitForGoalConditioning with reference action head config.")
    print(f"Device: {device}")
    print(f"Action dim: {xnav_config.action_head_cfg.action_dim}")
    print(f"Action horizon: {xnav_config.action_head_cfg.action_horizon}")
    print(f"State dim (max): {xnav_config.action_head_cfg.max_state_dim}")
    print(f"Backbone embedding dim: {xnav_config.action_head_cfg.backbone_embedding_dim}")
    dm_cfg = xnav_config.action_head_cfg.diffusion_model_cfg
    if isinstance(dm_cfg, dict):
        inner_dim = dm_cfg["num_attention_heads"] * dm_cfg["attention_head_dim"]
        num_layers = dm_cfg["num_layers"]
    else:
        inner_dim = dm_cfg.num_attention_heads * dm_cfg.attention_head_dim
        num_layers = dm_cfg.num_layers
    print(f"Diffusion inner dim: {inner_dim}")
    print(f"#DiT layers: {num_layers}")
    vl_cfg = xnav_config.action_head_cfg.vl_self_attention_cfg
    if isinstance(vl_cfg, dict):
        vl_layers = vl_cfg.get("num_layers")
    else:
        vl_layers = vl_cfg.num_layers if vl_cfg is not None else None
    print(f"#VL SA layers: {vl_layers}")
    print("Done (no forward pass as requested).")
    
    
    
    print("\n++++++++++++++++Parameter counts:++++++++++++++++")
    total_params = count_parameters(model)
    print(f"Total Model Parameters: Total={total_params['total_params'] / 1024 / 1024:.2f}M, Trainable={total_params['trainable_params'] / 1024 / 1024:.2f}M")
    
    nav_dit_params = count_parameters(model.nav_dit)
    print(f"Nav DiT Parameters: Total={nav_dit_params['total_params'] / 1024 / 1024:.2f}M, Trainable={nav_dit_params['trainable_params'] / 1024 / 1024:.2f}M")

    vision_encoder_params = count_parameters(model.nav_dit.vision_encoder)
    print(f"Vision Encoder Parameters: Total={vision_encoder_params['total_params'] / 1024 / 1024:.2f}M, Trainable={vision_encoder_params['trainable_params'] / 1024 / 1024:.2f}M")
    
    vision_proj_params = count_parameters(model.nav_dit.vision_proj)
    print(f"Vision Projection Parameters: Total={vision_proj_params['total_params'] / 1024 / 1024:.2f}M, Trainable={vision_proj_params['trainable_params'] / 1024 / 1024:.2f}M")

    action_head_params = count_parameters(model.nav_dit.action_head)
    print(f"Action Head Parameters: Total={action_head_params['total_params'] / 1024 / 1024:.2f}M, Trainable={action_head_params['trainable_params'] / 1024 / 1024:.2f}M")
    
    dit_params = count_parameters(model.nav_dit.action_head.model)
    print(f"Diffusion Model Parameters: Total={dit_params['total_params'] / 1024 / 1024:.2f}M, Trainable={dit_params['trainable_params'] / 1024 / 1024:.2f}M")
    
    point_encoder_params = count_parameters(model.point_goal_proj)
    print(f"Point Goal Encoder Parameters: Total={point_encoder_params['total_params'] / 1024 / 1024:.2f}M, Trainable={point_encoder_params['trainable_params'] / 1024 / 1024:.2f}M")
    
    image_goal_proj = count_parameters(model.image_goal_proj)
    print(f"Image Goal Encoder Parameters: Total={image_goal_proj['total_params'] / 1024 / 1024:.2f}M, Trainable={image_goal_proj['trainable_params'] / 1024 / 1024:.2f}M")



    print("\n\n\n+++++++++++++++Loading GR00T pretrained weights+++++++++++++++\n")
    gr00t_state_dict = torch.load("./checkpoints/GR00T-N1.5-3B.pth")
    action_head_state_dict = {}
    for key in gr00t_state_dict:
        if "action_head" in key:
            # print(f"GR00T Param: {key} | Shape: {gr00t_state_dict[key].shape}")
            k = key.split("action_head.")[1]
            action_head_state_dict[k] = gr00t_state_dict[key]


    missing_keys, unexpected_keys = model.nav_dit.action_head.load_state_dict(action_head_state_dict, strict=False)
    print(f"Loaded GR00T action head into XNav DiT action head.")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    
    print("\n+++++++++++++++Loading DepthAnythingV2 pretrained weights+++++++++++++++\n")
    depthanything_state_dict = torch.load('/home/shrelic/models/depth_anything_v2_vits.pth', map_location='cpu')
   
    missing_keys, unexpected_keys = model.nav_dit.vision_encoder.load_state_dict(depthanything_state_dict, strict=False)
    print(f"Loaded DepthAnythingV2 vision encoder into XNav DiT vision encoder.")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    model.save_pretrained("./checkpoints/XNavDit")

    # check loading
    print("\n+++++++++++++++Verifying saved XNavDit checkpoint+++++++++++++++\n")
    model = XNavDitForGoalConditioning.from_pretrained("./checkpoints/XNavDit")
    print("Loaded XNavDit model from saved checkpoint.")
    print(f"model type: {type(model)}")


if __name__ == "__main__":
    main()
