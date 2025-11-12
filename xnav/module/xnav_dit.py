import torch
import torch.nn as nn
from dataclasses import dataclass, field
from transformers import BatchFeature, PretrainedConfig, PreTrainedModel

from xnav.module.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig, FlowmatchingActionHead
from xnav.module.encoder.vision import DinoV2Encoder, ImageGoalEncoder, grid_pooling
from xnav.module.encoder.point import PointEncoder
from xnav.module.projector.q_former import QFormerProjector

@dataclass
class XNavDitConfig(PretrainedConfig):
    model_type = "xnav_dit"

    # Note: Using default_factory here requires that we explicitly set this in __init__,
    # because we override dataclass' auto-generated __init__.
    action_head_cfg: FlowmatchingActionHeadConfig = field(
        default_factory=FlowmatchingActionHeadConfig,
        metadata={"help": "Action head configuration."}
    )

    num_vision_tokens: int = field(
        default=32,
        metadata={"help": "Number of vision tokens to feed into the DiT action head."}
    )
    
    num_condition_tokens: int = field(
        default=32,
        metadata={"help": "Number of condition tokens to feed into DiT cross attention."}
    )

    dinov2_encoder_name: str = field(
        default='vits',
        metadata={"help": "Name of the DINOv2 vision encoder to use."}
    )

    q_former_layers: int = field(
        default=4,
        metadata={"help": "Number of Q-Former layers to use for vision token projection."}
    )

    q_former_heads: int = field(
        default=32,
        metadata={"help": "Number of attention heads in the Q-Former projector."}
    )

    q_former_dropout: float = field(
        default=0.2,
        metadata={"help": "Dropout probability for Q-Former projector."}
    )

    def __init__(
        self,
        action_head_cfg: FlowmatchingActionHeadConfig | dict | None = None,
        num_vision_tokens: int = 32,
        num_condition_tokens: int = 32,
        dinov2_encoder_name: str = 'vits',
        q_former_layers: int = 4,
        q_former_heads: int = 32,
        q_former_dropout: float = 0.2,
        **kwargs,
    ):
        # Initialize base PretrainedConfig
        super().__init__(**kwargs)

        # Explicitly set instance attributes (do not rely on dataclass auto-init)
        if isinstance(action_head_cfg, dict):
            self.action_head_cfg = FlowmatchingActionHeadConfig(**action_head_cfg)
        else:
            self.action_head_cfg = action_head_cfg or FlowmatchingActionHeadConfig()
        self.num_vision_tokens = num_vision_tokens
        self.num_condition_tokens = num_condition_tokens
        self.dinov2_encoder_name = dinov2_encoder_name
        self.q_former_layers = q_former_layers
        self.q_former_heads = q_former_heads
        self.q_former_dropout = q_former_dropout

class XNavPretrainedModel(PreTrainedModel):
    """ Base class for XNav models.
    """
    config_class = XNavDitConfig
    supports_gradient_checkpointing = True
    

class XNavDit(XNavPretrainedModel):
    def __init__(self, config: XNavDitConfig):
        super().__init__(config)
        self.config = config
        # Initialize model components here based on config

        self.vision_encoder = DinoV2Encoder(encoder=self.config.dinov2_encoder_name)
        self.vision_proj = QFormerProjector(
            d_model=config.action_head_cfg.input_embedding_dim,
            num_queries=config.action_head_cfg.num_target_vision_tokens,
            num_heads=config.q_former_heads,
            num_layers=config.q_former_layers,
            dropout=config.q_former_dropout,
            input_dim=self.vision_encoder.emb_dim,
            output_dim=config.action_head_cfg.input_embedding_dim,
        )
        
        self.action_head = FlowmatchingActionHead(self.config.action_head_cfg)

    def forward(self,
                condition: BatchFeature, 
                images: torch.Tensor,
                action_input: BatchFeature
               ) -> BatchFeature:
        """
        Forward pass of the XNavDit model.
        Args:
        - condition: BatchFeature containing conditioning information. condition.nav_embeddings with shape of (B, L, d_emb) is expected.
        
        - images: Tensor containing observation image data. Shape (B, T, C, H, W) is expected, where T is window size.
        
        - action_input: BatchFeature containing action input information. action.state action.action action.action_mask action.embodiment_id is expected. action.state shape is (B, T, d_state), action.action shape is (B, T, d_action), action.action_mask shape is (B, T, d_action), and action.embodiment_id shape is (B,).
        """
        B, T, C, H, W = images.shape
        patch_size = self.vision_encoder.pretrained.patch_size
        H_patches = H // patch_size
        W_patches = W // patch_size
        assert H % patch_size == 0 and W % patch_size == 0, "Image height and width must be divisible by the DINOv2 patch size."
        # split images into histories images and current image

        # [B, T-1, C, H, W]
        history_images = images[:, :-1]

        # [B, 1, C, H, W]
        current_image = images[:, -1:]

        # (B, T-1, N_his, d_vis)
        history_vis_tokens = self.vision_encoder.emb_images(history_images)
        
        # use grid pooling to reduce num_vision_tokens to 1 / 4x4
        # (B, T-1, N_his / 16, d_vis)
        history_vis_tokens = grid_pooling(history_vis_tokens, H_patches, W_patches, 4, 4)
        
        # (B, 1, N_curr, d_vis)
        current_vis_tokens = self.vision_encoder.emb_images(current_image)
        
        # flatten and concat history and current vis tokens
        # (B, (T-1) * N_his / 16, d_vis)
        history_vis_tokens = history_vis_tokens.reshape(B, -1, history_vis_tokens.shape[-1])
        # (B, N_curr, d_vis)
        current_vis_tokens = current_vis_tokens.reshape(B, -1, current_vis_tokens.shape[-1])
        # (B, L_total, d_vis)
        vis_tokens = torch.cat([history_vis_tokens, current_vis_tokens], dim=1)  
        
        # project vision tokens to action head input dim
        # (B, num_vision_tokens, d_action_head_input)
        dit_vis_tokens = self.vision_proj(vis_tokens)
        
        observation = BatchFeature({
            'vision_tokens': dit_vis_tokens
        })
        outputs = self.action_head(
            condition=condition,
            observation=observation,
            action_input=action_input
        )
        return outputs
    

class XNavDitForGoalConditioning(XNavPretrainedModel):
    def __init__(self, config: XNavDitConfig):
        super().__init__(config)
        self.config = config
        self.nav_dit = XNavDit(config)
        # diffusion_model_cfg may be stored as a dict for serialization
        dm_cfg = config.action_head_cfg.diffusion_model_cfg
        cross_attention_dim = dm_cfg["cross_attention_dim"] if isinstance(dm_cfg, dict) else dm_cfg.cross_attention_dim

        self.point_goal_proj = PointEncoder(
            embed_dim=cross_attention_dim,
            num_tokens=config.num_condition_tokens,
            qformer_layers=config.q_former_layers,
            qformer_heads=config.q_former_heads,
            dropout=config.q_former_dropout,
        )

        self.image_goal_proj = ImageGoalEncoder(
            encoder=config.dinov2_encoder_name,
            num_tokens=config.num_condition_tokens,
            emb_dim=cross_attention_dim,
            q_former_layers=config.q_former_layers,
            q_former_heads=config.q_former_heads,
            dropout=config.q_former_dropout,
        )
    
    def forward(self,
                goal: BatchFeature, 
                images: torch.Tensor,
                action_input: BatchFeature
               ) -> BatchFeature:
        """
        goal: BatchFeature containing goal information. goal.point or goal.image is expected. goal.point shape is (B, 4) and goal.image shape is (B, C, H, W).
        """
        assert 'point' in goal or 'image' in goal, "Goal must contain either 'point' or 'image' key."
        
        if "point" in goal:
            # project point goal to vision tokens
            # (B, num_condition_tokens, d)
            goal_vis_tokens = self.point_goal_proj(goal['point'])
        
        if "image" in goal:
            # project image goal to vision tokens
            # (B, num_condition_tokens, d)
            image_goal_tokens = self.image_goal_proj(goal['image'])
            goal_vis_tokens = image_goal_tokens
            
        # average goal tokens if both point and image are provided
        if "point" in goal and "image" in goal:
            goal_vis_tokens = (goal_vis_tokens + image_goal_tokens) / 2.0
            
        condition = BatchFeature({
            'nav_embeddings': goal_vis_tokens
        })
        
        outputs = self.nav_dit(
            condition=condition,
            images=images,
            action_input=action_input
        )

        return outputs

if __name__ == "__main__":
    # Build a fully-specified config (no relying on defaults) and run a smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Define the nested action head config dict (borrowed from your reference)
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
            "positional_embeddings": None,
        },
        "hidden_size": 1024,
        "input_embedding_dim": 1536,
        # Kept for compatibility with the source config; will be stored on the config object.
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
    }

    # 2) Instantiate the action head config and then the top-level model config explicitly
    action_head_config = FlowmatchingActionHeadConfig(**action_head_cfg_dict)

    xnav_config = XNavDitConfig(
        action_head_cfg=action_head_config,
        num_vision_tokens=action_head_config.num_target_vision_tokens,
        dinov2_encoder_name='vits',
        q_former_layers=4,
        q_former_heads=32,
        q_former_dropout=0.2,
    )

    # 3) Instantiate the model
    model = XNavDit(xnav_config).to(device)
    model.eval()

    # 4) Prepare dummy inputs consistent with the explicit config
    B = 2
    T = 16  # align with action_horizon for convenience in this smoke test
    H = 224
    W = 224
    C = 3
    L_nav = 20

    d_nav = action_head_config.backbone_embedding_dim       # 2048
    d_state = action_head_config.max_state_dim              # 64
    d_action = action_head_config.action_dim                # 32

    # Use 0..255 image range to match the encoder's normalization path
    images = torch.randint(0, 256, (B, T, C, H, W), device=device, dtype=torch.uint8).float()

    condition = BatchFeature({
        'nav_embeddings': torch.randn(B, L_nav, d_nav, device=device)
    })
    action_input = BatchFeature({
        'state': torch.randn(B, T, d_state, device=device),
        'action': torch.randn(B, T, d_action, device=device),
        'action_mask': torch.ones(B, T, d_action, device=device),
        'embodiment_id': torch.zeros(B, dtype=torch.long, device=device)
    })

    # 5) Run forward
    with torch.no_grad():
        outputs = model(
            condition=condition,
            images=images,
            action_input=action_input,
        )
    print(f"Device: {device}")
    print("Forward pass successful. Output keys:", outputs.keys())
    print(f"loss: {outputs['loss']}")
    
    # 6) Goal conditioning examples using XNavDitForGoalConditioning
    model_goal = XNavDitForGoalConditioning(xnav_config).to(device)
    model_goal.eval()

    # Point goal (e.g., x,y,z,yaw) shape (B,4)
    point_goal = BatchFeature({
        'point': torch.randn(B, 4, device=device)
    })

    # Image goal (single target image per batch) shape (B,C,H,W)
    image_goal = BatchFeature({
        'image': torch.randint(0, 256, (B, C, H, W), device=device, dtype=torch.uint8).float()
    })

    # Both point + image goal (will be averaged inside forward)
    combined_goal = BatchFeature({
        'point': point_goal['point'],
        'image': image_goal['image'],
    })

    with torch.no_grad():
        out_point = model_goal(goal=point_goal, images=images, action_input=action_input)
        out_image = model_goal(goal=image_goal, images=images, action_input=action_input)
        out_both = model_goal(goal=combined_goal, images=images, action_input=action_input)

    print("Goal conditioning (point) loss:", out_point['loss'])
    print("Goal conditioning (image) loss:", out_image['loss'])
    print("Goal conditioning (point+image averaged) loss:", out_both['loss'])
    