import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model
import logging

class SAMWithLoRA(nn.Module):
    def __init__(self, checkpoint, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=lora_dropout,
            bias="none"
        )
        self.encoder = get_peft_model(self.sam.image_encoder, lora_config)
        self.decoder = self.sam.mask_decoder
        self.prompt_encoder = self.sam.prompt_encoder
        
        # Freeze non-LoRA encoder parameters
        for name, param in self.encoder.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        
        # Make decoder and prompt encoder trainable
        for param in self.decoder.parameters():
            param.requires_grad = True
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Trainable parameters: {trainable_params}")
        logging.info(f"Total parameters: {total_params}")
        logging.info(f"Percentage trainable: {trainable_params / total_params * 100:.2f}%")
        
        trainable_names = [name for name, param in self.named_parameters() if param.requires_grad]
        logging.info(f"Trainable parameter names: {trainable_names}")
    
    def forward(self, images, bboxes):
        image_embeddings = self.encoder(images)
        
        masks_list = []
        for i, bbox_batch in enumerate(bboxes):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=bbox_batch.to(images.device),
                masks=None
            )
            masks, _ = self.decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            masks = torch.clamp(masks, min=-10.0, max=10.0)
            masks_list.append(masks.squeeze(1))  # [N_i, 256, 256]
        
        return masks_list