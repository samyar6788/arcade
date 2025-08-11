"""
Jewelry-specific improvements for Stable Diffusion
Addresses prompt adherence and aesthetic drift issues
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import json


class JewelryTermEmbeddings:
    """Enhanced embeddings for jewelry-specific terminology"""
    
    def __init__(self, tokenizer, text_encoder):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        # Jewelry terminology dictionary with enhanced descriptions
        self.jewelry_terms = {
            "channel-set": "parallel groove gemstones",
            "threader": "thread-through earring",
            "bezel-set": "rim-enclosed gemstone",
            "eternity band": "full-band gemstones",
            "huggie": "small close hoop",
            "bypass": "overlapping band ring",
            "pavé": "small set stones",
            "signet": "flat engraved ring",
            "cuff": "open bracelet",
            "cluster": "grouped gemstones"
        }
        
        # Modern aesthetic descriptors
        self.modern_aesthetics = {
            "clean": "minimalist design with smooth surfaces and precise geometric lines",
            "contemporary": "current design trends with refined simplicity and intentional craftsmanship",
            "refined": "sophisticated elegance with attention to subtle details and proportions",
            "modern": "current aesthetic with clean lines, quality materials, and thoughtful design"
        }
        
    def enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt with better jewelry terminology"""
        enhanced_prompt = prompt
        
        # Replace jewelry terms with enhanced descriptions
        for term, description in self.jewelry_terms.items():
            if term in prompt.lower():
                # Add context without replacing the original term
                enhanced_prompt = enhanced_prompt.replace(
                    term, f"{term} ({description})"
                )
        
        # # Add modern aesthetic context
        # aesthetic_keywords = ["modern", "contemporary", "refined", "clean"]
        # for keyword in aesthetic_keywords:
        #     if keyword in prompt.lower():
        #         enhanced_prompt += f", {self.modern_aesthetics.get(keyword, '')}"
        
        return enhanced_prompt


class ModernAestheticPrompting:
    """Handles modern aesthetic guidance and negative prompting"""
    
    def __init__(self):
        # Negative prompts to avoid vintage/cheap aesthetics
        self.negative_base = [
            "vintage", "antique", "ornate", "fussy", "busy", "cluttered",
            "cheap", "plastic", "fake", "costume jewelry", "tacky",
            "overly decorative", "baroque", "rococo", "gaudy", "flashy",
            "low quality", "mass produced", "generic", "outdated style"
        ]
        
        # Positive modern aesthetic reinforcement
        self.modern_reinforcement = [
            "high-end jewelry", "luxury craftsmanship", "premium materials",
            "contemporary design", "minimalist elegance", "refined aesthetics",
            "modern luxury", "sophisticated styling", "clean design language",
            "intentional craftsmanship", "quality construction"
        ]
    
    def get_negative_prompt(self) -> str:
        """Generate negative prompt to avoid aesthetic drift"""
        return ", ".join(self.negative_base)
    
    def enhance_with_modern_aesthetic(self, prompt: str) -> str:
        """Add modern aesthetic reinforcement to prompt"""
        # Add modern aesthetic terms strategically
        modern_terms = ", ".join(self.modern_reinforcement[:3])  # Use top 3 terms
        return f"{prompt}, {modern_terms}"


class AttentionWeighting:
    """Implements attention weighting for critical jewelry terms"""
    
    def __init__(self):
        # Define critical terms that need attention weighting
        self.critical_terms = [
            "channel-set", "threader", "bezel-set", "eternity band", "huggie",
            "bypass", "pavé", "signet", "cuff", "cluster",
            "diamond", "sapphire", "gold", "platinum", "silver"
        ]
        
    def apply_attention_weights(self, prompt: str, weight: float = 1.3) -> str:
        """Apply attention weighting to critical terms using (term:weight) syntax"""
        weighted_prompt = prompt
        
        for term in self.critical_terms:
            if term in prompt.lower():
                # Find exact matches and apply weighting
                pattern = r'\b' + re.escape(term) + r'\b'
                weighted_prompt = re.sub(
                    pattern, 
                    f"({term}:{weight})", 
                    weighted_prompt, 
                    flags=re.IGNORECASE
                )
        
        return weighted_prompt


class ImprovedJewelryPipeline:
    """Main pipeline combining all improvements"""
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        
        # Determine if this is SDXL or regular SD
        self.is_sdxl = "xl" in model_id.lower()
        
        if self.is_sdxl:
            # Load SDXL pipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
        else:
            # Load regular SD pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
        
        # Use DPM solver for better quality
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(device)
        
        # Initialize improvement modules
        # For SDXL, use the first text encoder
        text_encoder = self.pipe.text_encoder if not self.is_sdxl else self.pipe.text_encoder
        tokenizer = self.pipe.tokenizer if not self.is_sdxl else self.pipe.tokenizer
        
        self.jewelry_embeddings = JewelryTermEmbeddings(tokenizer, text_encoder)
        self.aesthetic_prompting = ModernAestheticPrompting()
        self.attention_weighting = AttentionWeighting()
        
    def generate_improved(
        self, 
        prompt: str, 
        num_images: int = 4,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Generate improved jewelry images"""
        
        # Set default dimensions based on model type
        if height is None:
            height = 1024 if self.is_sdxl else 512
        if width is None:
            width = 1024 if self.is_sdxl else 512
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Apply all improvements
        enhanced_prompt = self.jewelry_embeddings.enhance_prompt(prompt)
        enhanced_prompt = self.aesthetic_prompting.enhance_with_modern_aesthetic(enhanced_prompt)
        enhanced_prompt = self.attention_weighting.apply_attention_weights(enhanced_prompt)
        
        negative_prompt = self.aesthetic_prompting.get_negative_prompt()
        
        print(f"Enhanced prompt: {enhanced_prompt}")
        print(f"Negative prompt: {negative_prompt}")
        
        # Generate multiple candidates with model-specific parameters
        generation_kwargs = {
            "prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "num_images_per_prompt": num_images,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "generator": generator
        }
        
        # SDXL-specific parameters
        if self.is_sdxl:
            # SDXL works better with these default parameters
            generation_kwargs.update({
                "guidance_scale": 5.0,  # SDXL typically uses lower guidance
                "num_inference_steps": 30,  # SDXL is more efficient
            })
        
        images = self.pipe(**generation_kwargs).images
        
        return images
    
    def generate_baseline(
        self, 
        prompt: str, 
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Generate baseline images without improvements"""
        
        # Set default dimensions based on model type
        if height is None:
            height = 1024 if self.is_sdxl else 512
        if width is None:
            width = 1024 if self.is_sdxl else 512
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate with model-specific parameters
        generation_kwargs = {
            "prompt": prompt,
            "num_images_per_prompt": num_images,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "generator": generator
        }
        
        # SDXL-specific parameters
        if self.is_sdxl:
            generation_kwargs.update({
                "guidance_scale": 5.0,  # SDXL typically uses lower guidance
                "num_inference_steps": 30,  # SDXL is more efficient
            })
        
        images = self.pipe(**generation_kwargs).images
        
        return images
