"""
Jewelry-specific improvements for Stable Diffusion
Addresses prompt adherence and aesthetic drift issues
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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
            "channel-set": "gemstones precisely set in parallel grooves, flush with the band for a sleek look",
            "threader": "long, delicate earring that threads through the ear with a thin chain or bar",
            "bezel-set": "gemstone fully enclosed by a thin metal rim for a modern, secure setting",
            "eternity band": "ring with a continuous circle of identical gemstones all around the band",
            "huggie": "small hoop earring that fits closely around the earlobe",
            "bypass": "ring where the band ends curve past each other without joining",
            "pavé": "surface covered with small, closely set stones creating a sparkling texture",
            "signet": "flat-topped ring, often engraved with a design or initials",
            "cuff": "rigid, open-ended bracelet worn on the wrist",
            "cluster": "group of multiple gemstones arranged together in one setting"
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
        
        # Add modern aesthetic context
        aesthetic_keywords = ["modern", "contemporary", "refined", "clean"]
        for keyword in aesthetic_keywords:
            if keyword in prompt.lower():
                enhanced_prompt += f", {self.modern_aesthetics.get(keyword, '')}"
        
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
            "channel-set", "threader", "bezel-set", "eternity", "huggie",
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
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
        self.device = device
        
        # Load the base pipeline
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
        self.jewelry_embeddings = JewelryTermEmbeddings(
            self.pipe.tokenizer, self.pipe.text_encoder
        )
        self.aesthetic_prompting = ModernAestheticPrompting()
        self.attention_weighting = AttentionWeighting()
        
    def generate_improved(
        self, 
        prompt: str, 
        num_images: int = 4,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Generate improved jewelry images"""
        
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
        
        # Generate multiple candidates
        images = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator
        ).images
        
        return images
    
    def generate_baseline(
        self, 
        prompt: str, 
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Generate baseline images without improvements"""
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        images = self.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator
        ).images
        
        return images
