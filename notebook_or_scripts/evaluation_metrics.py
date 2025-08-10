"""
Evaluation metrics for jewelry image generation
Implements CLIP similarity, aesthetic scoring, and prompt adherence metrics
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import re
import requests
from sklearn.metrics.pairwise import cosine_similarity


class CLIPEvaluator:
    """CLIP-based evaluation for prompt adherence"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(device)
        
    def compute_clip_similarity(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity between image and prompt"""
        inputs = self.processor(
            text=[prompt], 
            images=[image], 
            return_tensors="pt", 
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            similarity = F.cosine_similarity(
                outputs.image_embeds, 
                outputs.text_embeds, 
                dim=1
            )
        
        return similarity.item()
    
    def extract_jewelry_terms(self, prompt: str) -> List[str]:
        """Extract jewelry-specific terms from prompt"""
        jewelry_terms = [
            "channel-set", "threader", "bezel-set", "eternity", "huggie",
            "bypass", "pavé", "signet", "cuff", "cluster", "diamond",
            "sapphire", "gold", "platinum", "silver", "rose-gold",
            "sterling silver", "yellow gold", "white gold"
        ]
        
        found_terms = []
        prompt_lower = prompt.lower()
        
        for term in jewelry_terms:
            if term in prompt_lower:
                found_terms.append(term)
        
        return found_terms
    
    def compute_term_adherence(self, image: Image.Image, prompt: str) -> Dict[str, float]:
        """Compute adherence to specific jewelry terms"""
        terms = self.extract_jewelry_terms(prompt)
        adherence_scores = {}
        
        for term in terms:
            # Create focused prompt for each term
            term_prompt = f"{term} jewelry, high quality product photography"
            score = self.compute_clip_similarity(image, term_prompt)
            adherence_scores[term] = score
        
        return adherence_scores


class AestheticEvaluator:
    """Evaluate modern aesthetic qualities"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_evaluator = CLIPEvaluator(device)
        
        # Define aesthetic criteria
        self.modern_criteria = [
            "modern jewelry design, clean lines, contemporary styling",
            "minimalist jewelry, refined elegance, sophisticated design",
            "high-end luxury jewelry, premium craftsmanship, quality materials",
            "contemporary jewelry photography, professional lighting, clean background"
        ]
        
        self.anti_criteria = [
            "vintage jewelry, antique style, ornate decoration",
            "cheap jewelry, costume jewelry, low quality materials",
            "cluttered design, busy patterns, excessive decoration",
            "poor photography, bad lighting, cluttered background"
        ]
    
    def compute_modern_aesthetic_score(self, image: Image.Image) -> float:
        """Compute how well image matches modern aesthetic"""
        modern_scores = []
        anti_scores = []
        
        # Score against modern criteria (higher is better)
        for criterion in self.modern_criteria:
            score = self.clip_evaluator.compute_clip_similarity(image, criterion)
            modern_scores.append(score)
        
        # Score against anti-criteria (lower is better)
        for criterion in self.anti_criteria:
            score = self.clip_evaluator.compute_clip_similarity(image, criterion)
            anti_scores.append(score)
        
        # Combine scores: high modern score, low anti-score
        modern_score = np.mean(modern_scores)
        anti_score = np.mean(anti_scores)
        
        # Weighted combination favoring modern aesthetics
        aesthetic_score = (modern_score * 0.7) + ((1 - anti_score) * 0.3)
        
        return aesthetic_score
    
    def analyze_composition(self, image: Image.Image) -> Dict[str, float]:
        """Analyze composition quality"""
        composition_criteria = [
            "professional jewelry photography, centered composition, proper framing",
            "clean white background, minimal distractions, focused subject",
            "proper lighting, good contrast, clear details",
            "high resolution, sharp focus, professional quality"
        ]
        
        scores = {}
        for i, criterion in enumerate(composition_criteria):
            score = self.clip_evaluator.compute_clip_similarity(image, criterion)
            scores[f"composition_{i+1}"] = score
        
        return scores


class PromptAdherenceAnalyzer:
    """Analyze how well images adhere to specific prompt elements"""
    
    def __init__(self, device: str = "cuda"):
        self.clip_evaluator = CLIPEvaluator(device)
        
    def parse_prompt_elements(self, prompt: str) -> Dict[str, List[str]]:
        """Parse prompt into different element categories"""
        elements = {
            "materials": [],
            "jewelry_type": [],
            "setting_style": [],
            "aesthetic_style": [],
            "photography_style": []
        }
        
        # Material patterns
        material_patterns = [
            r"(\d+k\s+(?:yellow\s+|rose\s+|white\s+)?gold)",
            r"(platinum)",
            r"(sterling\s+silver)",
            r"(rhodium)"
        ]
        
        # Jewelry type patterns
        jewelry_patterns = [
            r"(ring|rings)",
            r"(earrings?)",
            r"(bracelet)",
            r"(necklace)",
            r"(band)"
        ]
        
        # Setting style patterns
        setting_patterns = [
            r"(channel-set)",
            r"(bezel-set)",
            r"(pavé)",
            r"(prong-set)"
        ]
        
        prompt_lower = prompt.lower()
        
        # Extract materials
        for pattern in material_patterns:
            matches = re.findall(pattern, prompt_lower)
            elements["materials"].extend(matches)
        
        # Extract jewelry types
        for pattern in jewelry_patterns:
            matches = re.findall(pattern, prompt_lower)
            elements["jewelry_type"].extend(matches)
        
        # Extract setting styles
        for pattern in setting_patterns:
            matches = re.findall(pattern, prompt_lower)
            elements["setting_style"].extend(matches)
        
        return elements
    
    def compute_element_adherence(self, image: Image.Image, prompt: str) -> Dict[str, Dict[str, float]]:
        """Compute adherence to specific prompt elements"""
        elements = self.parse_prompt_elements(prompt)
        adherence_scores = {}
        
        for category, element_list in elements.items():
            if element_list:
                category_scores = {}
                for element in element_list:
                    # Create specific test prompt
                    test_prompt = f"{element} jewelry, product photography"
                    score = self.clip_evaluator.compute_clip_similarity(image, test_prompt)
                    category_scores[element] = score
                adherence_scores[category] = category_scores
        
        return adherence_scores


class ComprehensiveEvaluator:
    """Main evaluator combining all metrics"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_evaluator = CLIPEvaluator(device)
        self.aesthetic_evaluator = AestheticEvaluator(device)
        self.adherence_analyzer = PromptAdherenceAnalyzer(device)
    
    def evaluate_image(self, image: Image.Image, prompt: str) -> Dict[str, any]:
        """Comprehensive evaluation of generated image"""
        results = {
            "prompt": prompt,
            "overall_clip_similarity": self.clip_evaluator.compute_clip_similarity(image, prompt),
            "modern_aesthetic_score": self.aesthetic_evaluator.compute_modern_aesthetic_score(image),
            "term_adherence": self.clip_evaluator.compute_term_adherence(image, prompt),
            "composition_analysis": self.aesthetic_evaluator.analyze_composition(image),
            "element_adherence": self.adherence_analyzer.compute_element_adherence(image, prompt)
        }
        
        # Compute overall quality score
        overall_score = (
            results["overall_clip_similarity"] * 0.4 +
            results["modern_aesthetic_score"] * 0.4 +
            np.mean(list(results["term_adherence"].values())) * 0.2
        )
        
        results["overall_quality_score"] = overall_score
        
        return results
    
    def compare_improvements(self, baseline_image: Image.Image, improved_image: Image.Image, prompt: str) -> Dict[str, any]:
        """Compare baseline vs improved image"""
        baseline_results = self.evaluate_image(baseline_image, prompt)
        improved_results = self.evaluate_image(improved_image, prompt)
        
        comparison = {
            "baseline": baseline_results,
            "improved": improved_results,
            "improvements": {
                "clip_similarity_gain": improved_results["overall_clip_similarity"] - baseline_results["overall_clip_similarity"],
                "aesthetic_score_gain": improved_results["modern_aesthetic_score"] - baseline_results["modern_aesthetic_score"],
                "overall_quality_gain": improved_results["overall_quality_score"] - baseline_results["overall_quality_score"]
            }
        }
        
        return comparison
    
    def batch_evaluate(self, image_pairs: List[Tuple[Image.Image, Image.Image]], prompts: List[str]) -> Dict[str, any]:
        """Evaluate a batch of baseline vs improved image pairs"""
        all_comparisons = []
        
        for (baseline_img, improved_img), prompt in zip(image_pairs, prompts):
            comparison = self.compare_improvements(baseline_img, improved_img, prompt)
            all_comparisons.append(comparison)
        
        # Aggregate statistics
        clip_gains = [comp["improvements"]["clip_similarity_gain"] for comp in all_comparisons]
        aesthetic_gains = [comp["improvements"]["aesthetic_score_gain"] for comp in all_comparisons]
        quality_gains = [comp["improvements"]["overall_quality_gain"] for comp in all_comparisons]
        
        summary = {
            "individual_comparisons": all_comparisons,
            "aggregate_stats": {
                "mean_clip_gain": np.mean(clip_gains),
                "mean_aesthetic_gain": np.mean(aesthetic_gains),
                "mean_quality_gain": np.mean(quality_gains),
                "improvements_count": sum(1 for gain in quality_gains if gain > 0),
                "total_images": len(all_comparisons)
            }
        }
        
        return summary
