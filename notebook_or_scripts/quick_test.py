#!/usr/bin/env python3
"""
Quick test script to verify the jewelry improvement pipeline works
Run this to test one prompt without running the full notebook
"""

import torch
from jewelry_improvements import ImprovedJewelryPipeline
from evaluation_metrics import ComprehensiveEvaluator
import os

def main():
    print("🔍 Testing Jewelry Image Generation Pipeline")
    print("=" * 50)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs("../test_output", exist_ok=True)
    
    # Test prompt
    test_prompt = "channel-set diamond eternity band, 2 mm width, hammered 18k yellow gold, product-only white background"
    print(f"Test prompt: {test_prompt}")
    
    try:
        # Initialize pipeline
        print("\n📦 Loading pipeline...")
        pipeline = ImprovedJewelryPipeline(device=device)
        print("✅ Pipeline loaded successfully!")
        
        # Generate baseline
        print("\n🎨 Generating baseline image...")
        baseline_img = pipeline.generate_baseline(
            prompt=test_prompt,
            num_images=1,
            seed=42,
            num_inference_steps=20  # Fast test
        )[0]
        baseline_path = "../test_output/test_baseline.png"
        baseline_img.save(baseline_path)
        print(f"✅ Baseline saved to: {baseline_path}")
        
        # Generate improved
        print("\n✨ Generating improved image...")
        improved_imgs = pipeline.generate_improved(
            prompt=test_prompt,
            num_images=2,  # Generate 2 candidates
            seed=42,
            num_inference_steps=20  # Fast test
        )
        improved_path = "../test_output/test_improved.png"
        improved_imgs[0].save(improved_path)
        print(f"✅ Improved saved to: {improved_path}")
        
        # Quick evaluation
        print("\n📊 Running evaluation...")
        evaluator = ComprehensiveEvaluator(device=device)
        
        baseline_eval = evaluator.evaluate_image(baseline_img, test_prompt)
        improved_eval = evaluator.evaluate_image(improved_imgs[0], test_prompt)
        
        print(f"\nResults:")
        print(f"  Baseline CLIP score:     {baseline_eval['overall_clip_similarity']:.4f}")
        print(f"  Improved CLIP score:     {improved_eval['overall_clip_similarity']:.4f}")
        print(f"  CLIP improvement:        {improved_eval['overall_clip_similarity'] - baseline_eval['overall_clip_similarity']:+.4f}")
        
        print(f"  Baseline aesthetic:      {baseline_eval['modern_aesthetic_score']:.4f}")
        print(f"  Improved aesthetic:      {improved_eval['modern_aesthetic_score']:.4f}")
        print(f"  Aesthetic improvement:   {improved_eval['modern_aesthetic_score'] - baseline_eval['modern_aesthetic_score']:+.4f}")
        
        print(f"\n🎉 Test completed successfully!")
        print(f"Check images in: ../test_output/")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
