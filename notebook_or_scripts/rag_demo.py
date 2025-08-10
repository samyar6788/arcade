"""
Demo script showing how to integrate Image RAG with your existing jewelry pipeline
Run this to see RAG enhancements in action!
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Import your existing modules
from jewelry_improvements import ImprovedJewelryPipeline
from evaluation_metrics import ComprehensiveEvaluator

# Import the new Image RAG system
from image_rag import JewelryImageRAG, RAGEnhancedJewelryPipeline


def demo_rag_integration():
    """Demonstrate Image RAG integration with existing pipeline"""
    
    print("🎭 RAG Integration Demo")
    print("=" * 50)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize your existing pipeline
    print("\n📦 Loading base jewelry pipeline...")
    base_pipeline = ImprovedJewelryPipeline(device=device)
    
    # Initialize RAG-enhanced pipeline
    print("\n🔍 Creating RAG-enhanced pipeline...")
    rag_pipeline = RAGEnhancedJewelryPipeline(base_pipeline, device=device)
    
    # Test with one of your assignment prompts
    test_prompt = "channel-set diamond eternity band, 2 mm width, hammered 18k yellow gold, product-only white background"
    
    print(f"\n🧪 Testing prompt: {test_prompt}")
    
    # Generate WITHOUT RAG (your original method)
    print("\n1️⃣ Generating WITHOUT RAG...")
    baseline_images = base_pipeline.generate_improved(
        prompt=test_prompt,
        num_images=1,
        seed=42,
        num_inference_steps=20  # Fast for demo
    )
    
    # Generate WITH RAG enhancement
    print("\n2️⃣ Generating WITH RAG...")
    rag_images, rag_analysis = rag_pipeline.generate_with_rag(
        prompt=test_prompt,
        k_retrievals=3,
        use_rag=True,
        num_images=1,
        seed=42,
        num_inference_steps=20  # Fast for demo
    )
    
    # Show the enhancement
    print(f"\n📊 RAG Enhancement Results:")
    print(f"Original prompt: {rag_analysis['original_prompt']}")
    print(f"Enhanced prompt: {rag_analysis['enhanced_prompt']}")
    print(f"Confidence: {rag_analysis.get('retrieval_confidence', 0):.3f}")
    
    # Optional: Evaluate differences
    print("\n📈 Evaluating improvements...")
    evaluator = ComprehensiveEvaluator(device=device)
    
    baseline_eval = evaluator.evaluate_image(baseline_images[0], test_prompt)
    rag_eval = evaluator.evaluate_image(rag_images[0], test_prompt)
    
    print(f"Baseline CLIP score: {baseline_eval['overall_clip_similarity']:.4f}")
    print(f"RAG CLIP score:      {rag_eval['overall_clip_similarity']:.4f}")
    print(f"Improvement:         {rag_eval['overall_clip_similarity'] - baseline_eval['overall_clip_similarity']:+.4f}")
    
    print(f"Baseline aesthetic:  {baseline_eval['modern_aesthetic_score']:.4f}")
    print(f"RAG aesthetic:       {rag_eval['modern_aesthetic_score']:.4f}")
    print(f"Improvement:         {rag_eval['modern_aesthetic_score'] - baseline_eval['modern_aesthetic_score']:+.4f}")
    
    # Save comparison
    print(f"\n💾 Saving comparison images...")
    baseline_images[0].save("../test_output/baseline_no_rag.png")
    rag_images[0].save("../test_output/enhanced_with_rag.png")
    
    print(f"✅ Demo complete! Check ../test_output/ for image comparison")
    
    return {
        'baseline_image': baseline_images[0],
        'rag_image': rag_images[0],
        'rag_analysis': rag_analysis,
        'baseline_eval': baseline_eval,
        'rag_eval': rag_eval
    }


def demo_standalone_rag():
    """Demonstrate standalone RAG system functionality"""
    
    print("\n🔬 Standalone RAG Demo")
    print("=" * 30)
    
    # Initialize RAG system
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag_system = JewelryImageRAG(device=device)
    
    # Test all 8 assignment prompts
    assignment_prompts = [
        "channel-set diamond eternity band, 2 mm width, hammered 18k yellow gold, product-only white background",
        "14k rose-gold threader earrings, bezel-set round lab diamond ends, lifestyle macro shot, soft natural light",
        "organic cluster ring with mixed-cut sapphires and diamonds, brushed platinum finish, modern aesthetic",
        "A solid gold cuff bracelet with blue sapphire, with refined simplicity and intentionally crafted for everyday wear",
        "modern signet ring, oval face, engraved gothic initial 'M', high-polish sterling silver, subtle reflection",
        "delicate gold huggie hoops, contemporary styling, isolated on neutral background",
        "stack of three slim rings: twisted gold, plain platinum, black rhodium pavé, editorial lighting",
        "bypass ring with stones on it, with refined simplicity and intentionally crafted for everyday wear"
    ]
    
    enhancements = []
    
    for i, prompt in enumerate(assignment_prompts, 1):
        print(f"\n📝 Prompt {i}: {prompt[:50]}...")
        
        # Analyze with RAG
        analysis = rag_system.analyze_prompt_and_retrieve(prompt, k=3)
        
        enhancement_info = {
            'prompt_id': i,
            'original': analysis['original_prompt'],
            'enhanced': analysis['enhanced_prompt'],
            'jewelry_type': analysis['detected_jewelry_type'],
            'confidence': analysis['retrieval_confidence'],
            'enhancement_added': len(analysis['enhanced_prompt']) - len(analysis['original_prompt'])
        }
        
        enhancements.append(enhancement_info)
        
        print(f"  Type: {enhancement_info['jewelry_type']}")
        print(f"  Confidence: {enhancement_info['confidence']:.3f}")
        print(f"  Enhancement: +{enhancement_info['enhancement_added']} chars")
    
    # Summary
    print(f"\n📊 RAG Enhancement Summary:")
    avg_confidence = np.mean([e['confidence'] for e in enhancements])
    avg_enhancement = np.mean([e['enhancement_added'] for e in enhancements])
    
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Average enhancement: +{avg_enhancement:.1f} characters")
    print(f"  Categories detected: {len(set([e['jewelry_type'] for e in enhancements]))}")
    
    return enhancements


def demo_custom_database():
    """Show how to set up a custom jewelry database"""
    
    print("\n🗄️ Custom Database Demo")
    print("=" * 25)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # This would be used with real jewelry images
    print("📁 Example: Setting up custom database")
    print("   (This is how you'd use it with real jewelry images)")
    
    example_code = """
    # Initialize RAG system
    rag_system = JewelryImageRAG(device=device)
    
    # Index your jewelry image collection
    # Folder should contain images like: mejuri_ring_01.jpg, catbird_earrings_02.jpg, etc.
    rag_system.index_jewelry_dataset(
        image_folder="path/to/your/jewelry/images",
        metadata_file="path/to/metadata.json"  # Optional
    )
    
    # Save the database for reuse
    rag_system.save_database("custom_jewelry_db")
    
    # Later, load the database
    rag_system.load_database("custom_jewelry_db")
    """
    
    print(example_code)
    
    # Show metadata format
    print("\n📋 Example metadata.json format:")
    example_metadata = {
        "mejuri_ring_solitaire_01.jpg": {
            "style": "modern minimalist solitaire contemporary",
            "material": "18k-gold diamond",
            "brand_style": "mejuri clean lines precision luxury",
            "setting_type": "prong-set solitaire",
            "aesthetic": "contemporary luxury refined"
        },
        "catbird_huggie_hoops_02.jpg": {
            "style": "delicate everyday minimal contemporary",
            "material": "14k-gold",
            "brand_style": "catbird subtle luxury daily wear",
            "setting_type": "seamless closure",
            "aesthetic": "modern minimalist refined"
        }
    }
    
    import json
    print(json.dumps(example_metadata, indent=2))


if __name__ == "__main__":
    print("🚀 Image RAG Demo Suite")
    print("=" * 60)
    
    # Run all demos
    try:
        # Demo 1: Integration with existing pipeline
        integration_results = demo_rag_integration()
        
        # Demo 2: Standalone RAG analysis
        standalone_results = demo_standalone_rag()
        
        # Demo 3: Custom database setup
        demo_custom_database()
        
        print(f"\n🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
