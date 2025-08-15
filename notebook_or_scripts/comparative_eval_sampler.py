#!/usr/bin/env python3
"""
Comparative Evaluation Sampler
Creates side-by-side comparison pairs for human evaluation
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import random
from itertools import combinations

def create_comparison_pairs(csv_path, images_dir, output_dir, comparison_strategy="systematic"):
    """
    Create comparison pairs for side-by-side evaluation
    
    Args:
        csv_path: Path to ultimate_comprehensive_results.csv
        images_dir: Directory containing all images
        output_dir: Directory to save comparison pairs
        comparison_strategy: "systematic", "best_vs_worst", "model_comparison", "sampler_comparison"
    """
    
    # Load results
    df = pd.read_csv(csv_path)
    df_success = df[df['image_path'].notna()].copy()
    
    print(f"📊 Loaded {len(df_success)} successful generations")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    if comparison_strategy == "systematic":
        return systematic_comparisons(df_success, images_dir, output_dir)
    elif comparison_strategy == "best_vs_worst":
        return best_vs_worst_comparisons(df_success, images_dir, output_dir)
    elif comparison_strategy == "model_comparison":
        return model_comparisons(df_success, images_dir, output_dir)
    elif comparison_strategy == "sampler_comparison":
        return sampler_comparisons(df_success, images_dir, output_dir)
    elif comparison_strategy == "strategy_comparison":
        return strategy_comparisons(df_success, images_dir, output_dir)

def systematic_comparisons(df, images_dir, output_dir, max_pairs=200):
    """
    Create systematic comparisons across all key dimensions - OPTIMIZED VERSION
    """
    comparison_pairs = []
    pairs_per_prompt = max_pairs // len(df['prompt_id'].unique())  # Distribute evenly across prompts
    
    print(f"🔄 Creating systematic comparison pairs (max {pairs_per_prompt} per prompt)...")
    
    # For each prompt, create targeted comparisons
    for prompt_id in df['prompt_id'].unique():
        prompt_data = df[df['prompt_id'] == prompt_id]
        
        if len(prompt_data) < 2:
            continue
            
        print(f"  Processing prompt {prompt_id} ({len(prompt_data)} images) -> {pairs_per_prompt} pairs...")
        
        prompt_pairs = []
        
        # STRATEGY 1: Sample representative configurations first
        # Get unique parameter combinations
        unique_models = prompt_data['model'].unique()
        unique_samplers = prompt_data['sampler'].unique()
        unique_strategies = prompt_data['strategy'].unique()
        unique_steps = prompt_data['steps'].unique()
        unique_cfg = prompt_data['cfg_scale'].unique()
        
        # OPTIMIZED: Create targeted comparisons rather than exhaustive
        comparison_categories = [
            ('model_comparison', 'model'),
            ('sampler_comparison', 'sampler'),  
            ('strategy_comparison', 'strategy'),
            ('cfg_comparison', 'cfg_scale'),
            ('steps_comparison', 'steps')
        ]
        
        target_pairs_per_category = pairs_per_prompt // len(comparison_categories)
        
        for comp_type, variable_param in comparison_categories:
            category_pairs = 0
            
            # SMARTER APPROACH: Find valid pairs directly instead of random sampling
            if comp_type == 'model_comparison':
                # Group by sampler+strategy+steps+cfg, find groups with multiple models
                grouped = prompt_data.groupby(['sampler', 'strategy', 'steps', 'cfg_scale'])
                for group_key, group_data in grouped:
                    if len(group_data['model'].unique()) >= 2:
                        # Create pairs within this group
                        for idx1, idx2 in combinations(group_data.index, 2):
                            if group_data.loc[idx1, 'model'] != group_data.loc[idx2, 'model']:
                                img1 = group_data.loc[idx1]
                                img2 = group_data.loc[idx2]
                                shared_config = f"{img1['sampler']}_{img1['strategy']}_{img1['steps']}s_{img1['cfg_scale']}cfg"
                                
                                prompt_pairs.append({
                                    'comparison_type': comp_type,
                                    'variable_dimension': variable_param,
                                    'prompt_id': prompt_id,
                                    'image_a': img1,
                                    'image_b': img2,
                                    'shared_config': shared_config
                                })
                                category_pairs += 1
                                if category_pairs >= target_pairs_per_category:
                                    break
                    if category_pairs >= target_pairs_per_category:
                        break
                        
            elif comp_type == 'sampler_comparison':
                # Group by model+strategy+steps+cfg, find groups with multiple samplers
                grouped = prompt_data.groupby(['model', 'strategy', 'steps', 'cfg_scale'])
                for group_key, group_data in grouped:
                    if len(group_data['sampler'].unique()) >= 2:
                        for idx1, idx2 in combinations(group_data.index, 2):
                            if group_data.loc[idx1, 'sampler'] != group_data.loc[idx2, 'sampler']:
                                img1 = group_data.loc[idx1]
                                img2 = group_data.loc[idx2]
                                shared_config = f"{img1['model']}_{img1['strategy']}_{img1['steps']}s_{img1['cfg_scale']}cfg"
                                
                                prompt_pairs.append({
                                    'comparison_type': comp_type,
                                    'variable_dimension': variable_param,
                                    'prompt_id': prompt_id,
                                    'image_a': img1,
                                    'image_b': img2,
                                    'shared_config': shared_config
                                })
                                category_pairs += 1
                                if category_pairs >= target_pairs_per_category:
                                    break
                    if category_pairs >= target_pairs_per_category:
                        break
                        
            elif comp_type == 'strategy_comparison':
                # Group by model+sampler+steps+cfg, find groups with multiple strategies
                grouped = prompt_data.groupby(['model', 'sampler', 'steps', 'cfg_scale'])
                for group_key, group_data in grouped:
                    if len(group_data['strategy'].unique()) >= 2:
                        for idx1, idx2 in combinations(group_data.index, 2):
                            if group_data.loc[idx1, 'strategy'] != group_data.loc[idx2, 'strategy']:
                                img1 = group_data.loc[idx1]
                                img2 = group_data.loc[idx2]
                                shared_config = f"{img1['model']}_{img1['sampler']}_{img1['steps']}s_{img1['cfg_scale']}cfg"
                                
                                prompt_pairs.append({
                                    'comparison_type': comp_type,
                                    'variable_dimension': variable_param,
                                    'prompt_id': prompt_id,
                                    'image_a': img1,
                                    'image_b': img2,
                                    'shared_config': shared_config
                                })
                                category_pairs += 1
                                if category_pairs >= target_pairs_per_category:
                                    break
                    if category_pairs >= target_pairs_per_category:
                        break
                        
            elif comp_type == 'cfg_comparison':
                # Group by model+sampler+strategy+steps, find groups with multiple CFG scales
                grouped = prompt_data.groupby(['model', 'sampler', 'strategy', 'steps'])
                for group_key, group_data in grouped:
                    if len(group_data['cfg_scale'].unique()) >= 2:
                        for idx1, idx2 in combinations(group_data.index, 2):
                            if group_data.loc[idx1, 'cfg_scale'] != group_data.loc[idx2, 'cfg_scale']:
                                img1 = group_data.loc[idx1]
                                img2 = group_data.loc[idx2]
                                shared_config = f"{img1['model']}_{img1['sampler']}_{img1['strategy']}_{img1['steps']}s"
                                
                                prompt_pairs.append({
                                    'comparison_type': comp_type,
                                    'variable_dimension': variable_param,
                                    'prompt_id': prompt_id,
                                    'image_a': img1,
                                    'image_b': img2,
                                    'shared_config': shared_config
                                })
                                category_pairs += 1
                                if category_pairs >= target_pairs_per_category:
                                    break
                    if category_pairs >= target_pairs_per_category:
                        break
                        
            elif comp_type == 'steps_comparison':
                # Group by model+sampler+strategy+cfg, find groups with multiple step counts
                grouped = prompt_data.groupby(['model', 'sampler', 'strategy', 'cfg_scale'])
                for group_key, group_data in grouped:
                    if len(group_data['steps'].unique()) >= 2:
                        for idx1, idx2 in combinations(group_data.index, 2):
                            if group_data.loc[idx1, 'steps'] != group_data.loc[idx2, 'steps']:
                                img1 = group_data.loc[idx1]
                                img2 = group_data.loc[idx2]
                                shared_config = f"{img1['model']}_{img1['sampler']}_{img1['strategy']}_{img1['cfg_scale']}cfg"
                                
                                prompt_pairs.append({
                                    'comparison_type': comp_type,
                                    'variable_dimension': variable_param,
                                    'prompt_id': prompt_id,
                                    'image_a': img1,
                                    'image_b': img2,
                                    'shared_config': shared_config
                                })
                                category_pairs += 1
                                if category_pairs >= target_pairs_per_category:
                                    break
                    if category_pairs >= target_pairs_per_category:
                        break
        
        # Add this prompt's pairs to the total
        comparison_pairs.extend(prompt_pairs[:pairs_per_prompt])  # Limit per prompt
        
        if len(comparison_pairs) >= max_pairs:
            print(f"🎯 Reached target of {max_pairs} pairs, stopping early")
            break
    
    # Final sampling if we still have too many
    if len(comparison_pairs) > max_pairs:
        print(f"🎲 Final sampling: {max_pairs} pairs from {len(comparison_pairs)} generated pairs")
        comparison_pairs = random.sample(comparison_pairs, max_pairs)
    
    print(f"✅ Created {len(comparison_pairs)} systematic comparison pairs")
    return create_comparison_interface_data(comparison_pairs, images_dir, output_dir, "systematic")

def best_vs_worst_comparisons(df, images_dir, output_dir, pairs_per_prompt=10):
    """
    Create comparisons between best and worst performers for each prompt
    """
    comparison_pairs = []
    
    print("🏆 Creating best vs worst comparison pairs...")
    
    # Create composite quality score if not present
    if 'quality_score' not in df.columns:
        print("📊 Creating composite quality score from CLIP and LAION metrics...")
        df = df.copy()
        # Normalize both scores to 0-1 scale
        clip_norm = (df['clip_top_confidence'] - df['clip_top_confidence'].min()) / (df['clip_top_confidence'].max() - df['clip_top_confidence'].min())
        laion_norm = df['laion_aesthetic_score'] / 10.0  # LAION is 0-10 scale
        # Weighted combination (60% CLIP, 40% LAION)
        df['quality_score'] = 0.6 * clip_norm + 0.4 * laion_norm
    
    for prompt_id in df['prompt_id'].unique():
        prompt_data = df[df['prompt_id'] == prompt_id]
        
        if len(prompt_data) < 4:  # Need enough for best/worst comparison
            continue
            
        # Sort by AI quality score
        sorted_data = prompt_data.sort_values('quality_score')
        
        # Get top and bottom performers
        top_performers = sorted_data.tail(pairs_per_prompt // 2)
        bottom_performers = sorted_data.head(pairs_per_prompt // 2)
        
        # Create all combinations of top vs bottom
        for _, top_img in top_performers.iterrows():
            for _, bottom_img in bottom_performers.iterrows():
                comparison_pairs.append({
                    'comparison_type': 'quality_extreme',
                    'variable_dimension': 'ai_quality_ranking',
                    'prompt_id': prompt_id,
                    'image_a': top_img,
                    'image_b': bottom_img,
                    'shared_config': f"best_vs_worst_quality",
                    'ai_prediction': 'A_better'  # AI thinks A is better
                })
    
    return create_comparison_interface_data(comparison_pairs, images_dir, output_dir, "best_vs_worst")

def model_comparisons(df, images_dir, output_dir, max_pairs=150):
    """
    Focus specifically on model comparisons
    """
    comparison_pairs = []
    
    print("🤖 Creating model-focused comparison pairs...")
    
    models = df['model'].unique()
    
    for prompt_id in df['prompt_id'].unique():
        prompt_data = df[df['prompt_id'] == prompt_id]
        
        # For each model pair
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                model_a_data = prompt_data[prompt_data['model'] == model_a]
                model_b_data = prompt_data[prompt_data['model'] == model_b]
                
                if len(model_a_data) > 0 and len(model_b_data) > 0:
                    # Get best performer from each model
                    best_a = model_a_data.loc[model_a_data['quality_score'].idxmax()]
                    best_b = model_b_data.loc[model_b_data['quality_score'].idxmax()]
                    
                    comparison_pairs.append({
                        'comparison_type': 'model_best_vs_best',
                        'variable_dimension': 'model',
                        'prompt_id': prompt_id,
                        'image_a': best_a,
                        'image_b': best_b,
                        'shared_config': f"{model_a}_vs_{model_b}_best"
                    })
    
    # Sample down if needed
    if len(comparison_pairs) > max_pairs:
        comparison_pairs = random.sample(comparison_pairs, max_pairs)
    
    return create_comparison_interface_data(comparison_pairs, images_dir, output_dir, "model_comparison")

def create_comparison_interface_data(comparison_pairs, images_dir, output_dir, strategy_name):
    """
    Create the comparison interface data structure and copy images
    """
    eval_dir = os.path.join(output_dir, f"comparative_eval_{strategy_name}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create pairs subdirectory
    pairs_dir = os.path.join(eval_dir, "comparison_pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    
    comparison_data = []
    
    for i, pair in enumerate(comparison_pairs):
        pair_id = f"pair_{i:04d}"
        
        # Extract image info
        img_a = pair['image_a']
        img_b = pair['image_b']
        
        # Create filenames
        img_a_filename = os.path.basename(img_a['image_path'])
        img_b_filename = os.path.basename(img_b['image_path'])
        
        # Copy images to pair directory
        pair_subdir = os.path.join(pairs_dir, pair_id)
        os.makedirs(pair_subdir, exist_ok=True)
        
        img_a_path = os.path.join(images_dir, img_a_filename)
        img_b_path = os.path.join(images_dir, img_b_filename)
        
        img_a_dest = os.path.join(pair_subdir, "image_A.png")
        img_b_dest = os.path.join(pair_subdir, "image_B.png")
        
        if os.path.exists(img_a_path) and os.path.exists(img_b_path):
            shutil.copy2(img_a_path, img_a_dest)
            shutil.copy2(img_b_path, img_b_dest)
            
            # Store comparison metadata
            comparison_data.append({
                'pair_id': pair_id,
                'comparison_type': pair['comparison_type'],
                'variable_dimension': pair['variable_dimension'],
                'prompt_id': pair['prompt_id'],
                'original_prompt': img_a['original_prompt'],
                'shared_config': pair['shared_config'],
                
                # Image A details
                'image_a_model': img_a['model'],
                'image_a_sampler': img_a['sampler'],
                'image_a_steps': img_a['steps'],
                'image_a_cfg': img_a['cfg_scale'],
                'image_a_strategy': img_a['strategy'],
                'image_a_clip_score': img_a.get('clip_top_confidence', ''),
                'image_a_laion_score': img_a.get('laion_aesthetic_score', ''),
                'image_a_quality_score': img_a.get('quality_score', ''),
                'image_a_gen_time': img_a.get('generation_time', ''),
                
                # Image B details
                'image_b_model': img_b['model'],
                'image_b_sampler': img_b['sampler'],
                'image_b_steps': img_b['steps'],
                'image_b_cfg': img_b['cfg_scale'],
                'image_b_strategy': img_b['strategy'],
                'image_b_clip_score': img_b.get('clip_top_confidence', ''),
                'image_b_laion_score': img_b.get('laion_aesthetic_score', ''),
                'image_b_quality_score': img_b.get('quality_score', ''),
                'image_b_gen_time': img_b.get('generation_time', ''),
                
                # AI predictions
                'ai_prediction': pair.get('ai_prediction', ''),
                
                # File paths
                'image_a_path': img_a_dest,
                'image_b_path': img_b_dest
            })
    
    # Save metadata
    metadata_df = pd.DataFrame(comparison_data)
    metadata_path = os.path.join(eval_dir, f"comparison_metadata_{strategy_name}.csv")
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"✅ Created {len(comparison_data)} comparison pairs in {eval_dir}")
    print(f"📋 Metadata saved to {metadata_path}")
    
    return eval_dir, metadata_path

if __name__ == "__main__":
    # Configuration
    csv_path = "/Users/samany/Library/CloudStorage/GoogleDrive-smn656@gmail.com/My Drive/arcade_comp_results/combined_experiment_results/ultimate_comprehensive_results.csv"
    images_dir = "/Users/samany/Library/CloudStorage/GoogleDrive-smn656@gmail.com/My Drive/arcade_comp_results/combined_experiment_results"
    output_dir = "/Users/samany/Library/CloudStorage/GoogleDrive-smn656@gmail.com/My Drive/arcade_comp_results/comparative_evaluation"
    
    print("🎯 Creating comparative evaluation pairs...")
    
    # Create systematic comparisons only (others have file path issues)
    strategies = ["systematic"]  # Only run the working one
    
    for strategy in strategies:
        print(f"\n🔄 Creating {strategy} comparisons...")
        eval_dir, metadata_path = create_comparison_pairs(
            csv_path, images_dir, output_dir, strategy
        )
    
    print(f"\n🎉 Comparative evaluation setup complete!")
    print(f"📁 Results in: {output_dir}")
