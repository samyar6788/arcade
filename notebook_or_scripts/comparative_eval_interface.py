#!/usr/bin/env python3
"""
Comparative Evaluation Interface
Side-by-side comparison interface for human evaluation
"""

import streamlit as st
import pandas as pd
import os
from PIL import Image
import json
from datetime import datetime

def load_comparison_data(eval_dir):
    """Load comparison pairs and metadata"""
    metadata_file = None
    for file in os.listdir(eval_dir):
        if file.startswith("comparison_metadata") and file.endswith(".csv"):
            metadata_file = os.path.join(eval_dir, file)
            break
    
    if not metadata_file:
        st.error("No comparison metadata file found!")
        return None
        
    return pd.read_csv(metadata_file)

def save_comparison_result(pair_id, choice, confidence, reasoning, output_file):
    """Save comparison result to JSON file"""
    result_data = {
        'pair_id': pair_id,
        'timestamp': datetime.now().isoformat(),
        'choice': choice,  # 'A', 'B', or 'tie'
        'confidence': confidence,  # 1-5 scale
        'reasoning': reasoning
    }
    
    # Load existing results or create new file
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Update or add result
    existing_idx = None
    for i, result in enumerate(all_results):
        if result['pair_id'] == pair_id:
            existing_idx = i
            break
    
    if existing_idx is not None:
        all_results[existing_idx] = result_data
    else:
        all_results.append(result_data)
    
    # Save back to file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

def display_image_with_details(image_path, side_label, image_details, show_ai_scores=False):
    """Display image with technical details"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Image {side_label}")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=300)
        else:
            st.error(f"Image not found: {image_path}")
    
    with col2:
        st.subheader("Configuration")
        st.write(f"**Model:** {image_details['model']}")
        st.write(f"**Sampler:** {image_details['sampler']}")
        st.write(f"**Steps:** {image_details['steps']}")
        st.write(f"**CFG Scale:** {image_details['cfg']}")
        st.write(f"**Strategy:** {image_details['strategy']}")
        st.write(f"**Gen Time:** {image_details['gen_time']:.1f}s")
        
        if show_ai_scores:
            st.subheader("AI Scores")
            st.write(f"**CLIP:** {image_details['clip_score']:.3f}")
            st.write(f"**LAION:** {image_details['laion_score']:.2f}")
            st.write(f"**Quality:** {image_details['quality_score']:.3f}")

def main():
    st.set_page_config(page_title="Comparative Jewelry Evaluation", layout="wide")
    
    st.title("⚔️ Comparative Jewelry AI Evaluation")
    st.markdown("Compare two AI-generated jewelry images side by side")
    
    # Configuration
    eval_dir = st.text_input(
        "Evaluation Directory:",
        "/Users/samany/Library/CloudStorage/GoogleDrive-smn656@gmail.com/My Drive/arcade_comp_results/comparative_evaluation/comparative_eval_systematic"
    )
    
    if not os.path.exists(eval_dir):
        st.error(f"Directory not found: {eval_dir}")
        return
    
    # Load comparison data
    comparison_df = load_comparison_data(eval_dir)
    if comparison_df is None:
        return
    
    # Progress tracking
    results_file = os.path.join(eval_dir, "comparison_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        evaluated_pairs = {r['pair_id'] for r in existing_results}
    else:
        evaluated_pairs = set()
    
    total_pairs = len(comparison_df)
    completed = len(evaluated_pairs)
    progress_pct = completed / total_pairs if total_pairs > 0 else 0
    
    # Sidebar controls with enhanced progress display
    st.sidebar.metric("Progress", f"{completed}/{total_pairs}", f"{progress_pct*100:.1f}%")
    st.sidebar.progress(progress_pct)
    
    # Estimated time remaining
    if completed > 0:
        avg_time_per_eval = 2  # Assume 2 minutes per evaluation
        remaining_time = (total_pairs - completed) * avg_time_per_eval
        hours = remaining_time // 60
        minutes = remaining_time % 60
        if hours > 0:
            time_str = f"~{hours}h {minutes}m remaining"
        else:
            time_str = f"~{minutes}m remaining"
        st.sidebar.caption(time_str)
    
    show_ai_scores = st.sidebar.checkbox("Show AI Scores", value=False, 
                                        help="Reveal AI metrics (may bias evaluation)")
    
    filter_option = st.sidebar.radio("Show:", ["All", "Unevaluated Only", "Evaluated Only"])
    
    # Filter data
    if filter_option == "Unevaluated Only":
        available_df = comparison_df[~comparison_df['pair_id'].isin(evaluated_pairs)].reset_index(drop=True)
    elif filter_option == "Evaluated Only":
        available_df = comparison_df[comparison_df['pair_id'].isin(evaluated_pairs)].reset_index(drop=True)
    else:
        available_df = comparison_df.reset_index(drop=True)
    
    # Smart navigation: Auto-jump to next unevaluated if current filter changed
    if 'last_filter_option' not in st.session_state:
        st.session_state.last_filter_option = filter_option
    
    if st.session_state.last_filter_option != filter_option:
        st.session_state.current_pair_idx = 0  # Reset to first item when filter changes
        st.session_state.last_filter_option = filter_option
    
    if len(available_df) == 0:
        st.info("No comparison pairs to show with current filter")
        return
    
    # Pair selector with session state for navigation
    st.sidebar.subheader("Navigation")
    
    # Initialize session state for current pair index
    if 'current_pair_idx' not in st.session_state:
        st.session_state.current_pair_idx = 0
    
    # Ensure current index is within bounds
    if st.session_state.current_pair_idx >= len(available_df):
        st.session_state.current_pair_idx = 0
    
    current_idx = st.session_state.current_pair_idx
    current_pair = available_df.iloc[current_idx]
    
    # Display current pair info
    st.sidebar.metric("Current Pair", f"{current_idx + 1} / {len(available_df)}")
    st.sidebar.write(f"**Pair ID:** {current_pair['pair_id']}")
    st.sidebar.write(f"**Type:** {current_pair['comparison_type'].replace('_', ' ').title()}")
    
    # Navigation buttons in sidebar
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("⬅️ Previous", disabled=(current_idx == 0)):
            st.session_state.current_pair_idx = max(0, current_idx - 1)
            st.rerun()
    
    with col2:
        if st.button("➡️ Next", disabled=(current_idx == len(available_df) - 1)):
            st.session_state.current_pair_idx = min(len(available_df) - 1, current_idx + 1)
            st.rerun()
    
    # Quick jump options
    st.sidebar.subheader("Quick Jump")
    
    # Jump to specific pair number
    target_pair = st.sidebar.number_input(
        "Go to pair #:", 
        min_value=1, 
        max_value=len(available_df), 
        value=current_idx + 1,
        step=1
    )
    
    if st.sidebar.button("🎯 Jump to Pair"):
        st.session_state.current_pair_idx = target_pair - 1
        st.rerun()
    
    # Advanced selector (for when you need it)
    with st.sidebar.expander("Advanced Selection"):
        selected_idx = st.selectbox(
            "Select Comparison Pair:",
            range(len(available_df)),
            index=current_idx,
            format_func=lambda x: f"{available_df.iloc[x]['pair_id']} - {available_df.iloc[x]['comparison_type']}",
            key="advanced_selector"
        )
        
        if st.button("🔄 Update Selection"):
            st.session_state.current_pair_idx = selected_idx
            st.rerun()
    pair_id = current_pair['pair_id']
    
    # Display comparison context
    st.subheader("🎯 Comparison Context")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Comparison Type", current_pair['comparison_type'].replace('_', ' ').title())
    with col2:
        st.metric("Variable Dimension", current_pair['variable_dimension'].replace('_', ' ').title())
    with col3:
        st.metric("Prompt ID", current_pair['prompt_id'])
    
    # Display prompt
    st.subheader("📝 Prompt")
    st.write(f"**Original Prompt:** {current_pair['original_prompt']}")
    st.write(f"**Shared Config:** {current_pair['shared_config']}")
    
    # Main comparison area
    st.subheader("🔍 Side-by-Side Comparison")
    
    col_a, col_space, col_b = st.columns([5, 1, 5])
    
    # Image A details
    image_a_details = {
        'model': current_pair['image_a_model'],
        'sampler': current_pair['image_a_sampler'],
        'steps': current_pair['image_a_steps'],
        'cfg': current_pair['image_a_cfg'],
        'strategy': current_pair['image_a_strategy'],
        'gen_time': current_pair['image_a_gen_time'],
        'clip_score': current_pair['image_a_clip_score'],
        'laion_score': current_pair['image_a_laion_score'],
        'quality_score': current_pair['image_a_quality_score']
    }
    
    # Image B details
    image_b_details = {
        'model': current_pair['image_b_model'],
        'sampler': current_pair['image_b_sampler'],
        'steps': current_pair['image_b_steps'],
        'cfg': current_pair['image_b_cfg'],
        'strategy': current_pair['image_b_strategy'],
        'gen_time': current_pair['image_b_gen_time'],
        'clip_score': current_pair['image_b_clip_score'],
        'laion_score': current_pair['image_b_laion_score'],
        'quality_score': current_pair['image_b_quality_score']
    }
    
    with col_a:
        display_image_with_details(current_pair['image_a_path'], "A", image_a_details, show_ai_scores)
    
    with col_b:
        display_image_with_details(current_pair['image_b_path'], "B", image_b_details, show_ai_scores)
    
    # Evaluation interface
    st.subheader("🎯 Your Evaluation")
    
    # Load existing evaluation if available
    existing_result = None
    if pair_id in evaluated_pairs:
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        for result in all_results:
            if result['pair_id'] == pair_id:
                existing_result = result
                break
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Which image is better?**")
        choice = st.radio(
            "Choice:",
            ["A is better", "B is better", "Tie/Equal"],
            index=["A", "B", "tie"].index(existing_result['choice']) if existing_result else 0,
            key=f"choice_{pair_id}"
        )
        choice_value = {"A is better": "A", "B is better": "B", "Tie/Equal": "tie"}[choice]
    
    with col2:
        st.write("**How confident are you?**")
        confidence = st.slider(
            "Confidence:",
            1, 5, 
            existing_result['confidence'] if existing_result else 3,
            help="1=Very unsure, 5=Very confident",
            key=f"confidence_{pair_id}"
        )
    
    with col3:
        st.write("**Why?**")
        reasoning = st.text_area(
            "Reasoning:",
            existing_result['reasoning'] if existing_result else "",
            help="What makes one better? (quality, realism, accuracy, etc.)",
            key=f"reasoning_{pair_id}"
        )
    
    # AI prediction display (if available)
    if current_pair['ai_prediction'] and show_ai_scores:
        ai_pred = current_pair['ai_prediction']
        ai_better = "A" if "A" in ai_pred else "B" if "B" in ai_pred else "Unknown"
        
        if ai_better != "Unknown":
            agreement = "✅ Agree" if choice_value == ai_better else "❌ Disagree"
            st.info(f"🤖 AI predicts: Image {ai_better} is better | Your choice: {agreement}")
    
    # Save evaluation and navigation
    st.subheader("💾 Save & Navigate")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("💾 Save & Next ➡️", type="primary", use_container_width=True):
            save_comparison_result(pair_id, choice_value, confidence, reasoning, results_file)
            # Auto-advance to next pair
            if current_idx < len(available_df) - 1:
                st.session_state.current_pair_idx = current_idx + 1
            st.success("✅ Evaluation saved!")
            st.rerun()
    
    with col2:
        if st.button("💾 Save Only", use_container_width=True):
            save_comparison_result(pair_id, choice_value, confidence, reasoning, results_file)
            st.success("✅ Saved!")
            st.rerun()
    
    with col3:
        if st.button("⏭️ Skip", use_container_width=True):
            if current_idx < len(available_df) - 1:
                st.session_state.current_pair_idx = current_idx + 1
                st.rerun()
    
    with col4:
        if st.button("🎲 Random", use_container_width=True):
            import random
            st.session_state.current_pair_idx = random.randint(0, len(available_df) - 1)
            st.rerun()
    
    # Keyboard shortcuts info
    st.caption("💡 **Tip:** Use 'Save & Next' to quickly evaluate pairs sequentially")
    
    # Quick stats
    if completed > 0:
        st.subheader("📊 Quick Stats")
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        
        choice_counts = {}
        for result in all_results:
            choice = result['choice']
            choice_counts[choice] = choice_counts.get(choice, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("A Wins", choice_counts.get('A', 0))
        with col2:
            st.metric("B Wins", choice_counts.get('B', 0))
        with col3:
            st.metric("Ties", choice_counts.get('tie', 0))

if __name__ == "__main__":
    main()
