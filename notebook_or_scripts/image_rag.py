"""
Image RAG (Retrieval-Augmented Generation) System for Jewelry
A complete implementation of image-based retrieval to enhance jewelry generation
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import requests
from io import BytesIO


class ImageEmbeddingExtractor:
    """Extract embeddings from images using CLIP"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = device
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("✅ CLIP model loaded successfully")
        
    def extract_image_embedding(self, image: Union[Image.Image, str, bytes]) -> np.ndarray:
        """Extract embedding from a single image"""
        if isinstance(image, str):
            if image.startswith('http'):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
            
        # Ensure RGB
        image = image.convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize embedding for cosine similarity
            image_features = F.normalize(image_features, p=2, dim=1)
            
        return image_features.cpu().numpy().flatten()
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract embedding from text using CLIP text encoder"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1)
            
        return text_features.cpu().numpy().flatten()


class ImageVectorStore:
    """Vector store for image embeddings with FAISS backend"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.image_metadata = []  # Store metadata for each image
        
    def add_images(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add image embeddings and metadata to the store"""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(embeddings.astype('float32'))
        self.image_metadata.extend(metadata)
        print(f"Added {len(metadata)} images to vector store (total: {len(self.image_metadata)})")
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        """Search for similar images"""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_metadata) and idx != -1:  # -1 indicates no match found
                results.append((float(score), self.image_metadata[idx]))
                
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, f"{filepath}.faiss")
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(self.image_metadata, f)
        print(f"💾 Vector store saved to {filepath}")
            
    def load(self, filepath: str):
        """Load the vector store from disk"""
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}_metadata.pkl"):
            self.index = faiss.read_index(f"{filepath}.faiss")
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                self.image_metadata = pickle.load(f)
            print(f"📂 Vector store loaded from {filepath} ({len(self.image_metadata)} images)")
            return True
        return False


class JewelryImageRAG:
    """Image RAG system specifically for jewelry"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.embedding_extractor = ImageEmbeddingExtractor(device=device)
        self.vector_store = ImageVectorStore()
        
        # Jewelry category mapping
        self.jewelry_categories = {
            'rings': ['ring', 'band', 'wedding', 'engagement', 'signet', 'eternity', 'bypass'],
            'earrings': ['earring', 'stud', 'hoop', 'drop', 'threader', 'huggie'],
            'necklaces': ['necklace', 'pendant', 'chain', 'choker'],
            'bracelets': ['bracelet', 'bangle', 'cuff', 'tennis'],
            'watches': ['watch', 'timepiece']
        }
        
        # Create sample jewelry database for demo
        self._create_sample_jewelry_database()
        
    def _create_sample_jewelry_database(self):
        """Create a sample jewelry database with synthetic metadata for demonstration"""
        print("🔍 Creating sample jewelry database...")
        
        # High-quality jewelry descriptions inspired by modern brands
        sample_jewelry_data = [
            {
                'description': 'modern diamond engagement ring platinum setting solitaire',
                'category': 'rings',
                'style': 'modern minimalist contemporary solitaire',
                'material': 'platinum diamond',
                'brand_style': 'mejuri-inspired clean lines precision setting',
                'setting_type': 'solitaire prong-set',
                'aesthetic': 'contemporary luxury refined'
            },
            {
                'description': 'gold huggie hoop earrings small delicate everyday wear',
                'category': 'earrings', 
                'style': 'contemporary delicate minimal everyday',
                'material': '14k-gold',
                'brand_style': 'catbird-inspired subtle luxury daily wear',
                'setting_type': 'seamless closure',
                'aesthetic': 'modern minimalist refined'
            },
            {
                'description': 'channel-set diamond eternity band yellow gold continuous diamonds',
                'category': 'rings',
                'style': 'classic elegant refined traditional',
                'material': '18k-yellow-gold diamond',
                'brand_style': 'traditional craftsmanship modern finish precision',
                'setting_type': 'channel-set continuous',
                'aesthetic': 'timeless elegant sophisticated'
            },
            {
                'description': 'threader earrings rose gold diamond bezel contemporary',
                'category': 'earrings',
                'style': 'modern delicate contemporary linear',
                'material': '14k-rose-gold diamond',
                'brand_style': 'vrai-inspired sophisticated simplicity modern',
                'setting_type': 'bezel-set secure',
                'aesthetic': 'contemporary refined delicate'
            },
            {
                'description': 'signet ring sterling silver engraved oval face gothic initial',
                'category': 'rings',
                'style': 'modern classic personalized traditional',
                'material': 'sterling-silver',
                'brand_style': 'contemporary traditional blend heritage modern',
                'setting_type': 'engraved surface',
                'aesthetic': 'classic modern personalized'
            },
            {
                'description': 'cuff bracelet gold sapphire minimalist everyday luxury',
                'category': 'bracelets',
                'style': 'minimalist contemporary refined open',
                'material': '18k-gold sapphire',
                'brand_style': 'luxury everyday wear sophisticated simple',
                'setting_type': 'bezel-set accent',
                'aesthetic': 'minimalist luxury contemporary'
            },
            {
                'description': 'cluster ring mixed-cut sapphires diamonds platinum organic',
                'category': 'rings',
                'style': 'organic contemporary artistic unique',
                'material': 'platinum sapphire diamond',
                'brand_style': 'artisanal modern organic natural asymmetric',
                'setting_type': 'mixed-cut cluster prong',
                'aesthetic': 'organic contemporary artistic'
            },
            {
                'description': 'bypass ring stones open band contemporary design',
                'category': 'rings',
                'style': 'contemporary modern open geometric',
                'material': 'mixed-metal gemstone',
                'brand_style': 'modern geometric asymmetric contemporary',
                'setting_type': 'bypass open-band',
                'aesthetic': 'contemporary geometric modern'
            },
            {
                'description': 'stacked rings twisted gold platinum rhodium pavé editorial',
                'category': 'rings',
                'style': 'stacked contemporary mixed-metal textured',
                'material': 'mixed-metal diamond',
                'brand_style': 'editorial luxury stacked contemporary mix',
                'setting_type': 'pavé twisted mixed',
                'aesthetic': 'editorial contemporary luxury'
            }
        ]
        
        # Generate embeddings for sample data
        embeddings = []
        metadata = []
        
        for i, data in enumerate(sample_jewelry_data):
            # Generate embedding from description
            embedding = self.embedding_extractor.extract_text_embedding(data['description'])
            embeddings.append(embedding)
            
            # Add comprehensive metadata
            meta = {
                'id': f'sample_{i+1:03d}',
                'path': f'sample_jewelry_{i+1:03d}.jpg',
                'description': data['description'],
                'category': data['category'],
                'style': data['style'],
                'material': data['material'],
                'brand_style': data['brand_style'],
                'setting_type': data['setting_type'],
                'aesthetic': data['aesthetic'],
                'quality_score': 0.85 + (i % 4) * 0.03,  # Synthetic quality scores
                'modern_score': 0.88 + (i % 3) * 0.04,  # Modern aesthetic scores
                'source': 'synthetic_demo_data'
            }
            metadata.append(meta)
        
        # Add to vector store
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.vector_store.add_images(embeddings_array, metadata)
            print(f"✅ Sample database created with {len(metadata)} jewelry references")
        
    def index_jewelry_dataset(self, image_folder: str, metadata_file: Optional[str] = None):
        """Index a folder of jewelry images for production use"""
        print(f"📁 Indexing jewelry dataset from: {image_folder}")
        
        image_paths = []
        embeddings = []
        metadata = []
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        for ext in image_extensions:
            image_paths.extend(Path(image_folder).glob(f"**/{ext}"))
            image_paths.extend(Path(image_folder).glob(f"**/{ext.upper()}"))
            
        print(f"Found {len(image_paths)} images to index...")
        
        # Load existing metadata if available
        existing_metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
            print(f"Loaded existing metadata for {len(existing_metadata)} images")
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            try:
                # Extract embedding
                embedding = self.embedding_extractor.extract_image_embedding(str(img_path))
                embeddings.append(embedding)
                
                # Create metadata
                img_metadata = {
                    'id': f'img_{i+1:05d}',
                    'path': str(img_path),
                    'filename': img_path.name,
                    'category': self._classify_jewelry_type(img_path.name),
                    'size': os.path.getsize(img_path),
                    'source': 'user_dataset'
                }
                
                # Add existing metadata if available
                if img_path.name in existing_metadata:
                    img_metadata.update(existing_metadata[img_path.name])
                    
                metadata.append(img_metadata)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images...")
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        # Add to vector store
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.vector_store.add_images(embeddings_array, metadata)
            print(f"✅ Successfully indexed {len(embeddings)} jewelry images")
        else:
            print("❌ No images were successfully processed")
            
    def _classify_jewelry_type(self, filename: str) -> str:
        """Classify jewelry type based on filename"""
        filename_lower = filename.lower()
        
        for category, keywords in self.jewelry_categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
                
        return 'unknown'
    
    def retrieve_similar_images(self, query: Union[str, Image.Image], k: int = 3) -> List[Tuple[float, Dict]]:
        """Retrieve similar jewelry images"""
        if isinstance(query, str):
            # Text query
            query_embedding = self.embedding_extractor.extract_text_embedding(query)
        else:
            # Image query
            query_embedding = self.embedding_extractor.extract_image_embedding(query)
            
        results = self.vector_store.search(query_embedding, k)
        
        print(f"🔍 Retrieved {len(results)} similar images for query")
        for i, (score, metadata) in enumerate(results):
            print(f"  {i+1}. Score: {score:.3f} | {metadata.get('description', 'No description')[:60]}...")
            
        return results
    
    def retrieve_by_category(self, category: str, k: int = 5) -> List[Dict]:
        """Retrieve images by jewelry category"""
        category_images = []
        for metadata in self.vector_store.image_metadata:
            if metadata.get('category') == category:
                category_images.append(metadata)
                if len(category_images) >= k:
                    break
                    
        print(f"📂 Found {len(category_images)} images in category '{category}'")
        return category_images
    
    def generate_rag_enhanced_prompt(self, base_prompt: str, retrieved_images: List[Tuple[float, Dict]], confidence_threshold: float = 0.3) -> str:
        """Generate RAG-enhanced prompt based on retrieved similar images"""
        if not retrieved_images:
            print("⚠️ No retrieved images available, using base prompt")
            return base_prompt
            
        print(f"🔧 Enhancing prompt with {len(retrieved_images)} retrieved references...")
        
        # Extract style information from high-confidence retrievals
        style_elements = []
        material_elements = []
        brand_styles = []
        aesthetic_elements = []
        setting_types = []
        
        high_confidence_count = 0
        for score, metadata in retrieved_images:
            if score > confidence_threshold:
                high_confidence_count += 1
                
                if 'style' in metadata:
                    style_elements.extend(metadata['style'].split())
                if 'material' in metadata:
                    material_elements.append(metadata['material'])
                if 'brand_style' in metadata:
                    brand_styles.append(metadata['brand_style'])
                if 'aesthetic' in metadata:
                    aesthetic_elements.extend(metadata['aesthetic'].split())
                if 'setting_type' in metadata:
                    setting_types.append(metadata['setting_type'])
        
        print(f"  Using {high_confidence_count} high-confidence references (score > {confidence_threshold})")
        
        # Build enhancement components
        enhancements = []
        
        # Add unique style elements (top 2)
        if style_elements:
            unique_styles = list(set(style_elements))
            # Prioritize modern aesthetic terms
            modern_priority = ['contemporary', 'modern', 'minimalist', 'refined', 'clean']
            priority_styles = [s for s in unique_styles if s in modern_priority]
            other_styles = [s for s in unique_styles if s not in modern_priority]
            selected_styles = (priority_styles + other_styles)[:2]
            enhancements.extend(selected_styles)
            
        # Add aesthetic guidance
        if aesthetic_elements:
            unique_aesthetics = list(set(aesthetic_elements))
            top_aesthetic = unique_aesthetics[0] if unique_aesthetics else ""
            if top_aesthetic and top_aesthetic not in enhancements:
                enhancements.append(top_aesthetic)
        
        # Add brand aesthetic guidance (pick most relevant)
        if brand_styles:
            brand_guidance = brand_styles[0]
            # Extract key terms from brand style
            brand_terms = [term for term in brand_guidance.split() if term in ['clean', 'lines', 'luxury', 'sophisticated', 'refined', 'modern', 'contemporary']]
            if brand_terms:
                enhancements.extend(brand_terms[:2])
        
        # Add material consistency hint
        if material_elements:
            primary_material = material_elements[0]
            if 'gold' in primary_material:
                enhancements.append("high-quality metalwork")
        
        # Construct enhanced prompt
        enhanced_prompt = base_prompt
        
        if enhancements:
            # Remove duplicates while preserving order
            unique_enhancements = []
            for item in enhancements:
                if item not in unique_enhancements:
                    unique_enhancements.append(item)
            
            enhancement_text = ", ".join(unique_enhancements[:4])  # Limit to 4 enhancements
            enhanced_prompt += f", {enhancement_text}"
            
        # Add RAG-specific aesthetic guidance
        enhanced_prompt += ", professional jewelry photography, studio lighting, contemporary aesthetic"
        
        print(f"  Original: {base_prompt}")
        print(f"  Enhanced: {enhanced_prompt}")
        
        return enhanced_prompt
    
    def analyze_prompt_and_retrieve(self, prompt: str, k: int = 5) -> Dict:
        """Comprehensive prompt analysis with retrieval"""
        print(f"\n🔍 Analyzing prompt: '{prompt}'")
        
        # Text-based retrieval
        text_results = self.retrieve_similar_images(prompt, k=k)
        
        # Category-based retrieval
        jewelry_type = self._extract_jewelry_type(prompt)
        category_results = self.retrieve_by_category(jewelry_type, k=3) if jewelry_type != 'unknown' else []
        
        # Generate enhanced prompt
        enhanced_prompt = self.generate_rag_enhanced_prompt(text_results, text_results)
        
        analysis = {
            'original_prompt': prompt,
            'detected_jewelry_type': jewelry_type,
            'text_similarity_results': text_results,
            'category_results': category_results,
            'enhanced_prompt': enhanced_prompt,
            'retrieval_confidence': np.mean([score for score, _ in text_results]) if text_results else 0.0
        }
        
        print(f"📊 Analysis complete:")
        print(f"  Jewelry type: {jewelry_type}")
        print(f"  Retrieved {len(text_results)} similar references")
        print(f"  Average confidence: {analysis['retrieval_confidence']:.3f}")
        
        return analysis
    
    def _extract_jewelry_type(self, prompt: str) -> str:
        """Extract jewelry type from prompt"""
        prompt_lower = prompt.lower()
        
        for category, keywords in self.jewelry_categories.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return category
                
        return 'unknown'
    
    def save_database(self, filepath: str):
        """Save the RAG database"""
        self.vector_store.save(filepath)
        
    def load_database(self, filepath: str) -> bool:
        """Load the RAG database"""
        return self.vector_store.load(filepath)


class RAGEnhancedJewelryPipeline:
    """Wrapper to integrate Image RAG with any jewelry generation pipeline"""
    
    def __init__(self, base_pipeline, device: str = "cuda", rag_database_path: Optional[str] = None):
        self.base_pipeline = base_pipeline
        self.device = device
        
        # Initialize Image RAG
        print("🚀 Initializing RAG-Enhanced Jewelry Pipeline...")
        self.image_rag = JewelryImageRAG(device=device)
        
        # Load existing database if provided
        if rag_database_path and self.image_rag.load_database(rag_database_path):
            print(f"📂 Loaded existing RAG database from {rag_database_path}")
        
    def generate_with_rag(self, prompt: str, k_retrievals: int = 3, use_rag: bool = True, **generation_kwargs) -> Tuple[List[Image.Image], Dict]:
        """Generate jewelry images using RAG-enhanced prompting"""
        
        if use_rag:
            # Perform RAG analysis and retrieval
            rag_analysis = self.image_rag.analyze_prompt_and_retrieve(prompt, k=k_retrievals)
            enhanced_prompt = rag_analysis['enhanced_prompt']
            
            print(f"\n🎨 Generating with RAG enhancement...")
            print(f"   Original: {prompt}")
            print(f"   Enhanced: {enhanced_prompt}")
        else:
            enhanced_prompt = prompt
            rag_analysis = {'original_prompt': prompt, 'enhanced_prompt': prompt}
        
        # Generate with enhanced prompt
        if hasattr(self.base_pipeline, 'generate_improved'):
            generated_images = self.base_pipeline.generate_improved(
                prompt=enhanced_prompt,
                **generation_kwargs
            )
        else:
            # Fallback for basic pipelines
            generated_images = self.base_pipeline(
                prompt=enhanced_prompt,
                **generation_kwargs
            ).images
        
        return generated_images, rag_analysis
    
    def setup_custom_database(self, image_folder: str, save_path: str, metadata_file: Optional[str] = None):
        """Set up a custom jewelry database"""
        print(f"🔨 Setting up custom jewelry database...")
        self.image_rag.index_jewelry_dataset(image_folder, metadata_file)
        self.image_rag.save_database(save_path)
        print(f"💾 Custom database saved to {save_path}")


# Demonstration and testing functions
def demo_image_rag():
    """Demonstrate Image RAG functionality"""
    print("🎭 Image RAG Demo Starting...")
    print("=" * 60)
    
    # Initialize RAG system
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    rag_system = JewelryImageRAG(device=device)
    
    # Test prompts from the assignment
    test_prompts = [
        "channel-set diamond eternity band, 2 mm width, hammered 18k yellow gold",
        "14k rose-gold threader earrings, bezel-set round lab diamond ends",
        "modern signet ring, oval face, engraved gothic initial 'M', high-polish sterling silver",
        "delicate gold huggie hoops, contemporary styling",
        "organic cluster ring with mixed-cut sapphires and diamonds, brushed platinum finish"
    ]
    
    print(f"\n🧪 Testing {len(test_prompts)} prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*20} Test {i}/{len(test_prompts)} {'='*20}")
        
        # Analyze prompt with RAG
        analysis = rag_system.analyze_prompt_and_retrieve(prompt, k=3)
        
        print(f"\n📋 Analysis Results:")
        print(f"  Jewelry Type: {analysis['detected_jewelry_type']}")
        print(f"  Confidence: {analysis['retrieval_confidence']:.3f}")
        print(f"  Retrieved: {len(analysis['text_similarity_results'])} references")
        
        print(f"\n📝 Prompt Enhancement:")
        print(f"  Before: {analysis['original_prompt']}")
        print(f"  After:  {analysis['enhanced_prompt']}")
        
        # Show top retrieval
        if analysis['text_similarity_results']:
            top_match = analysis['text_similarity_results'][0]
            print(f"\n🎯 Top Match (score: {top_match[0]:.3f}):")
            print(f"     {top_match[1].get('description', 'No description')}")
            print(f"     Style: {top_match[1].get('style', 'N/A')}")
    
    print(f"\n✅ Demo completed!")


if __name__ == "__main__":
    demo_image_rag()
