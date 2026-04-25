#!/usr/bin/env python3
"""
Download HotpotQA dataset from Hugging Face using the datasets library.
This method works out-of-the-box on Ubuntu with Python 3.8+.
"""

import os
import sys
from datasets import load_dataset

def main():
    print("=" * 60)
    print("HotpotQA Dataset Downloader")
    print("=" * 60)
    
    # Configuration
    dataset_name = "hotpotqa/hotpot_qa"
    # Choose one of the two configurations:
    # - "distractor"  (simpler, smaller ~1.2GB)
    # - "fullwiki"    (full Wikipedia context, ~1.3GB)
    config_name = "fullwiki"   # or "distractor"
    
    # Optional: set cache directory (default is ~/.cache/huggingface/datasets)
    # os.environ["HF_DATASETS_CACHE"] = "./hotpotqa_cache"
    
    print(f"\n📥 Loading dataset: {dataset_name}")
    print(f"📁 Configuration: {config_name}")
    print("⏳ This may take several minutes (downloading ~1.3 GB)...\n")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)
        
        print("\n✅ Download complete!\n")
        print("📊 Dataset structure:")
        print(f"   {dataset}")
        
        # Show basic statistics
        print("\n📈 Split sizes:")
        for split_name, split_data in dataset.items():
            print(f"   {split_name}: {len(split_data)} examples")
        
        # Display first example from train split
        if "train" in dataset:
            print("\n🔍 First training example:")
            example = dataset["train"][0]
            print(f"   Question: {example['question'][:150]}...")
            print(f"   Answer: {example['answer']}")
            print(f"   Type: {example['type']}")
            print(f"   Level: {example['level']}")
            print(f"   Supporting facts: {len(example['supporting_facts']['title'])} facts")
        
        print("\n💾 Dataset is now cached locally.")
        print(f"   Cache location: {os.environ.get('HF_DATASETS_CACHE', '~/.cache/huggingface/datasets')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:")
        print("1. Ensure you have enough disk space (at least 3 GB free)")
        print("2. Try: pip install --upgrade datasets huggingface_hub")
        print("3. If behind a proxy, set HTTP_PROXY/HTTPS_PROXY environment variables")
        sys.exit(1)

if __name__ == "__main__":
    main()