import argparse
import os
# Importa o anotador que acabamos de corrigir
from utils.landmark_annotator import LandmarkAnnotator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset-fraction', type=float, default=1.0)
    args = parser.parse_args()

    print("=== PREPARAÇÃO DE DADOS (GPU) ===")
    
    annotator = LandmarkAnnotator(cache_dir="landmark_cache")
    dataset_name = os.path.basename(args.root.rstrip('/'))
    
    # Gera o JSON
    annotator.annotate_dataset(
        dataset_root=args.root,
        dataset_name=dataset_name,
        limit_fraction=args.dataset_fraction
    )

if __name__ == '__main__':
    main()