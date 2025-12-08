import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import random
import logging

# Configuração de Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
LOGGER = logging.getLogger("LandmarkAnnotator")

try:
    # Nova importação baseada na estrutura do repositório analisado
    from uniface import RetinaFace
    UNIFACE_AVAILABLE = True
except ImportError:
    UNIFACE_AVAILABLE = False
    LOGGER.warning("uniface não instalado ou versão incompatível.")

class LandmarkAnnotator:
    def __init__(
        self,
        cache_dir: str = "landmark_cache",
        conf_threshold: float = 0.5,
        image_size: int = 112
    ):
        if not UNIFACE_AVAILABLE:
            raise ImportError("Instale o uniface atualizado: pip install git+https://github.com/yakhyo/uniface.git")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.detector = None
        self.stats = {'total_images': 0, 'annotated': 0, 'failed': 0}

    def _init_detector(self):
        if self.detector is None:
            LOGGER.info("Inicializando Uniface (RetinaFace) v1.1+...")
            try:
                # O Uniface detecta providers (CUDA/CPU) automaticamente no __init__
                self.detector = RetinaFace(conf_thresh=self.conf_threshold)
                LOGGER.info("Uniface inicializado.")
            except Exception as e:
                LOGGER.error(f"Erro ao iniciar Uniface: {e}")
                raise e

    def _get_cache_path(self, dataset_name: str) -> Path:
        return self.cache_dir / f"{dataset_name}_landmarks.json"

    def _extract_landmarks(self, image_path: str) -> Optional[List[List[float]]]:
        try:
            # Uniface/OpenCV espera BGR
            img = cv2.imread(image_path)
            if img is None: return None

            # Detecta faces
            # Retorno na v1.1.2 é uma lista de dicts: [{'bbox':..., 'confidence':..., 'landmarks':...}]
            faces = self.detector.detect(img)
            
            if not faces:
                return None

            # Encontrar a face com maior confiança
            best_face = None
            best_conf = 0.0

            for face in faces:
                # A chave agora é 'confidence' ou 'score' dependendo da versão exata, o código usa 'confidence'
                conf = face.get('confidence', 0.0)
                if conf >= self.conf_threshold and conf > best_conf:
                    best_conf = conf
                    best_face = face
            
            if best_face is None:
                return None
            
            # Extrai landmarks (chave 'landmarks')
            # Formato esperado: array (5, 2)
            kps = best_face.get('landmarks')
            
            if kps is None:
                return None

            # Normaliza para [0, 1] para a rede neural
            h, w = img.shape[:2]
            normalized = [[float(p[0]/w), float(p[1]/h)] for p in kps]
            
            return normalized

        except Exception as e:
            # Em caso de erro de leitura ou processamento
            return None

    def annotate_dataset(
        self,
        dataset_root: str,
        dataset_name: str,
        limit_fraction: float = 1.0,
        force_reannotate: bool = False
    ) -> Dict[str, List[List[float]]]:
        
        # Cache nomeado com a fração
        cache_name = f"{dataset_name}_frac{limit_fraction}" if limit_fraction < 1.0 else dataset_name
        cache_path = self._get_cache_path(cache_name)
        
        if cache_path.exists() and not force_reannotate:
            LOGGER.info(f"Cache encontrado: {cache_path}")
            with open(cache_path, 'r') as f:
                data = json.load(f)
                landmarks = data.get('landmarks', {})
                if len(landmarks) > 0:
                    LOGGER.info(f"Carregados {len(landmarks)} landmarks do cache.")
                    return landmarks

        self._init_detector()
        
        LOGGER.info(f"Mapeando imagens em: {dataset_root}")
        image_paths = []
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for root, _, files in os.walk(dataset_root):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_ext:
                    image_paths.append(os.path.join(root, file))
        
        total_found = len(image_paths)
        LOGGER.info(f"Total encontrado: {total_found}")

        # --- APLICA O FILTRO DE PORCENTAGEM (Sanity Check) ---
        if limit_fraction < 1.0:
            target_size = int(total_found * limit_fraction)
            LOGGER.info(f"Limitando dataset a {target_size} imagens ({limit_fraction*100}%)...")
            random.shuffle(image_paths)
            image_paths = image_paths[:target_size]
        # -----------------------------------------------------

        self.stats['total_images'] = len(image_paths)
        landmarks_dict = {}
        
        LOGGER.info("Iniciando extração com Uniface...")
        for img_path in tqdm(image_paths, mininterval=1.0):
            lmks = self._extract_landmarks(img_path)
            
            if lmks is not None:
                rel_path = os.path.relpath(img_path, dataset_root)
                landmarks_dict[rel_path] = lmks
                self.stats['annotated'] += 1
            else:
                self.stats['failed'] += 1
        
        # Salva o cache
        with open(cache_path, 'w') as f:
            json.dump({
                'stats': self.stats,
                'landmarks': landmarks_dict,
                'config': {'fraction': limit_fraction, 'backend': 'uniface_v1'}
            }, f)
            
        LOGGER.info(f"Anotação finalizada. Sucesso: {self.stats['annotated']}/{self.stats['total_images']}")
        
        return landmarks_dict