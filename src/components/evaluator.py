# Local: src/components/evaluator.py
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from src.utils.metrics import (
    compute_cosine_similarity, 
    find_best_threshold, 
    calculate_metrics,
    plot_roc_curve,
    plot_confusion_matrix
)
from sklearn.metrics import roc_curve, auc

class LFWEvaluator:
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        """
        Isola a parte do modelo que gera embeddings, ignorando a cabeça de classificação.
        Tenta localizar a camada 'cosface_loss' ou usa a penúltima camada.
        """
        try:
            # Tenta pegar a entrada da camada CosFace (o embedding puro)
            cosface_layer = self.model.get_layer("cosface_loss")
            # input[0] é o embedding, input[1] é o label
            return Model(inputs=self.model.input[0], outputs=cosface_layer.input[0])
        except ValueError:
            self.logger.warning("Camada 'cosface_loss' não encontrada. Usando output da camada anterior à última.")
            # Fallback genérico: assume que as últimas camadas são Head/Softmax
            return Model(inputs=self.model.input[0], outputs=self.model.layers[-3].output)

    def preprocess_image(self, image_path, image_size):
        """Carrega e aplica pré-processamento do ResNet50."""
        if not os.path.exists(image_path):
            return None
        
        img_raw = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        
        # Importante: Mesmo pré-processamento do treino (caffe/resnet)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img

    def extract_features_tta(self, image_path, image_size):
        """
        Extrai features usando Test Time Augmentation (Original + Flip).
        """
        img = self.preprocess_image(image_path, image_size)
        if img is None:
            return None

        # TTA: Batch com imagem original e imagem espelhada
        img_flipped = tf.image.flip_left_right(img)
        batch = tf.stack([img, img_flipped])
        
        # Inferência
        features = self.feature_extractor.predict(batch, verbose=0)
        
        # Concatena os vetores (estratégia robusta de TTA)
        return np.concatenate([features[0], features[1]], axis=0)

    def parse_pairs(self, pairs_path, lfw_dir):
        """Lê o arquivo pairs.txt do LFW."""
        pairs = []
        with open(pairs_path, 'r') as f:
            lines = f.readlines()
            if len(lines[0].strip().split()) == 1:
                lines = lines[1:] # Pula cabeçalho

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3: # Mesma pessoa
                    path1 = os.path.join(lfw_dir, parts[0], f"{parts[0]}_{int(parts[1]):04d}.jpg")
                    path2 = os.path.join(lfw_dir, parts[0], f"{parts[0]}_{int(parts[2]):04d}.jpg")
                    label = 1
                elif len(parts) == 4: # Pessoas diferentes
                    path1 = os.path.join(lfw_dir, parts[0], f"{parts[0]}_{int(parts[1]):04d}.jpg")
                    path2 = os.path.join(lfw_dir, parts[2], f"{parts[2]}_{int(parts[3]):04d}.jpg")
                    label = 0
                else:
                    continue
                pairs.append((path1, path2, label))
        return pairs

    def evaluate(self, pairs_path, lfw_dir, output_dir, image_size=112):
        """Executa o pipeline completo de avaliação."""
        self.logger.info(f"Iniciando avaliação no diretório: {lfw_dir}")
        pairs = self.parse_pairs(pairs_path, lfw_dir)
        self.logger.info(f"Total de pares para processar: {len(pairs)}")

        y_true = []
        y_scores = []
        
        for i, (p1, p2, label) in enumerate(pairs):
            f1 = self.extract_features_tta(p1, image_size)
            f2 = self.extract_features_tta(p2, image_size)

            if f1 is None or f2 is None:
                continue

            similarity = compute_cosine_similarity(f1, f2)
            y_true.append(label)
            y_scores.append(similarity)

            if i % 100 == 0:
                print(f"Processando... {i}/{len(pairs)}", end='\r')

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # 1. Calcular Curva ROC e AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 2. Encontrar melhor Threshold
        best_thresh, best_acc = find_best_threshold(y_true, y_scores, thresholds)
        
        # 3. Calcular Métricas Detalhadas
        y_pred = (y_scores > best_thresh).astype(int)
        metrics = calculate_metrics(y_true, y_pred)
        
        # 4. Salvar Gráficos
        os.makedirs(output_dir, exist_ok=True)
        plot_roc_curve(fpr, tpr, roc_auc, os.path.join(output_dir, "roc_curve.png"))
        plot_confusion_matrix(metrics['confusion_matrix'], best_thresh, os.path.join(output_dir, "confusion_matrix.png"))

        # Logar Resultados
        self.logger.info("-" * 40)
        self.logger.info(f"RESULTADOS DA AVALIAÇÃO (TTA ATIVADO)")
        self.logger.info("-" * 40)
        self.logger.info(f"AUC (Area Under Curve): {roc_auc:.4f}")
        self.logger.info(f"Melhor Threshold:       {best_thresh:.4f}")
        self.logger.info(f"Acurácia Máxima:        {best_acc:.4f}")
        self.logger.info(f"Precision:              {metrics['precision']:.4f}")
        self.logger.info(f"Recall:                 {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score:               {metrics['f1']:.4f}")
        self.logger.info("-" * 40)

        return {
            'auc': roc_auc,
            'accuracy': best_acc,
            'threshold': best_thresh,
            **metrics
        }
