# Local: src/utils/metrics.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score

def compute_cosine_similarity(feat1, feat2):
    """Calcula a similaridade de cosseno entre dois vetores."""
    # Adiciona epsilon para evitar divisão por zero
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    return np.dot(feat1, feat2) / (norm1 * norm2 + 1e-5)

def find_best_threshold(y_true, y_scores, thresholds):
    """Encontra o limiar (threshold) que maximiza a acurácia."""
    best_acc = 0
    best_thresh = 0
    
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        acc = np.mean(y_pred == y_true)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de classificação padrão."""
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Gera e salva o gráfico da curva ROC."""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, threshold, save_path):
    """Gera e salva o gráfico da matriz de confusão."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Diferente', 'Mesma'], 
                yticklabels=['Diferente', 'Mesma'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão (Threshold={threshold:.4f})')
    plt.savefig(save_path)
    plt.close()