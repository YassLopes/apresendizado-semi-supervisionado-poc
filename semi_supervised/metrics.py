import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def calculate_metrics(y_true, y_pred):
    """
    Calcula várias métricas de avaliação para o modelo
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Calculando a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics

def print_metrics(metrics):
    """
    Imprime as métricas de forma formatada
    """
    print("\nMétricas de Avaliação:")
    print(f"Acurácia: {metrics['accuracy']:.4f}")
    print(f"Precisão: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Imprimindo a matriz de confusão
    print("\nMatriz de Confusão:")
    print(metrics['confusion_matrix']) 