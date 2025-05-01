import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics_history(metrics_history, save_path=None):
    """
    Plota a evolução das métricas ao longo das iterações
    """
    # Extraindo métricas
    iterations = range(1, len(metrics_history) + 1)
    accuracies = [m['accuracy'] for m in metrics_history]
    precisions = [m['precision'] for m in metrics_history]
    recalls = [m['recall'] for m in metrics_history]
    f1_scores = [m['f1'] for m in metrics_history]
    
    # Criando o gráfico
    plt.figure(figsize=(12, 6))
    
    plt.plot(iterations, accuracies, marker='o', label='Acurácia')
    plt.plot(iterations, precisions, marker='s', label='Precisão')
    plt.plot(iterations, recalls, marker='^', label='Recall')
    plt.plot(iterations, f1_scores, marker='*', label='F1-Score')
    
    plt.xlabel('Iteração')
    plt.ylabel('Valor da Métrica')
    plt.title('Evolução das Métricas durante o Self-Training')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(confusion_matrix, save_path=None):
    """
    Plota a matriz de confusão do modelo
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_label_distribution(y_labeled, y_unlabeled, save_path=None):
    """
    Plota a distribuição de labels nos conjuntos rotulado e não rotulado
    """
    plt.figure(figsize=(12, 6))
    
    # Contando ocorrências de cada label
    labeled_counts = np.bincount(y_labeled.astype(int))
    unlabeled_counts = np.bincount(y_unlabeled.astype(int))
    
    x = np.arange(len(labeled_counts))
    width = 0.35
    
    plt.bar(x - width/2, labeled_counts, width, label='Rotulado')
    plt.bar(x + width/2, unlabeled_counts, width, label='Não Rotulado')
    
    plt.xlabel('Dígito')
    plt.ylabel('Contagem')
    plt.title('Distribuição de Labels')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 