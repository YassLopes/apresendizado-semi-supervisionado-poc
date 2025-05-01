from semi_supervised.data import load_mnist, prepare_semi_supervised_data
from semi_supervised.model import SelfTrainingModel
from semi_supervised.visualization import (
    plot_metrics_history,
    plot_confusion_matrix,
    plot_label_distribution
)
from semi_supervised.metrics import print_metrics

def main():
    # Carregando e preparando os dados
    X_train, X_test, y_train, y_test = load_mnist()
    X_labeled, y_labeled, X_unlabeled = prepare_semi_supervised_data(
        X_train, y_train, n_labeled=100
    )
    
    # Criando e treinando o modelo
    model = SelfTrainingModel(
        confidence_threshold=75,
        n_iterations=5
    )
    
    print("\nIniciando o processo de self-training...")
    metrics_history = model.fit(
        X_labeled, y_labeled,
        X_unlabeled, X_test, y_test
    )
    
    # Plotando resultados
    print("\nGerando visualizações...")
    plot_metrics_history(metrics_history, 'metrics_evolution.png')
    plot_confusion_matrix(
        metrics_history[-1]['confusion_matrix'],
        'confusion_matrix.png'
    )
    plot_label_distribution(y_labeled, y_train[len(y_labeled):], 'label_distribution.png')
    
    # Imprimindo métricas finais
    print("\nMétricas Finais:")
    print_metrics(metrics_history[-1])
    
    print("\nProcesso concluído! Os resultados foram salvos em:")
    print("- metrics_evolution.png")
    print("- confusion_matrix.png")
    print("- label_distribution.png")

if __name__ == "__main__":
    main() 