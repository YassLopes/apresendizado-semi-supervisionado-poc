import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .metrics import calculate_metrics

class SelfTrainingModel:
    def __init__(self, base_model=None, confidence_threshold=75, n_iterations=5):
        """
        Inicializa o modelo de Self-Training
        """
        self.base_model = base_model or RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.confidence_threshold = confidence_threshold
        self.n_iterations = n_iterations
        self.metrics_history = []
    
    def train_and_predict(self, X_labeled, y_labeled, X_unlabeled, X_test, y_test):
        """
        Treina o modelo e faz predições
        """
        # Treinando o modelo
        self.base_model.fit(X_labeled, y_labeled)
        
        # Fazendo predições
        y_pred_unlabeled = self.base_model.predict(X_unlabeled)
        y_pred_test = self.base_model.predict(X_test)
        
        # Calculando métricas
        metrics = calculate_metrics(y_test, y_pred_test)
        
        return self.base_model, y_pred_unlabeled, metrics
    
    def select_confident_examples(self, X_unlabeled, y_pred_unlabeled):
        """
        Seleciona exemplos confiantes para adicionar ao conjunto rotulado
        """
        # Obtendo probabilidades das predições
        probas = self.base_model.predict_proba(X_unlabeled)
        confidences = np.max(probas, axis=1)
        
        # Selecionando exemplos confiantes
        threshold = np.percentile(confidences, self.confidence_threshold)
        confident_mask = confidences >= threshold
        
        # Separando exemplos confiantes
        confident_X = X_unlabeled[confident_mask]
        confident_y = y_pred_unlabeled[confident_mask]
        remaining_X = X_unlabeled[~confident_mask]
        
        return confident_X, confident_y, remaining_X
    
    def fit(self, X_labeled, y_labeled, X_unlabeled, X_test, y_test):
        """
        Executa o processo de self-training
        """
        current_X_labeled = X_labeled.copy()
        current_y_labeled = y_labeled.copy()
        current_X_unlabeled = X_unlabeled.copy()
        
        for i in range(self.n_iterations):
            print(f"\nIteração {i+1}/{self.n_iterations}")
            print(f"Número de exemplos rotulados: {len(current_X_labeled)}")
            
            # Treinando e obtendo métricas
            _, _, metrics = self.train_and_predict(
                current_X_labeled, current_y_labeled,
                current_X_unlabeled, X_test, y_test
            )
            
            self.metrics_history.append(metrics)
            
            # Selecionando exemplos confiantes
            if len(current_X_unlabeled) > 0:
                confident_X, confident_y, remaining_X = self.select_confident_examples(
                    current_X_unlabeled, self.base_model.predict(current_X_unlabeled)
                )
                
                # Atualizando conjuntos
                current_X_labeled = np.vstack([current_X_labeled, confident_X])
                current_y_labeled = np.concatenate([current_y_labeled, confident_y])
                current_X_unlabeled = remaining_X
        
        return self.metrics_history 