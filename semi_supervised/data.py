import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(normalize=True):
    """
    Carrega o dataset MNIST e realiza pré-processamento básico
    """
    print("Carregando o dataset MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if normalize:
        X = X / 255.0
    
    # Dividindo o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def prepare_semi_supervised_data(X_train, y_train, n_labeled=100):
    """
    Prepara os dados para aprendizado semi-supervisionado
    """
    X_labeled = X_train[:n_labeled]
    y_labeled = y_train[:n_labeled]
    X_unlabeled = X_train[n_labeled:]
    
    return X_labeled, y_labeled, X_unlabeled 