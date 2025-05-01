# Aprendizado Semi-Supervisionado: Uma Prova de Conceito

## O que é Aprendizado Semi-Supervisionado?

O aprendizado semi-supervisionado é uma abordagem de machine learning que combina elementos do aprendizado supervisionado e não supervisionado. Enquanto no aprendizado supervisionado tradicional precisamos de um grande conjunto de dados rotulados, e no não supervisionado trabalhamos apenas com dados sem rótulos, o semi-supervisionado utiliza uma pequena quantidade de dados rotulados junto com uma grande quantidade de dados não rotulados.

Esta abordagem é particularmente útil em cenários onde:
- O custo de rotular dados é alto
- Temos acesso a muitos dados, mas poucos deles estão rotulados
- Queremos aproveitar a estrutura dos dados não rotulados para melhorar o desempenho do modelo

## Sobre o Projeto

Este projeto implementa uma prova de conceito de aprendizado semi-supervisionado usando o dataset MNIST (dígitos manuscritos). A implementação utiliza a estratégia de self-training, onde:

1. Começamos com um pequeno conjunto de dados rotulados (100 exemplos)
2. Treinamos um modelo inicial com esses dados
3. Usamos o modelo para predizer rótulos nos dados não rotulados
4. Selecionamos as predições mais confiantes para adicionar ao conjunto rotulado
5. Repetimos o processo iterativamente

O projeto foi estruturado para demonstrar claramente como o modelo melhora seu desempenho à medida que mais exemplos são rotulados automaticamente. Além disso, implementamos várias métricas de avaliação e visualizações para acompanhar o progresso do modelo.

## Estrutura do Projeto

```
.
├── semi_supervised/
│   ├── __init__.py
│   ├── data.py          # Manipulação de dados
│   ├── model.py         # Implementação do self-training
│   ├── metrics.py       # Cálculo de métricas
│   └── visualization.py # Funções de visualização
├── main.py              # Script principal
├── requirements.txt     # Dependências
└── README.md           # Este arquivo
```

## Requisitos

- Python 3.7+
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd [NOME_DO_DIRETÓRIO]
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Executando o Projeto

Para executar a prova de conceito, basta rodar:

```bash
python main.py
```

O script irá:
1. Carregar o dataset MNIST
2. Executar o algoritmo de self-training
3. Gerar visualizações dos resultados
4. Printar métricas de avaliação

## Resultados

O script gera três visualizações:

1. `metrics_evolution.png`: Mostra a evolução das métricas (acurácia, precisão, recall e F1-score) ao longo das iterações
2. `confusion_matrix.png`: Exibe a matriz de confusão do modelo final
3. `label_distribution.png`: Mostra a distribuição dos labels nos conjuntos rotulado e não rotulado

## Personalização

Você pode ajustar os seguintes parâmetros no arquivo `main.py`:

- `n_labeled`: Número de exemplos rotulados iniciais
- `confidence_threshold`: Percentil para seleção de exemplos confiantes
- `n_iterations`: Número de iterações do self-training

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.