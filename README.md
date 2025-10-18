# 🍦 Previsão de Vendas de Sorvete (Gelato Mágico) com Machine Learning e MLOps

## 🎯 Objetivo do Projeto

Desenvolver um modelo preditivo de Regressão Linear para prever a demanda diária de sorvetes com base na temperatura. O projeto utiliza o MLflow para rastrear experimentos, garantir reprodutibilidade e gerenciar o ciclo de vida do modelo (MLOps).

## 🧠 Racional Analítico

As vendas de sorvete (variável target) possuem uma clara **relação linear e positiva** com a temperatura (variável feature). Utilizamos a **Regressão Linear Simples** como ponto de partida, pois é o modelo mais interpretável para este cenário.

* **Feature:** Temperatura (°C)
* **Target:** Vendas (Unidades)
* **Métrica:** Utilizamos o **Erro Quadrático Médio (MSE)** para quantificar o erro médio das previsões e o **R2 Score** para medir o quão bem a regressão se ajusta aos dados.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Machine Learning:** `scikit-learn` (Regressão Linear)
* **Gerenciamento de Experimentos (MLOps):** **MLflow**
* **Visualização:** `matplotlib`

## ⚙️ Pipeline e Processo de MLOps

O fluxo de trabalho foi estruturado em um pipeline simples e rastreável, conforme a metodologia MLOps:

1.  **Carga e Preparação de Dados:** Dados de temperatura/vendas carregados de `inputs/vendas_temperatura.csv`.
2.  **Treinamento do Modelo:** O script `src/train_model.py` treina o modelo de Regressão Linear.
3.  **Rastreamento com MLflow:**
    * **Parâmetros Registrados:** Coeficiente Angular, Intercepto e `fit_intercept` (usando `mlflow.log_param()`).
    * **Métricas Registradas:** MSE e R2 Score (usando `mlflow.log_metric()`).
    * **Artefatos Registrados:** O Modelo treinado (`mlflow.sklearn.log_model()`) e um gráfico de dispersão com a linha de regressão (`mlflow.log_artifact()`).
4.  **Registro do Modelo:** O modelo é registrado no **MLflow Model Registry** sob o nome `VendasSorveteModel`.
5.  **Simulação de Previsão:** O script `src/predict_model.py` simula uma API de produção, carregando a versão mais recente do modelo registrado para previsões em tempo real.

### 🖼️ Screenshots do MLflow UI

**(Aqui você deve incluir prints que você mesmo gerou ao rodar o projeto)**

1.  **Print da Lista de Experimentos no MLflow:** (Mostre a página principal com o nome do experimento `Previsao_Vendas_Sorvete`).
2.  **Print dos Parâmetros e Métricas Registrados:** (Mostre a página de um Run, destacando o R2, MSE e os coeficientes).
3.  **Print do Gráfico de Artefatos:** (Mostre o gráfico `vendas_regressao_plot.png` que prova a linearidade).

## 💡 Insights e Possibilidades de Melhoria

### Insights Chave do Modelo

* **Impacto da Temperatura:** O Coeficiente Angular registrado pelo MLflow indica o aumento de vendas para cada grau Celsius adicional (ex: um coeficiente de 25 significa que, a cada 1°C a mais, as vendas aumentam em 25 unidades).
* **Planejamento de Produção:** Com o modelo, a Gelato Mágico pode agora usar a previsão do tempo para planejar a produção do dia seguinte, minimizando o desperdício (previsão acima da real) e a perda de vendas (previsão abaixo da real).

### Possibilidades de Melhoria (Próximos Passos)

1.  **Adicionar Features:** O modelo pode ser aprimorado incluindo outras variáveis correlacionadas, como: **Umidade**, **Dia da Semana** (fim de semana tem mais vendas?), e **Preço**.
2.  **Hyperparameter Tuning:** Para um modelo mais complexo (ex: Ridge, Lasso, ou SVR), poderíamos usar um otimizador de hiperparâmetros (como Optuna) e usar as políticas de encerramento antecipado (Bandit, Median) para otimizar a busca.
3.  **CI/CD Pipeline:** Automatizar o treinamento. Sempre que houver novos dados, um **pipeline de Azure DevOps/GitHub Actions** seria acionado para: treinar, testar (verificar se o R2 é aceitável) e, se aprovado, promover o novo modelo para o "Staging" no MLflow Model Registry.
4.  **Implantação Real:** Em vez de simulação, o modelo poderia ser empacotado e implantado como um **endpoint HTTP** em serviços de cloud (Azure ML Endpoint, AWS SageMaker, Google Vertex AI) para ser consumido por um aplicativo de inventário ou produção.
