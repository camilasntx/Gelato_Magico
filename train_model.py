import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

# --- 1. Configuração do Ambiente e Pipeline ---
# Define o nome do experimento no MLflow
EXPERIMENT_NAME = "Previsao_Vendas_Gelato_Magico"
mlflow.set_experiment(EXPERIMENT_NAME)

# Path do arquivo de dados
DATA_PATH = 'inputs/temperatura_vendas.csv'

# Define as variáveis globais para a aplicação (simulação do ID do App)
appId = os.environ.get('APP_ID', 'gelato-magico-v1')


def treinar_modelo():
    """
    Carrega os dados, treina o modelo de Regressão Linear, avalia
    e registra o modelo e métricas no MLflow.
    """
    print(f"Iniciando o experimento MLflow: {EXPERIMENT_NAME}")

    # Inicia um novo MLflow Run
    with mlflow.start_run(run_name="Regressao_Linear_Simples") as run:
        try:
            # --- 2. Carga e Preparação dos Dados ---
            df = pd.read_csv(DATA_PATH)
            
            # Variáveis independentes (Temperatura) e dependente (Vendas)
            X = df[['Temperatura']].values
            y = df['Vendas'].values
            
            # Divisão dos dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # --- 3. Treinamento do Modelo ---
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)

            # --- 4. Previsão e Avaliação ---
            y_pred = modelo.predict(X_test)
            
            # Métricas de avaliação
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # --- 5. Rastreamento e Registro com MLflow ---
            
            # Log de Parâmetros
            mlflow.log_param("modelo_tipo", "Regressao_Linear")
            mlflow.log_param("intercepto", modelo.intercept_)
            mlflow.log_param("coeficiente", modelo.coef_[0])
            mlflow.log_param("feature", "Temperatura")
            
            # Log de Métricas
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
            
            print(f"MSE: {mse:.2f}")
            print(f"R-squared: {r2:.2f}")

            # Log do Modelo (salva o modelo na pasta artifacts do MLflow)
            mlflow.sklearn.log_model(
                sk_model=modelo,
                artifact_path="modelos",
                registered_model_name=f"{appId}-VendasSorveteModel"
            )
            
            run_id = run.info.run_id
            print(f"\n--- Sucesso! ---")
            print(f"Modelo registrado com sucesso!")
            print(f"ID da Execução (Run ID): {run_id}")
            print(f"Nome do Modelo Registrado: {appId}-VendasSorveteModel")
            
            # Exemplo de Previsão em Tempo Real (Simulação de uso do modelo)
            temp_hoje = 26.0
            previsao = modelo.predict(np.array([[temp_hoje]]))
            print(f"\nPrevisão para uma temperatura de {temp_hoje}°C: {int(previsao[0]):,} sorvetes.")

        except FileNotFoundError:
            print(f"ERRO: Arquivo de dados não encontrado em {DATA_PATH}.")
            print("Certifique-se de ter criado a pasta 'inputs' e o arquivo 'temperatura_vendas.csv'.")
        except Exception as e:
            print(f"Ocorreu um erro durante o treinamento: {e}")

if __name__ == "__main__":
    # Comando para iniciar o servidor de rastreamento local (opcional, se não estiver usando um servidor remoto)
    # Recomenda-se rodar 'mlflow ui' em um terminal separado para visualizar os resultados.
    
    # Assegura que o diretório inputs existe
    if not os.path.exists('inputs'):
        os.makedirs('inputs')

    # Executa o pipeline de treinamento
    treinar_modelo()

    print("\nPara visualizar o rastreamento, abra um novo terminal e execute: 'mlflow ui'")
