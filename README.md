# **🍦 Projeto Gelato Mágico: Previsão de Vendas de Sorvete com MLOps**

## **🎯 Objetivo**

Desenvolver um **modelo de regressão preditiva** para otimizar a produção da sorveteria "Gelato Mágico" prevendo a demanda de sorvetes com base na temperatura. Este projeto demonstra os passos essenciais do ciclo de vida do Machine Learning, utilizando o **MLflow** para rastreabilidade e gestão de modelos.

## **🧠 Racional do Modelo (Regressão Linear)**

A correlação entre Temperatura e Vendas sugere uma **Regressão Linear Simples**. Este modelo é ideal, pois:

1. **Interpretabilidade:** O coeficiente da regressão indica o aumento exato de vendas para cada aumento de um grau na temperatura, tornando a decisão de negócio (quanto produzir) clara e direta.  
2. **Eficiência:** É rápido de treinar e fácil de implementar em ambientes de produção.

O pipeline de ML se concentra em:

1. **Entrada:** Temperatura (°C).  
2. **Saída:** Vendas (Unidades).  
3. **Métricas:** **R2 Score** (Qualidade do ajuste) e **MSE** (Magnitude do erro).

## **🛠️ Tecnologias e Ferramentas**

| Categoria | Ferramenta | Uso no Projeto |
| :---- | :---- | :---- |
| **Linguagem** | Python | Linguagem principal para todo o pipeline. |
| **Modelo ML** | Scikit-learn | Implementação do modelo de Regressão Linear. |
| **MLOps/Tracking** | **MLflow** | Essencial para registrar execuções, salvar métricas, parâmetros e gerenciar o ciclo de vida do modelo. |
| **Visualização** | Matplotlib | Geração de gráficos para análise e log como artefatos. |
| **Dados** | Pandas, Numpy | Manipulação de dados e cálculos. |

## **⚙️ Pipeline e Processo de MLOps**

O fluxo de trabalho é definido nos scripts Python e rastreado pelo MLflow:

1. **Carga de Dados:** O script src/train\_model.py lê os dados do arquivo inputs/vendas\_temperatura.txt, processando o formato de texto fornecido.  
2. **Treinamento e Avaliação:** O modelo é treinado e avaliado (R2 e MSE).  
3. **Rastreamento (MLflow Tracking):**  
   * **Parâmetros Registrados:** Algoritmo, Coeficiente Angular e Intercepto (registrados via mlflow.log\_param()).  
   * **Métricas Registradas:** R2 Score e MSE (registradas via mlflow.log\_metric()).  
   * **Artefatos Registrados:** O arquivo de dados brutos, o gráfico de dispersão com a linha de regressão, e o modelo serializado.  
4. **Gerenciamento de Modelos (MLflow Model Registry):** O modelo é promovido para o registro centralizado sob o nome **VendasSorveteModel**, o que facilita a recuperação da "última versão" para o deploy.  
5. **Simulação de Deploy:** O script src/predict\_model.py simula uma chamada a um endpoint, carregando a versão mais recente do modelo do MLflow Registry para previsões em tempo real.

## **🚀 Como Executar o Projeto**

Siga estes passos para replicar o ambiente e executar o pipeline completo:

### **1\. Preparação do Ambiente**

\# 1\. Clone o repositório  
git clone \<URL\_DO\_SEU\_REPOSITORIO\>  
cd \<NOME\_DO\_SEU\_REPOSITORIO\>

\# 2\. Crie um ambiente virtual (Recomendado)  
python \-m venv venv  
source venv/bin/activate  \# Linux/macOS  
venv\\Scripts\\activate     \# Windows

\# 3\. Instale as dependências  
pip install \-r requirements.txt

### **2\. Execução do Pipeline de Treinamento**

Execute o script que treina o modelo e registra todos os metadados no MLflow.

python src/train\_model.py

### **3\. Visualização e Gerenciamento (MLflow UI)**

Inicie a interface de usuário local do MLflow para inspecionar os resultados:

mlflow ui

Abra o endereço http://localhost:5000 no seu navegador.

### **4\. Simulação de Previsão em Tempo Real**

Execute o script que simula o serviço de deploy, carregando o modelo *diretamente do registro do MLflow*.

python src/predict\_model.py

## **📸 Screenshots e Evidências**

**(Inclua seus próprios prints aqui após rodar os passos 2 e 3\)**

### **1\. Interface de Experimentos (MLflow UI)**

*Descrição: Este print demonstra que o experimento foi rastreado com sucesso, registrando uma nova execução (run) associada ao treinamento do modelo.*

### **2\. Métricas, Parâmetros e Artefatos**

*Descrição: Evidência do sucesso da regressão com um R2 Score alto, comprovando a forte correlação. Os parâmetros de entrada e métricas de desempenho estão devidamente registrados via mlflow.log\_param() e mlflow.log\_metric().*

### **3\. Visualização do Modelo (Artifact)**

*Descrição: O gráfico de regressão, salvo como um artefato (vendas\_regressao\_plot.png), confirma visualmente a excelente adequação do modelo aos dados históricos de temperatura e vendas.*

## **💡 Insights e Próximos Passos (Conclusão)**

### **Insights Adquiridos**

1. **Interpretabilidade do Coeficiente:** O valor do **Coeficiente Angular** registrado no MLflow (\~25.04) é o insight de negócio mais valioso. Ele indica que, em média, para cada aumento de **1°C** na temperatura, a Gelato Mágico pode esperar vender **25 unidades** a mais de sorvete.  
2. **Qualidade do Ajuste:** O **R2 Score** próximo de 1.0 sugere que a temperatura sozinha explica quase toda a variação nas vendas, tornando o modelo simples extremamente confiável para o planejamento.  
3. **Reprodutibilidade:** O uso do MLflow garante que qualquer colega de equipe possa inspecionar o run\_id, carregar os mesmos parâmetros e o mesmo modelo, garantindo a reprodutibilidade da análise.

### **Possibilidades de Melhoria (MLOps e Business)**

* **Aprimoramento do Modelo:** Adicionar recursos preditivos (Features) como **Umidade**, **Dia da Semana** (sábado/domingo), **Promoções** e **Feriados**. Isso exigiria um modelo de Regressão Múltipla.  
* **Versionamento de Dados:** Utilizar ferramentas como DVC (Data Version Control) para rastrear o vendas\_temperatura.txt juntamente com o modelo, fechando o ciclo de linhagem de dados.  
* **Deploy Automatizado:** Integrar o registro de modelo do MLflow com um serviço de Cloud (como Azure ML ou AWS SageMaker) para criar um **Endpoint de API** real, permitindo que o sistema de inventário da sorveteria chame o modelo em produção de forma contínua.  
* **Monitoramento:** Implementar monitoramento em produção para detectar **Drift de Dados** (ex: o clima muda e o modelo fica obsoleto) ou **Drift de Conceito** (ex: o comportamento do consumidor muda), garantindo que o modelo seja retreinado quando necessário.
