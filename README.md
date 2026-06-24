# Análise de Depressão em Estudantes com Machine Learning

Este repositório contém o desenvolvimento do **Projeto 2** da disciplina de Machine Learning. O objetivo principal é investigar o *Student Depression Dataset* para responder de forma interpretável à questão clínica central: 

> **"Quais fatores acadêmicos e socioeconômicos aumentam a chance de depressão em estudantes?"**

O projeto foi construído seguindo rigorosos padrões de engenharia de dados, permitindo a execução de um pipeline completo: desde a ingestão do dado bruto até a aplicação de testes estatísticos avançados (Friedman e Wilcoxon) para a validação dos modelos.

## Fonte dos Dados (Dataset)

Os dados originais utilizados neste projeto pertencem ao **Student Depression Dataset**, disponibilizado publicamente na plataforma Kaggle.

Para reproduzir os resultados, faça o download dos dados através do link abaixo:
* **Link para download:** [Student Depression Dataset (Kaggle)](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

Após o download, coloque o arquivo `.csv` original dentro da pasta `data/raw/` do repositório.

## Abordagem e Metodologia

Para responder à pergunta do projeto com rigor científico, dividimos a nossa modelagem em duas frentes:
1. **Modelos de Alta Precisão (Caixas Pretas):** Utilização de Redes Neurais (MLP), SVM e KNN otimizados via GridSearch e K-Fold Cross Validation para garantir o diagnóstico mais acurado possível (foco em F1-Score e AUC).
2. **Modelos de Interpretabilidade (Caixas Brancas):** Utilização de Árvores de Decisão (Entropia) e Random Forest para abrir a "caixa preta" matemática e extrair o feature_importances_, revelando exatamente **quais fatores** causam a depressão.

Além disso, o pipeline foi desenhado para rodar **dois cenários distintos** a fim de evitar Data Leakage (vazamento de dados) com a variável de pensamentos suicidas.

## Estrutura do Repositório

* `data/`: Dados
  * `     raw/`: Base de dados original (arquivos CSV baixados do Kaggle).
  * `     results/`: Diretório de saída automatizada contendo as métricas (`tables/`) e os gráficos (`figures/`).
* `notebooks/`: Análise Exploratória de Dados (EDA) inicial da equipe.
* `src/`: Pipeline de Machine Learning modularizado:
  * `preprocessing.py`: Módulo de limpeza, encoding e StandardScaling.
  * `models/`: Implementação isolada de cada algoritmo com hiperparametrização.
  * `evaluation.py`: Módulo gerador de Matrizes de Confusão, ROC e F1-Score.
  * `statistical_tests.py`: Motor de testes de Wilcoxon e Friedman.
* `main.py`: Orquestrador mestre para execução sequencial de todo o projeto.

## Como Executar o Projeto

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/student-depression-ml.git
cd student-depression-ml
```

2. **Crie e ative um ambiente virtual (Recomendado):**
No Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
No Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Inicie o Pipeline Completo:**
O comando abaixo consumirá os dados em `data/raw/` e executará todas as etapas de pré-processamento, treinamento, validação e testes estatísticos, exportando os gráficos finais na pasta `results/`.
```bash
python main.py
```

## Protótipo interativo com Streamlit

Além do pipeline completo de modelagem, este projeto também possui um protótipo interativo desenvolvido com **Streamlit**.

O protótipo permite carregar os modelos treinados em formato `.pkl`, preencher os dados de um estudante por meio de uma interface gráfica e visualizar a predição do modelo para a classe-alvo `Depression`.

Para reproduzir a aplicação, gerar os arquivos necessários no notebook e executar o Streamlit localmente, consulte:

[Como rodar o protótipo Streamlit](docs/como-rodar-streamlit.md)

## Equipe de Desenvolvimento

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/alocinny">
        <img src="https://github.com/alocinny.png" width="100px;" alt=""/><br>
        <sub><b>Ana Beatriz Soares</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Danielle-sn">
        <img src="https://github.com/Danielle-sn.png" width="100px;" alt=""/><br>
        <sub><b>Danielle Stephany</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/michellydarquia">
        <img src="https://github.com/michellydarquia.png" width="100px;" alt=""/><br>
        <sub><b>Michelly Darquia</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="#">
        <img src="https://github.com/identicon.png" width="100px;" alt=""/><br>
        <sub><b>Gabriel Alves</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="#">
        <img src="https://github.com/identicon.png" width="100px;" alt=""/><br>
        <sub><b>Gabriel Agra</b></sub>
      </a>
    </td>
  </tr>
</table>

<div align="center">
  <p><b>Orientação Acadêmica</b></p>
  <img src="https://img.shields.io/badge/Orientador-Prof._Byron_Leite_Bezerra-0d1117?style=flat-square&logo=lecture&logoColor=39d353&color=0d1117&labelColor=0d1117" />
</div>
