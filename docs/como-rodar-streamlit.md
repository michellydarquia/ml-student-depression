
# Como rodar o protótipo Streamlit

Este documento explica como executar o protótipo interativo do projeto, desde a geração dos arquivos `.pkl` no notebook de modelagem até a inicialização da aplicação com Streamlit.

O objetivo do protótipo é demonstrar, de forma visual e interativa, o funcionamento dos modelos treinados para predição de depressão em estudantes.

---

## 1. Estrutura esperada do projeto

A estrutura recomendada para executar a aplicação é:

```text
ml-student-depression/
├── notebooks/
│   └── Modelagem_e_Avaliacao_dos_Algoritmos.ipynb
├── streamlit/
│   └── app.py
├── model_artifacts/
│   ├── bundle_sem_leak.pkl
│   └── bundle_com_leak.pkl
├── requirements.txt
└── README.md
```

A pasta `model_artifacts/` armazena os arquivos `.pkl` gerados após o treinamento dos modelos.

Esses arquivos são necessários para que o Streamlit consiga carregar os modelos e realizar as predições.

---

## 2. Gerar os arquivos `.pkl` no notebook

Antes de executar a aplicação Streamlit, é necessário gerar os arquivos dos modelos treinados.

Abra o notebook:

```text
notebooks/Modelagem_e_Avaliacao_dos_Algoritmos.ipynb
```

Execute as células do notebook até a etapa de exportação dos bundles.

Ao final da execução, os seguintes arquivos devem ser gerados:

```text
bundle_sem_leak.pkl
bundle_com_leak.pkl
```

Esses arquivos representam dois cenários de modelagem:

| Arquivo               | Descrição                                                   |
| --------------------- | ----------------------------------------------------------- |
| `bundle_sem_leak.pkl` | Cenário principal, sem a variável de pensamentos suicidas   |
| `bundle_com_leak.pkl` | Cenário comparativo, com a variável de pensamentos suicidas |

O arquivo `bundle_sem_leak.pkl` deve ser usado como cenário principal da apresentação, pois evita o uso de uma variável com forte risco de vazamento de dados (*target leakage*).

O arquivo `bundle_com_leak.pkl` é mantido apenas para comparação acadêmica.

---

## 3. Mover os arquivos para a pasta correta

Depois que os arquivos `.pkl` forem gerados, mova-os para a pasta:

```text
model_artifacts/
```

Caso a pasta ainda não exista, crie com:

```bash
mkdir -p model_artifacts
```

Depois mova os arquivos:

```bash
mv bundle_sem_leak.pkl model_artifacts/
mv bundle_com_leak.pkl model_artifacts/
```

Ao final, a estrutura deve ficar assim:

```text
model_artifacts/
├── bundle_sem_leak.pkl
└── bundle_com_leak.pkl
```

---

## 4. Criar e ativar o ambiente virtual

Na raiz do projeto, crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente.

No Linux/macOS:

```bash
source .venv/bin/activate
```

No Windows:

```bash
.venv\Scripts\activate
```

---

## 5. Instalar as dependências

Com o ambiente virtual ativado, instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

Caso ocorra erro relacionado ao `scikit-learn`, verifique se a versão usada no ambiente é compatível com a versão usada para salvar os arquivos `.pkl`.

Para este projeto, recomenda-se usar:

```bash
pip install scikit-learn==1.6.1
```

Essa versão evita avisos de incompatibilidade ao carregar os modelos salvos com `joblib`.

---

## 6. Executar a aplicação Streamlit

Na raiz do projeto, execute:

```bash
streamlit run streamlit/app.py
```

Se o comando `streamlit` não for reconhecido, use:

```bash
python -m streamlit run streamlit/app.py
```

Após a execução, o terminal exibirá um endereço local semelhante a:

```text
http://localhost:8501
```

Abra esse endereço no navegador para acessar a interface da aplicação.

---

## 7. Como usar a aplicação

Na interface do Streamlit:

1. Selecione o cenário de modelagem na barra lateral.
2. Escolha o modelo treinado que deseja testar.
3. Preencha os dados do estudante no formulário.
4. Clique em **Prever**.
5. Observe a classe prevista e, quando disponível, a probabilidade estimada.

A aplicação permite testar diferentes entradas e observar como os modelos reagem a mudanças nos atributos informados.

---

## 8. Cenários disponíveis

A aplicação possui dois cenários:

### Cenário principal

```text
Sem a variável-alvo de risco
```

Este é o cenário recomendado para demonstração principal, pois não utiliza a variável de pensamentos suicidas na entrada do modelo.

### Cenário comparativo

```text
Com a variável de pensamentos suicidas
```

Este cenário é usado apenas para comparação, pois essa variável pode causar *target leakage*, já que possui relação muito direta com a variável-alvo.

---

## 9. Possíveis problemas

### Erro: arquivo `.pkl` não encontrado

Verifique se os arquivos estão dentro da pasta:

```text
model_artifacts/
```

Os nomes esperados são:

```text
bundle_sem_leak.pkl
bundle_com_leak.pkl
```

### Aviso: `InconsistentVersionWarning`

Esse aviso indica que o modelo foi salvo com uma versão do `scikit-learn` e está sendo carregado com outra.

Para corrigir, instale a versão compatível:

```bash
pip install scikit-learn==1.6.1
```

### Aviso: `missing ScriptRunContext`

Esse aviso geralmente aparece quando o app é executado com:

```bash
python streamlit/app.py
```

O correto é executar com:

```bash
streamlit run streamlit/app.py
```

---

## 10. Observação acadêmica

Esta aplicação tem finalidade demonstrativa e acadêmica.

Ela não deve ser usada como ferramenta clínica ou diagnóstica. A predição feita pelo modelo representa apenas o comportamento estatístico aprendido a partir da base de dados utilizada no projeto.
