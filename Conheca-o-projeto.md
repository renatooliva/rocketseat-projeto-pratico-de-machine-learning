Conheça o projeto
Projeto prático de Machine Learning com dados de clientes, envolvendo predição de renda, classificação de categoria e segmentação por clusters, com notebooks, métricas, modelos salvos e entrega via GitHub.

Instruções
Estrutura, regras e requisitos do projeto

Introdução
Ao longo desta pós-graduação você estudou os fundamentos práticos de Machine Learning: análise exploratória, pré-processamento, regressão, classificação, clusterização, validação de modelos e otimização de hiperparâmetros. Este desafio final é o momento de unir tudo isso em um único projeto aplicado a um cenário realista.

Cenário
Você foi contratado como cientista de dados júnior por uma instituição financeira fictícia. A área de Inteligência de Negócios precisa de três sistemas analíticos complementares para apoiar campanhas de marketing:

Estimar a renda mensal de potenciais clientes a partir de dados cadastrais e comportamentais.
Classificar a categoria do cliente (Bronze / Prata / Ouro / Platinum) para segmentação de ofertas e campanhas.
Segmentar clientes em grupos naturais (sem rótulo pré-definido) para apoiar ações de marketing direcionadas.
O arquivo de entrada que você receberá é o clientes.csv (2000 linhas), com as seguintes colunas:

Coluna	Tipo	Descrição
ID_Cliente	inteiro	Identificador único (descartar na modelagem)
Nome	texto	Nome do cliente (descartar na modelagem)
Idade	inteiro	Idade em anos (18-70)
Escolaridade	categórica	Fundamental / Médio / Superior / Pós
Tempo_Emprego	inteiro	Anos no emprego atual (0-40)
Estado_Civil	categórica	Solteiro / Casado / Divorciado / Viúvo
Dependentes	inteiro	Número de dependentes (0-5)
Score_Credito	inteiro	Pontuação de crédito (0-1000)
Gasto_Mensal_Cartao	float	Gasto médio mensal no cartão (R$)
Investimentos	float	Valor total investido (R$)
Dividas	float	Valor total de dívidas (R$)
Renda	float	TARGET do Desafio 1 — renda mensal (R$)
Categoria_Cliente	categórica	TARGET do Desafio 2 — Bronze / Prata / Ouro / Platinum
Atenção: o arquivo clientes.csv será disponibilizado separadamente. Crie uma pasta datasets/ no seu projeto, coloque o clientes.csv dentro dela e use exatamente o caminho datasets/clientes.csv nos notebooks.

Requisitos do Desafio
Requisitos Obrigatórios
Cada requisito está mapeado para os conteúdos estudados. Todos precisam estar presentes e podem ser verificados pela leitura do código dos notebooks.

Desafio 1 — Predição de Renda
Conteúdo relacionado: Regressão linear e avaliação de regressão.

O notebook desafio_1_renda.ipynb deve carregar datasets/clientes.csv e fazer a exploração inicial (info, head, describe).
O código deve plotar histogramas das variáveis numéricas.
O código deve calcular e plotar a matriz de correlação das numéricas com a target.
O código deve aplicar ANOVA (f_oneway) de Renda por variáveis categóricas.
O pré-processamento deve remover ID_Cliente e Nome.
O pré-processamento deve aplicar encoding nas variáveis categóricas.
Deve haver split treino/teste (ex.: train_test_split).
O modelo deve usar sklearn.linear_model.LinearRegression.
O código deve calcular RMSE, MAE e R² no conjunto de teste.
O código deve aplicar validação cruzada de 5 folds (ex.: cross_val_score) e calcular o R² médio.
O código deve plotar a análise de resíduos: gráfico de real x predito e histograma dos resíduos.
O código deve montar uma tabela dos coeficientes do modelo ordenada por magnitude (os mais influentes).
O modelo treinado deve ser salvo como modelo_renda.pkl (joblib.dump).
O código deve conter uma seção de conclusões (em markdown ou comentário) comentando os resultados de RMSE, MAE e R².
Desafio 2 — Classificação de Categoria do Cliente
Conteúdo relacionado: Classificação multi-classe e tuning de hiperparâmetros.

O notebook desafio_2_categoria.ipynb deve carregar os dados e fazer a exploração inicial.
A variável Renda NÃO pode ser usada como feature (está correlacionada com a target e trivializaria o problema).
O código deve calcular/plotar a distribuição das classes da target (value_counts).
O código deve plotar boxplots das numéricas por categoria.
O código deve aplicar ANOVA (f_oneway) das numéricas e qui-quadrado (chi2_contingency) das categóricas em relação à categoria.
O pré-processamento deve remover ID_Cliente, Nome e Renda.
O pré-processamento deve aplicar encoding nas variáveis categóricas.
O split deve ser estratificado pela target (stratify).
O modelo deve usar sklearn.ensemble.RandomForestClassifier.
Deve haver um baseline com parâmetros padrão e um tuning via GridSearchCV em pelo menos 2 hiperparâmetros (ex.: n_estimators, max_depth).
Deve haver validação cruzada estratificada de 5 folds com scoring='f1_macro'.
O código deve chamar o classification_report do modelo final no conjunto de teste.
O código deve calcular/plotar a matriz de confusão (confusion_matrix).
O código deve obter/plotar a feature importance (feature_importances_) do modelo final.
O modelo treinado deve ser salvo como modelo_categoria.pkl (joblib.dump).
O código deve conter uma seção de conclusões (em markdown ou comentário) comentando o f1_macro e as classes com melhor e pior desempenho.
Desafio 3 — Segmentação de Clientes
Conteúdo relacionado: Clusterização não supervisionada e otimização de k.

O notebook desafio_3_segmentacao.ipynb deve carregar os dados e usar apenas as 5 features numéricas: Renda, Investimentos, Score_Credito, Gasto_Mensal_Cartao e Dividas.
É proibido usar ID_Cliente, Nome, Categoria_Cliente, Idade, Tempo_Emprego, Dependentes, Escolaridade e Estado_Civil na clusterização.
O código deve plotar histogramas das 5 numéricas.
O código deve plotar boxplots para detecção de outliers.
O código deve calcular/plotar a matriz de correlação das 5 features.
O pré-processamento deve usar um ColumnTransformer com StandardScaler sobre as 5 features.
A otimização de n_clusters deve usar optuna.create_study com GridSampler sobre range(2, 11).
A objective do Optuna deve retornar o silhouette_score, com direction="maximize".
O notebook deve treinar um Pipeline final (ColumnTransformer + KMeans).
O código deve perfilar os clusters: média de cada feature por cluster e o tamanho de cada cluster (contagem e percentual, ex.: via groupby).
O Pipeline completo (preprocessor + KMeans) deve ser salvo como modelo_segmentacao.pkl (joblib.dump) — não um KMeans isolado.
O código deve conter uma seção de conclusões (em markdown ou comentário) comentando o melhor k, o silhouette_score e o perfil de cada cluster.
Organização e qualidade do código
Conteúdo relacionado: Boas práticas de código e reprodutibilidade.

O código não deve concentrar toda a lógica em um único bloco; as etapas (exploração, pré-processamento, modelagem, avaliação) devem estar separadas em seções ou funções com responsabilidade clara.
Variáveis e funções devem ter nomes descritivos.
O código deve conter markdown ou comentários explicando as etapas e decisões.
O pré-processamento (encoders/scalers) deve ser ajustado após o split treino/teste, para evitar vazamento de dados.
Caso utilize tratamento de erros, prefira capturar exceções específicas a um except genérico.
Requisitos Opcionais
Os requisitos opcionais não são obrigatórios, mas demonstram domínio adicional de análise de dados. Implemente um ou mais para enriquecer o projeto.

Visualização dos clusters com PCA 2D
Conteúdo relacionado: Redução de dimensionalidade para visualização.

Aplique PCA para projetar as 5 features numéricas em 2 dimensões e gere um gráfico de dispersão colorido pelos clusters do Desafio 3.
A projeção PCA é apenas para visualização — não deve entrar no pipeline persistido em modelo_segmentacao.pkl.
Enriquecimento da análise
Conteúdo relacionado: Interpretação de resultados e comunicação.

Nomeie os clusters em linguagem de negócio (ex.: Premium, Risco, Conservador).
Adicione gráficos extras de apoio à análise.
Explore tuning adicional de hiperparâmetros.
Crie documentação complementar do projeto.
Checklist Antes de Enviar o Projeto
Antes de publicar o repositório no GitHub, revise cada item abaixo:

Código
 O código salva os três modelos com joblib.dump e os nomes corretos.
 O Desafio 2 não usa Renda como feature.
 O Desafio 3 usa apenas as 5 features numéricas e persiste o Pipeline completo.
Repositório
 O repositório está público.
 Os três notebooks estão no repositório.
 A pasta datasets/ com o clientes.csv está no repositório.
 O README.md básico foi incluído.
Opcional
 Se implementou extras, eles estão separados e não quebram os obrigatórios.
Observações Finais sobre a Entrega
Formato obrigatório: o projeto deve ser publicado em um repositório público no GitHub contendo os três notebooks .ipynb. Entregas por e-mail, Google Drive ou qualquer outro meio não serão aceitas.

Modelos salvos pelo código: o que é avaliado é o código que salva cada modelo com joblib.dump e o nome correto — não é necessário versionar os arquivos .pkl no repositório. Gerá-los ao executar o notebook é suficiente.

Plágio: o projeto é individual. Código idêntico ou muito semelhante ao de outro aluno resultará em nota zero para ambos, sem exceções.

Bibliotecas: o projeto usa scikit-learn, optuna, joblib e jupyter (além de pandas, matplotlib e seaborn para análise). Para rodar localmente:

pip install scikit-learn optuna joblib jupyter pandas matplotlib seaborn
Compatibilidade: os notebooks devem rodar com Python 3.12, tanto no Google Colab quanto no Jupyter Notebook local.

Caminho do dataset: a leitura dos dados deve usar exatamente datasets/clientes.csv.

Nomeação dos arquivos: use exatamente os nomes desafio_1_renda.ipynb, desafio_2_categoria.ipynb e desafio_3_segmentacao.ipynb para os notebooks, e salve os modelos via joblib.dump com os nomes exatos modelo_renda.pkl, modelo_categoria.pkl e modelo_segmentacao.pkl. Nomes diferentes serão desconsiderados na avaliação.

Dicas para Realizar o Desafio
Por onde começar?
Organize o dataset. Crie a pasta datasets/ e coloque o clientes.csv dentro, garantindo o caminho datasets/clientes.csv.
Prepare o ambiente no Python 3.12 com as dependências instaladas.
Faça um notebook por desafio, seguindo as etapas dos Requisitos Obrigatórios (exploração → pré-processamento → modelagem → avaliação → conclusões).
Salve cada modelo com o nome exato (modelo_renda.pkl, modelo_categoria.pkl, modelo_segmentacao.pkl).
Revise os notebooks antes de enviar.
Dicas por desafio
Desafio	Dica
Desafio 1	Ajuste encoders/scalers com estado (que aprendem do treino, ex.: StandardScaler, OneHotEncoder) após o split treino/teste para evitar vazamento; transformações sem estado como pd.get_dummies podem ser aplicadas antes.
Desafio 2	Lembre-se de remover Renda antes de treinar; compare o baseline com o modelo tunado, nunca o tunado contra ele mesmo.
Desafio 3	Calcule o silhouette sobre as features transformadas pelo preprocessor do pipeline; features sem sinal de agrupamento derrubam o score, por isso a seleção criteriosa das 5 numéricas.
Metas de referência
Use os valores abaixo como referência de qualidade dos seus modelos — eles indicam um modelo bem construído:

Desafio	Métrica de referência
Desafio 1	R² no teste ≥ 0.70
Desafio 2	f1_macro no teste ≥ 0.75
Desafio 3	silhouette_score do melhor k ≥ 0.18, com k entre 3 e 6
Conteúdos de referência
Consulte os materiais estudados na pós conforme avança em cada desafio:

Desafio	Conteúdos relacionados
Geral (análise exploratória)	Estatística aplicada (correlação, ANOVA, qui-quadrado), Meu primeiro modelo com Scikit-Learn
Desafio 1 — Renda	Regressão Linear, Regressão Linear Múltipla, Regressão Polinomial
Desafio 2 — Categoria	Introdução ao Scikit-Learn e Ensemble, Random Forest, Classificação com Árvore de Decisão, Naive Bayes e Regressão Logística
Desafio 3 — Segmentação	Clusterização com K-Means, Clusterização Hierárquica, Tópicos complementares de aprendizado não supervisionado
Opcional (PCA)	Redução de dimensionalidade (PCA), t-SNE
Como Será Feita a Entrega
A entrega é feita exclusivamente via repositório público no GitHub contendo os três notebooks.

Estrutura esperada do repositório
pos-ml-desafio/
├── datasets/
│   └── clientes.csv             ← dataset fornecido (mesmo caminho usado nos notebooks)  ✅
├── desafio_1_renda.ipynb        ← notebook do Desafio 1  ✅
├── desafio_2_categoria.ipynb    ← notebook do Desafio 2  ✅
├── desafio_3_segmentacao.ipynb  ← notebook do Desafio 3  ✅
└── README.md                    ← descrição básica do projeto           ✅
Os arquivos modelo_*.pkl são gerados ao executar os notebooks (joblib.dump) e não precisam ser versionados — o que é avaliado é o código que os salva.

Passo a passo
Revise os notebooks e confira que cada etapa do desafio está implementada.
Confirme que o código salva os três modelos via joblib.dump com os nomes exatos.
Crie um repositório público no 
github.com
 e suba os notebooks, a pasta datasets/ com o clientes.csv e o README.md.
Envie o link público do repositório conforme o canal de entrega indicado.
O que deve constar no README.md
O README.md do seu repositório deve conter no mínimo:

Título e descrição do projeto.
Como executar (abrir no Jupyter ou Colab e rodar todas as células em ordem).
O que cada notebook gera como saída (métricas e o arquivo .pkl correspondente).
Link de entrega
Após publicar o repositório, envie o link público do repositório no GitHub no canal do desafio, assim ele será avaliado pelo time.

Considerações Finais
Este desafio representa um projeto de Machine Learning de ponta a ponta. Não é apenas sobre fazer o modelo treinar — é sobre construir um fluxo que outra pessoa consiga ler, entender e reproduzir, com decisões justificadas e resultados interpretados.

Se em algum momento você travar, releia o material da disciplina correspondente à etapa que está implementando. Bons estudos e boa prática! 💜