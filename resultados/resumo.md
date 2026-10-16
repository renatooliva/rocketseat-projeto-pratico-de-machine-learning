# Resultados da execução

Execução em 2026-10-16 11:56. Python 3.12.7. Os três notebooks rodaram da raiz do repositório, de cima para baixo.

Versões: pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, scipy 1.18.1, optuna 5.0.0, joblib 1.6.0.

A saída completa de cada notebook, incluindo tabelas, está no `saida.txt` da pasta correspondente.

## Desafio 1 — Predição de renda

```
<class 'pandas.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 13 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   ID_Cliente           2000 non-null   int64  
 1   Nome                 2000 non-null   str    
 2   Idade                2000 non-null   int64  
 3   Escolaridade         2000 non-null   str    
 4   Tempo_Emprego        2000 non-null   int64  
 5   Estado_Civil         2000 non-null   str    
 6   Dependentes          2000 non-null   int64  
 7   Score_Credito        2000 non-null   int64  
 8   Gasto_Mensal_Cartao  2000 non-null   float64
 9   Investimentos        2000 non-null   float64
 10  Dividas              2000 non-null   float64
 11  Renda                2000 non-null   float64
 12  Categoria_Cliente    2000 non-null   str    
dtypes: float64(4), int64(5), str(4)
memory usage: 203.3 KB
Escolaridade: F=421.38  p=4.986e-212
Estado_Civil: F=8.45  p=1.409e-05
Categoria_Cliente: F=110.25  p=4.736e-66
RMSE: 995.63
MAE: 808.95
R²: 0.873
R² médio (5 folds): 0.872
Referências do one-hot: Escolaridade=Fundamental, Estado_Civil=Casado
No teste, o R² foi 0.873, o RMSE foi R$ 995.63 e o MAE foi R$ 808.95. A renda média do teste é R$ 7,654.46, então o MAE corresponde a 10.6% desse valor. A validação cruzada de 5 folds no treino teve R² médio de 0.872.
A média dos resíduos no teste foi R$ -4.82. O gráfico de real contra predito e o histograma dos resíduos mostram esse erro em torno de zero.
Maiores magnitudes:
```

Figuras:

- `desafio_1_renda/figuras/distribuicao_numericas.png`
- `desafio_1_renda/figuras/correlacao.png`
- `desafio_1_renda/figuras/residuos.png`

## Desafio 2 — Classificação da categoria

```
<class 'pandas.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 13 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   ID_Cliente           2000 non-null   int64  
 1   Nome                 2000 non-null   str    
 2   Idade                2000 non-null   int64  
 3   Escolaridade         2000 non-null   str    
 4   Tempo_Emprego        2000 non-null   int64  
 5   Estado_Civil         2000 non-null   str    
 6   Dependentes          2000 non-null   int64  
 7   Score_Credito        2000 non-null   int64  
 8   Gasto_Mensal_Cartao  2000 non-null   float64
 9   Investimentos        2000 non-null   float64
 10  Dividas              2000 non-null   float64
 11  Renda                2000 non-null   float64
 12  Categoria_Cliente    2000 non-null   str    
dtypes: float64(4), int64(5), str(4)
memory usage: 203.3 KB
ANOVA Idade: F=49.73  p=5.448e-31
ANOVA Tempo_Emprego: F=42.91  p=7.23e-27
ANOVA Dependentes: F=0.01  p=0.9984
ANOVA Score_Credito: F=2686.43  p=0
ANOVA Gasto_Mensal_Cartao: F=89.53  p=2.348e-54
ANOVA Investimentos: F=1327.77  p=0
ANOVA Dividas: F=337.37  p=3.451e-177
ANOVA Renda: F=110.25  p=4.736e-66

Qui-quadrado Escolaridade: chi2=118.39  gl=9  p=2.859e-21
Qui-quadrado Estado_Civil: chi2=6.11  gl=9  p=0.7284
F1 macro do baseline na validação cruzada: 0.8119
F1 macro do modelo tunado na validação cruzada: 0.8121
Melhores parâmetros: {'floresta__max_depth': None, 'floresta__n_estimators': 200}
              precision    recall  f1-score   support

      Bronze      0.947     0.926     0.937       136
        Ouro      0.667     0.659     0.663        88
    Platinum      0.816     0.714     0.762        56
       Prata      0.763     0.833     0.797       120

    accuracy                          0.810       400
   macro avg      0.798     0.783     0.790       400
weighted avg      0.812     0.810     0.810       400

f1_macro no teste: 0.790
f1_macro do baseline no teste: 0.797
O baseline teve f1_macro de 0.8119 na validação cruzada e 0.797 no teste. O modelo tunado teve 0.8121 na validação cruzada e f1_macro de 0.790 no teste. A busca escolhe pelo resultado da validação cruzada. No teste, o baseline ficou levemente acima, e a diferença é pequena. O arquivo salvo é o modelo escolhido na validação.
Melhor classe no teste: Bronze (F1=0.937). Pior classe no teste: Ouro (F1=0.663).
F1 por classe:
```

Figuras:

- `desafio_2_categoria/figuras/distribuicao_classes.png`
- `desafio_2_categoria/figuras/boxplots_por_categoria.png`
- `desafio_2_categoria/figuras/matriz_confusao.png`
- `desafio_2_categoria/figuras/importancia_variaveis.png`

## Desafio 3 — Segmentação

```
Melhor k: 4
silhouette_score: 0.393
Melhor k: 4. silhouette_score: 0.393.
Perfil de cada cluster:
Conservador (cluster 0): 868 clientes (43.4%). Renda média R$ 5,132, investimentos R$ 19,838, score 556, gasto no cartão R$ 1,035, dívidas R$ 5,527.
Premium (cluster 1): 465 clientes (23.2%). Renda média R$ 9,849, investimentos R$ 131,911, score 906, gasto no cartão R$ 1,836, dívidas R$ 4,952.
Risco (cluster 2): 359 clientes (17.9%). Renda média R$ 9,096, investimentos R$ 5,838, score 262, gasto no cartão R$ 1,376, dívidas R$ 34,086.
Alto gasto (cluster 3): 308 clientes (15.4%). Renda média R$ 9,468, investimentos R$ 18,110, score 512, gasto no cartão R$ 3,346, dívidas R$ 14,997.
```

Figuras:

- `desafio_3_segmentacao/figuras/distribuicao_features.png`
- `desafio_3_segmentacao/figuras/boxplots.png`
- `desafio_3_segmentacao/figuras/correlacao.png`
- `desafio_3_segmentacao/figuras/pca.png`
- `desafio_3_segmentacao/figuras/perfil_padronizado.png`

## Modelos gravados na raiz
- `modelo_renda.pkl`: Pipeline ['preprocessor', 'regressao'], 4.1 KB
- `modelo_categoria.pkl`: Pipeline ['preprocessor', 'floresta'], 9.8 MB
- `modelo_segmentacao.pkl`: Pipeline ['preprocessor', 'kmeans'], 10.3 KB

