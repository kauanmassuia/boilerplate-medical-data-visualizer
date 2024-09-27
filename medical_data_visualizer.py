import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Exibir todas as colunas no DataFrame
pd.set_option('display.max_columns', None)

# Lê o arquivo CSV com os dados médicos
df = pd.read_csv('medical_examination.csv')

# Calcula o IMC e cria uma nova coluna 'overweight' (acima do peso)
# Se o IMC for maior que 25, marca como 1 (acima do peso), se não, marca 0
df['overweight'] = (df['weight'] / ((df["height"] / 100) ** 2)).apply(lambda x : 1 if x > 25 else 0)

# Ajusta os valores de colesterol e glicose: se for 1, deixa como 0 (normal), senão vira 1 (alterado)
df['cholesterol'] = df['cholesterol'].apply(lambda x : 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x : 0 if x == 1 else 1)


def draw_cat_plot():
    # Faz a transformação dos dados para plotar o gráfico categórico
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Cria uma coluna chamada 'total' para contar os valores
    df_cat['total'] = 1
    # Agrupa os dados pelo status de 'cardio', pela variável, e pelo valor (0 ou 1)
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index = False).count()

    # Faz o gráfico de barras categórico
    fig = sns.catplot(x='variable', y='total', data=df_cat, hue='value', kind='bar', col='cardio').figure
    # Salva a figura
    fig.savefig('catplot.png')
    
    return fig


def draw_heat_map():

    # Limpa os dados removendo as linhas onde pressão diastólica (ap_lo) é maior que a sistólica (ap_hi)
    # Também remove os outliers de altura e peso (usando quantis)
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975)) 
    ]

    # Calcula a matriz de correlação usando o método de Pearson
    corr = df_heat.corr(method='pearson')

    # Cria uma máscara para o triângulo superior da matriz de correlação (para esconder)
    mask = np.triu(corr)

    # Cria a figura e o eixo para o heatmap
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plota o heatmap com a matriz de correlação
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=1, square=True, cbar_kws={'shrink': 0.5}, center=0.08, ax=ax)

    # Salva o heatmap em um arquivo
    fig.savefig('heatmap.png')

    return fig
