import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - lendo o arquivo
df = pd.read_csv('medical_examination.csv')

# 2 - criando a coluna de overweight com base na formula do BMI e transformando a altura pra metros
df['overweight'] = df['weight'] / ((df['height'] / 100) * (df['height'] / 100))
# transformando os valores pra 0 e 1, com 0 sendo "não overweight" e 1 sendo "sim overweight"
df['overweight'] = df['overweight'].apply(lambda x: 1 if x > 25 else 0) # lambda é parecido com as funções anonimas do JS

# 3 - normalizando cholesterol e gluc pra ficar entre 0 e 1
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5 - usando o pd.melt pra formatar o df para o grafico ser criado
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6 - agrupando, reformatando os dados para separar por cardio e mostrando o total de cada feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    

    # 7 - criando o grafico
    sns.catplot(data=df_cat, x='variable', y='size', hue='value', col='cardio', kind='bar').set_ylabels('total')


    # 8 - salvando o grafico
    fig = plt.gcf()


    # 9- exportando o grafico
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11 - eliminando as linhas que não seguem as condições dentro dos [ ]
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 - calculando a correlation matrix selecionando as colunas certas pra passar na desgraça do teste
    corr = df_heat[
        ['id',
        'age',
        'sex',
        'height',
        'weight',
        'ap_hi',
        'ap_lo',
        'cholesterol',
        'gluc',
        'smoke',
        'alco',
        'active',
        'cardio',
        'overweight']
    ].corr()

    # 13 - criando a mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - cria a figura
    fig, ax = plt.subplots(figsize=(10, 10))

    # 15 - cria o grafico heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=.5, ax=ax, cmap='coolwarm')

    # 16 salva o grafico
    fig.savefig('heatmap.png')
    return fig
