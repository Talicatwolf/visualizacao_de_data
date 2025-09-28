import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Leitura do arquivo CSV
df = pd.read_csv('ecommerce_estatistica.csv')

# Análise inicial dos dados
print(df.info(),"="* 30, end="\n" )
print(df.describe(),"="* 30, end="\n")
print(df.head(),"="* 30, end="\n")

# Exemplo de colunas para análise (ajuste conforme seu dataset)
# Supondo colunas: 'Idade', 'ValorCompra', 'Categoria', 'Sexo', 'Avaliação'

# Histograma: Distribuição do Valor das Compras
plt.figure(figsize=(12,9))
plt.hist(df['Preço'], bins=30, color='blue', edgecolor='black')
plt.title('Distribuição do Valor das Compras')
plt.xlabel('Valor da Compra')
plt.ylabel('Frequência')
plt.xlim(0, df['Preço'].max()) # Set x-axis limit to the maximum price
plt.show()

# Gráfico de Dispersão: nota vs Valor da Compra
plt.figure(figsize=(12,9))
plt.scatter(df['Nota'], df['Preço'], alpha=0.5)
plt.title('Nota vs Preço')
plt.xlabel('Nota')
plt.ylabel('Preço')
plt.show()

# Mapa de Calor: Correlação entre variáveis numéricas
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Mapa de Calor das Correlações')
plt.show()

# Gráfico de Barra: Média de Valor da Compra por Categoria
plt.figure(figsize=(12,9))
df.groupby('Temporada')['Preço'].mean().sort_values().plot(kind='bar', color='orange')
plt.title('Média do Valor do Preço por Temporada')
plt.xlabel('Temporada')
plt.ylabel('Média do Valor do Preço')
plt.show()

# Gráfico de Pizza: Distribuição de Compras por Sexo
plt.figure(figsize=(10,10))
gender_counts = df['Gênero'].value_counts().nlargest(5)
colors = plt.cm.get_cmap('tab10', len(gender_counts))
plt.pie(gender_counts, startangle=90, colors=colors.colors)
plt.title('Distribuição de Compras por Gênero (Top 5)')
plt.ylabel('')
plt.legend(gender_counts.index, title="Gênero", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

# Gráfico de Densidade: Valor da Compra
plt.figure(figsize=(12,9))
sns.kdeplot(df['Preço'], fill=True, color='green') # Use fill instead of shade
plt.title('Densidade do Valor do Preço')
plt.xlabel('Valor do Preço')
plt.ylabel('Densidade')
plt.show()

# Gráfico de Regressão: Desconto vs N_Avaliações
plt.figure(figsize=(12,9))
sns.regplot(x='Desconto', y='N_Avaliações', data=df, scatter_kws={'alpha':0.5})
plt.title('Regressão: Desconto vs N_Avaliações')
plt.xlabel('Desconto')
plt.ylabel('N_Avaliações')
plt.show()