#Machine Learning
#Alessandro Pereira Torres
#criando um modelo de classificacao para Pulsares e Nao Pulsares


#1)DATASET
#O conjunto de dados contendo todas as caracteristicas de varias estrelas de neutrons foram
#obtidas pelo kaggle.
#https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star

#2)Introdução
#Uma estrela é uma grande e luminosa esfera de plasma, composta basicamente por hélio e hidrogênio,
#entretanto,  apos consumir todo o seu hidrogênio, e se a estrela for grande,ela explode em uma supernova. 
#apois isso, ela vira um corpo celeste extremamente denso e compacto sem átomos, 
#apenas nêutrons. Por isso o nome: estrela de nêutrons.

#Uma estrela de nêutrons gira muuuuito rapido. 
#Quando o campo magnético da estrela de nêutrons não coincide com o seu eixo de rotação temos um pulsar: 
#uma estrela que emite um feixe de radiaçao nos polos magneticos, proveniente de seu movimento de rotação, 
#de forma rapida e periodicamente regular.

#3)Objetivo 
#O objetivo deste trabalho é criar um modelo de classificação para estrelas de neutrons de modo a classificar
#elas em Pulsares ou Nao pulsares atravez das demais caracteristicas que elas possuem , usando normalização e KMeans

#4)Descrição dos resultados obtidos, explicando o que foi feito 

#5)Análise dos dados, com as figuras

#Importando Bibliotecas#############################################
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


#Importando o DataSet - Pulsares####################################
print("Criando um modelo de classificacao para Pulsares e Nao Pulsares")
print("")
dataset = pd.read_csv('pulsar_stars.csv')

print("Variaveis do banco de dados :")
print(dataset.dtypes)
#obs: target_class 
#     1 = Pulsar
#     0 = Nao Pulsar



#Contando os Pulsares e Nao Pulsares#################################
Y_data = dataset.iloc[:,-1].values
unique , counts = np.unique(Y_data,return_counts=True)
print("")
print("Obs: 0 = Nao Pulsar , 1 = Pulsar")
print(unique,counts)
print("")


#Normalizando os dados###############################################
print("Normalizando os dados")
dataset_scaled = preprocessing.scale(dataset.iloc[:,0:8])
print("Media : ",round(dataset_scaled.mean()))
print("Variancia : ",dataset_scaled.std())

#Criando um modelo de PCA########################################
pca = PCA(n_components=2)

#Transformando o DataSet 
#pegando todos os dados para PCA (menos target_class)
reduced_data2_pca = pca.fit_transform(dataset_scaled)


print("")
print("Com os dados normalizados e com a ajuda do Kmeans")
print("vamos criar um modelo de classicicacao para dizer")
print("o que seria Pulsar e Nao Pulsar")
print("")



# rodando o Kmeans com 2 clusters (pulsar e nao pulsar)
kmeans = KMeans(n_clusters=2, random_state=0)
dataset_scaled = preprocessing.scale(dataset.iloc[:,0:8])
clusters = kmeans.fit_predict(dataset_scaled)

#Plotando o grafico dos Pulsares e Nao Pulsares Linearmente separados##
print("")
print("AVISO-Caso o kmeans troque o 0 com 1 basta rodar o codigo de novo, esse erro ocorre por conta do enorme numero de nao pulsares(0) em relacao aos pulsares(1)-AVISO" )
colors = ['blue', 'red']
for i in range(len(colors)):
    x = reduced_data2_pca[:, 0][clusters == i]
    y = reduced_data2_pca[:, 1][clusters == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(unique, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Primeira Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.title("Grafico da APC Normalizado + Kmeans")
plt.show()
print("")


#Resultados###########################################################
print("Observa-se que o Kmeans consequiu de forma eficiente")
print("separar linearmente os dados e fazer uma boa classificacao")
print("de Pulsares e Nao Pulsares")
print("")
print (classification_report(dataset.iloc[:,-1], clusters))
cmtx = pd.DataFrame(
    confusion_matrix(dataset.iloc[:,-1], clusters), 
    index=['Real:NPulsar', 'Real:Pulsar'], 
    columns=['Predito:NPulsar', 'Pretido:Pulsar']
)
print("Matrix de Confucao")
print(cmtx)