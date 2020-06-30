# Classificando-estrelas-com-Kmeans
Criando um modelo de classificação para estrelas de nêutrons



### 1)DATASET
O conjunto de dados contendo todas as características de várias estrelas de nêutrons foram
obtidas pelo kaggle.
https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star


### 2)INTRODUÇÃO
Uma estrela é uma grande e luminosa esfera de plasma, composta basicamente por hélio e hidrogênio,
entretanto, após consumir todo o seu hidrogênio, e se a estrela for grande, ela explode em uma supernova. 
após isso, ela vira um corpo celeste extremamente denso e compacto sem átomos, apenas nêutrons. Por isso o nome: estrela de nêutrons.
![1](https://user-images.githubusercontent.com/50224653/86123767-6d5c5700-bab0-11ea-954e-75f224f65bd7.PNG)

Uma estrela de nêutrons gira muuuuito rápido. 
Quando o campo magnético da estrela de nêutrons não coincide com o seu eixo de rotação temos um pulsar: 
uma estrela que emite um feixe de radiação nos polos magnéticos, proveniente de seu movimento de rotação, de forma rápida e periodicamente regular
(Azul = não pulsar / Vermelho = pulsar).
![2](https://user-images.githubusercontent.com/50224653/86123761-6a616680-bab0-11ea-9212-a67662e1434f.png)

### 3)OBJETIVO
O objetivo deste trabalho é criar um modelo de classificação para estrelas de nêutrons de modo a classificar elas em pulsares ou não pulsares através das demais características que elas possuem, usando normalização e KMeans

```Python
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
dataset = pd.read_csv('pulsar_stars.csv')
#obs: target_class 
#     1 = Pulsar
#     0 = Nao Pulsar

#Contando os Pulsares e Nao Pulsares#################################
Y_data = dataset.iloc[:,-1].values
unique , counts = np.unique(Y_data,return_counts=True)

#Normalizando os dados###############################################
dataset_scaled = preprocessing.scale(dataset.iloc[:,0:8])

#Criando um modelo de PCA########################################
pca = PCA(n_components=2)

#Transformando o DataSet 
#pegando todos os dados para PCA (menos target_class)
reduced_data2_pca = pca.fit_transform(dataset_scaled)

# rodando o Kmeans com 2 clusters (pulsar e nao pulsar)
kmeans = KMeans(n_clusters=2, random_state=0)
dataset_scaled = preprocessing.scale(dataset.iloc[:,0:8])
clusters = kmeans.fit_predict(dataset_scaled)

#Plotando o grafico dos Pulsares e Nao Pulsares Linearmente separados##
#AVISO-Caso o kmeans troque o 0 com 1 basta rodar o codigo de novo, 
#esse erro ocorre por conta do enorme numero de não pulsares(0) em 
#relacao aos pulsares(1)

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
```
![3](https://user-images.githubusercontent.com/50224653/86123766-6cc3c080-bab0-11ea-9d2a-911b4dcaaeba.png)

```Python
#Resultados###########################################################
print (classification_report(dataset.iloc[:,-1], clusters))
cmtx = pd.DataFrame(
    confusion_matrix(dataset.iloc[:,-1], clusters), 
    index=['Real:NPulsar', 'Real:Pulsar'], 
    columns=['Predito:NPulsar', 'Pretido:Pulsar']
)
print("Matriz_de_Confusão")
print(cmtx)

#Observa-se que o Kmeans consequiu de forma eficiente
#separar linearmente os dados e fazer uma boa classificação
#de pulsares e não pulsares

              precision    recall  f1-score   support

           0       0.98      0.95      0.96     16259
           1       0.62      0.78      0.69      1639

    accuracy                           0.94     17898
   macro avg       0.80      0.87      0.83     17898
weighted avg       0.94      0.94      0.94     17898

#Matriz_de_Confusão
              Predito:NPulsar  Pretido:Pulsar
Real:NPulsar            15481             778
Real:Pulsar               358            1281

```
