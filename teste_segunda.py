import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from pickle import dump

#
dados = pd.read_csv ('C:/Users/muril/OneDrive/Área de Trabalho/Teste segunda/breast-cancer.csv', sep=',')
print('Frequencia de class (atributos)')
print(dados.Class.value_counts())

#Balancear os dados 
dados.classes = dados ['Class']#somente a coluna class
dados.atributos = dados.drop(columns=['Class'])# Todas as colunas exceto class

#Normalizando os dados 
dados.atributos_normalizados = pd.get_dummies(dados.atributos)
rotulos_normalizados = dados.atributos_normalizados.columns #preservando os rotulos da colunas normalizadas para usar depois
print(dados.atributos.head)
print(dados.atributos_normalizados.head)

#construção de um objeto a partir do smote a executar o metodo fit_sampler
resampler = SMOTE()

dados.atributos_b, dados.classe_b = resampler.fit_resample(dados.atributos_normalizados,dados.classes)

print('## FERQUENCIA DAS CLASSES APÓS O BALANCEAMENTO ##')
class_count = Counter(dados.classe_b)
print(class_count)

#treinando e usando RandomForest
rf = RandomForestClassifier()
scoring =['precision_macro', 'recall_macro']
scores_cross = cross_validate(rf,dados.atributos_b,dados.classe_b, cv=10, scoring = scoring)
print('Matriz de sensibilidade:', scores_cross['test_precision_macro'])
print('Matriz de especificidade', scores_cross['test_recall_macro'])

#metricas finais: calcular medidas
print('Especifidade:', scores_cross['test_precision_macro'].mean())
print('Sensibilidade:', scores_cross['test_recall_macro'].mean())

#treinando modelo definitivo 
breast_cancer_rf = rf.fit(dados.atributos_b, dados.classe_b)
dump(breast_cancer_rf, open('C:/Users/muril/OneDrive/Área de Trabalho/Teste segunda/breast-cancer_rf.pkl', 'wb'))



