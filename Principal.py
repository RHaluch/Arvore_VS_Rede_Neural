from sklearn import neural_network, metrics, datasets, tree
import numpy as np

db = datasets.load_digits()

X = db.data
Y = db.target

np.random.seed(0)

n_samples = len(X)
divisao = 0.75

ordem = np.random.permutation(n_samples)

X = X[ordem]
Y = Y[ordem]

X_teste = X[int(divisao*n_samples):]
Y_teste = Y[int(divisao*n_samples):]

X_treino = X[:int(divisao*n_samples)]
Y_treino = Y[:int(divisao*n_samples)]

#classificador ARVORE DE DECISAO --------------------------------
'''clf = tree.DecisionTreeClassifier()
clf.fit(X_treino,Y_treino)

predicao = clf.predict(X_teste)

print(clf.score(X_teste,Y_teste))

matriz=metrics.confusion_matrix(Y_teste, predicao)

for item in matriz:

    print(item)'''

#classificador REDEN NEURAL MLP ---------------------------------
clf = neural_network.MLPClassifier(activation='logistic')

clf.fit(X_treino,Y_treino)

predicao = clf.predict(X_teste)

print(clf.score(X_teste,Y_teste)*100," %")

matriz = metrics.confusion_matrix(Y_teste,predicao)

for linha in matriz:
    print(linha)
