import datasets

X, Y = datasets.load_linear_example1()
print(X)
print(X[0])
print(Y)


import regression

model = regression.LinearRegression()
#model.x, model.y = X, Y

import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X, Y)
print(model.theta)

print(model.score(X, Y))

#print(model.x)
print("ﾎﾞｸﾊﾎｹﾞﾀﾛｳ!")