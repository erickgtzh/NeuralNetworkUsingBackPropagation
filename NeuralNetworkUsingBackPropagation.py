# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

class RedNeuronal(object):
    def __init__(self):
        ##La primera capa de nuestra red neuronal
        self.numeroNeuronasEntrada = 4
        ##La segunda capa de nuestra red neuronal, que llamarémos escondida1
        self.numeroNeuronasEscondidas1 = 6
        ##La tercera capa de nuestra red neuronal, que llamarémos escondida2
        self.numeroNeuronasEscondidas2 = 6
        ##La salida de nuestra red neuronal
        self.numeroNeuronasSalida = 1

        ##Creación de w1 por medio de matriz del tamaño de neuronas de entrada y las neuronas escondida1, de números flotantes del 0 al 1
        self.W1 = np.random.rand(self.numeroNeuronasEntrada,self.numeroNeuronasEscondidas1)    
        ##Creación de w2 por medio de matriz del tamaño de neuronas escondidas1 y las neuronas escondidas2, de números flotantes del 0 al 1
        self.W2 = np.random.rand(self.numeroNeuronasEscondidas1,self.numeroNeuronasEscondidas2)
        ##Creación de w2 por medio de matriz del tamaño de neuronas escondidas2 y la neurona de salida, de números flotantes del 0 al 1
        self.W3 = np.random.rand(self.numeroNeuronasEscondidas2,self.numeroNeuronasSalida)
        
    def propagation(self,x):
        ##Se crean las variables
            ##Z2, A2, Z3, A3, Z4, Y*
        #A nos permite la activación de su respectiva Z
        self.Z2 = np.dot(x,self.W1)
        self.A2 = self.sigmoide(self.Z2)
        self.Z3 = np.dot(self.A2,self.W2)
        self.A3 = self.sigmoide(self.Z3)
        self.Z4 = np.dot(self.A3,self.W3)
        Ygorrito = self.sigmoide(self.Z4)
        return Ygorrito
    
    def sigmoide(self,z):
        return 1/( (1+np.exp(-z))**2 )
    
    def derivadaSigmoide(self,z):
        return np.exp(-z)/( (1+np.exp(-z))**2 )
    
    def funcionDeCosto(self,x,y):
        self.Ygorrito = self.propagation(x)
        J = 0.5*sum( (y-self.Ygorrito)**2 )
        return J
    
    def derivadafuncionDeCosto(self,x,y):
        self.Ygorrito = self.propagation(x)
        E = y-self.Ygorrito
        delta3 = np.multiply(-(E),self.derivadaSigmoide(self.Z4))
        djdw3 = np.dot( np.transpose(self.A3),delta3)
        delta2 = np.dot(delta3,np.transpose(self.W3)) * self.derivadaSigmoide(self.Z3)
        djdw2 = np.dot(np.transpose(self.A2),delta2)
        delta1 = np.dot(delta2,np.transpose(self.W2))*self.derivadaSigmoide(self.Z2)
        djdw1 = np.dot(np.transpose(x),delta1)
        return djdw1,djdw2,djdw3
        
    def obtenerParametros(self):
        W1_vector = self.W1.ravel()
        W2_vector = self.W2.ravel()
        W3_vector = self.W3.ravel()
        parametros = np.concatenate( (W1_vector,W2_vector,W3_vector) )
        return parametros
        
    def setearParametros(self,parametros):
        W1_start = 0
        W1_end = self.numeroNeuronasEntrada*self.numeroNeuronasEscondidas1
    
        W2_start = W1_end
        W2_end = W1_end+self.numeroNeuronasEscondidas1*self.numeroNeuronasEscondidas2
        
        W3_start = W2_end
        W3_end = W2_end+self.numeroNeuronasEscondidas2*self.numeroNeuronasSalida
        
        self.W1 = np.reshape( parametros[W1_start:W1_end], (self.numeroNeuronasEntrada,self.numeroNeuronasEscondidas1) )
        self.W2 = np.reshape( parametros[W2_start:W2_end], (self.numeroNeuronasEscondidas1,self.numeroNeuronasEscondidas2) )
        self.W3 = np.reshape( parametros[W3_start:W3_end], (self.numeroNeuronasEscondidas2,self.numeroNeuronasSalida) )
        
    def calcularGradienteconBackPropagation(self,x,y):
        djdw1,djdw2,djdw3 = self.derivadafuncionDeCosto(x,y)
        djdw1_vector = djdw1.ravel()
        djdw2_vector = djdw2.ravel()
        djdw3_vector = djdw3.ravel()
        vector_derivadas = np.concatenate( (djdw1_vector,djdw2_vector,djdw3_vector) )
        return vector_derivadas
    
class Entrenador(object):
    def __init__(self,N):
        self.N = N
        
    def contenedorPesos(self,params,X,y):
        self.N.setearParametros(params)
        cost=self.N.funcionDeCosto(X,y)
        grad=self.N.calcularGradienteconBackPropagation(X,y)
        return cost,grad
        
    def llamadaVuelta(self,params):
        self.N.setearParametros(params)
        self.J.append(self.N.funcionDeCosto(self.X,self.y))
        
    def entrenar(self,x,y):
        self.X = x
        self.y = y
        params0=self.N.obtenerParametros()
        self.J=[]
        _res = minimize(self.contenedorPesos,params0,(x,y),method="BFGS",jac=True,tol=None,callback=self.llamadaVuelta,options={'maxiter':100, 'disp':True})
        self.N.setearParametros(_res.x)
        plt.plot(self.J)
redNeuronal = RedNeuronal()
#print("W1",redNeuronal.W1)
#print("W2",redNeuronal.W2)
#print("W3",redNeuronal.W3)
X = np.array( ([6,5,3,2],[5,7,2,4],[8,5,1,5],[7,8,2,5]),dtype=float )
resultados = np.array( ([82],[74],[85],[79]),dtype=float )

print("Entradas")
print(X)
print("Resultados a los que tenemos que llegar")
print(resultados)

X=X/24
resultados=resultados/100
#print("Entrada:",X)
#print(resultados)
print("Propagation:")
print(redNeuronal.propagation(X))

print(redNeuronal.funcionDeCosto(X,resultados))

a,b,c = redNeuronal.derivadafuncionDeCosto(X,resultados)
#print(a,b,c)

print("Derivadas con Backpropagation")
#print(redNeuronal.obtenerParametros())
print(redNeuronal.calcularGradienteconBackPropagation(X,resultados))

#print("Resultado back")
#print(redNeuronal.propagation(X))

entrenador = Entrenador(redNeuronal)
entrenador.entrenar(X,resultados)

print("Resultados con entrenamiento")
print(redNeuronal.propagation(X))