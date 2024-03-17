import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mh

################################################
# NAZWY KOLUMN
quantity1 = "cisnienie"
quantity2 = "opor"
# ŻĄDANY STOPIEŃ WIELOMIANU
deg = 4
# DOKŁADNOŚĆ WYKRESU
acc = 1000
#===============================================


# funkcje
# temperatura od oporu
A = 3.9083e-3
B = -5.775e-7
C = 4.183e-12
R0 = 100

# wielomian
def pol(fit, x, y):
    for i, val in enumerate(fit):
        y += val*x**(deg-i)
    return y

# pochodna wielomianu
def der(fit, x, y):
    for i, val in enumerate(fit):
        y += (deg-i)*val*x**(deg-i-1)
    return y

# Numeryczne rozwiązywanie wielomianu R(t)
def tem(R):
    roots = np.roots([R0*C, 100*R0*C, R0*B, R0*A, R0-R])
    root = 0.
    for i in roots:
        if i > -220 and i < 0:
            root = i
    return root.real
#print(tem(24))

# Ciepło w punkcie B
def heat(U, I, t1, t2, h1, h2, d):
    return 4*U*I*t1*t2/rho/mh.pi/d**2/(t1*h2-t2*h1)


# pobieranie danych
excel = pd.read_excel('BYKz04c113v0.xlsx')
#print(data)


# wartości tablicowe
# gestosc ciekłego azotu
rho = 808


# punkt A

# ciśnienie atmosferyczne
P0 = 1.020
DP0 = 0.001
DPM = ( (1.6*0.25/100)**2 + (0.005)**2 )**0.5
DP = ( DP0**2 + DPM**2 )**0.5
print("Niepewność ciśnienia: " + str(DP) + " bar")
data = pd.DataFrame(excel[0:40][[quantity1, quantity2]])
x = np.array(data[:][quantity1])
x = x + P0
y = np.array(data[:][quantity2])
# konwersja oporu na temperaturę
for i, val in enumerate(y):
    y[i] = tem(val)
    y[i] = y[i] + 273.15
# drukowanie wyników dla punktów pomiarówych
print("Ciśnienie, Temperatura")
for i,j in zip(x, y):
    print(str(str(round(i, 2))) + "   " + str(round(j, 2)))

# fittowanie wielomianu
fit = np.polyfit(x, y, deg)
for i, val in enumerate(fit):
    print("w" + str(deg-i) + " = " + str(val))

xpol = np.linspace(x[0], x[-1], acc)
ypol = np.zeros(acc)
ypol = pol(fit, xpol, ypol)

# pochodna w punkcie p = 1
yP0 = 0
derP0 = der(fit, P0, yP0)
derP0si = derP0/100000 # konwersja na si
print("(dP/dT)_P0 = " + str(round(derP0si, 7)) + "K/Pa")
# wspóczynnik stycznej
b = pol(fit, P0, 0) - derP0*P0
# styczna
yder = derP0*xpol + b

# gestosc azotu
# stala gazowa
R = 8.314
m1mol = 0.026
    # ciekly - podane wyzej jako rho
T0 = pol(fit, P0, 0)
# gestosc azotu ciekłego
rho1 = rho
# gestosc azotu gazowego
rho2 = m1mol*P0*100000/R/T0 #konwersja P0 na si
print("Gęstość azotu gazowego: " + str(rho2) + "kg/m^3")

#derP0 = 1.62
Qp = T0*(1/rho2 + 1/rho1)/derP0si
print("Ciepło parowania A: " + str(round(Qp, 0)))   

# temperatura punktu potrójnego
Tp = tem(14.26) + 273.15
print("Temperatura punktu potrójnego: " + str(round(Tp)) + "K")

plt.scatter(x, y, s=15)
plt.plot(xpol, ypol)
plt.plot(xpol, yder)
plt.errorbar(x, y, xerr=DP, capsize=3, ls = 'none', color = "black")
plt.xlabel("Cisnienie [bar]")
plt.ylabel("Temperatura [K]")


# punkt B
# dane
U = 20.
DU = 2.
I = 0.205
DI = 0.007
# z grzałką
tab_t1 = np.array([188.62, 213.10, 217.77])
h1 = 0.04
# bez grzałki
tab_t2 = np.array([117.31, 103.64, 110.99])
h2 = 0.04
Dh = 0.005
# średnica kriostatu [w cm]
tab_d = np.array([1.465, 1.465, 1.470])
tab_d = tab_d / 100

# analiza
t1 = np.mean(tab_t1)
t2 = np.mean(tab_t2)
Dt = 15
d = np.mean(tab_d)
Dd = 0.000005

Q = heat(U, I, t1, t2, h1, h2, d)
print("Ciepło parowania B: " + str(round(Q, 2)))


# pochodne od niepewności
def dQdU():
    return (  4*I*t1*t2/rho/mh.pi/d**2/(t1*h2-t2*h1)  )*DU

def dQdI():
    return (  4*U*t1*t2/rho/mh.pi/d**2/(t1*h2-t2*h1)  )*DI

def dQdh():
    return (  ( rho*mh.pi*d**2*h2 * 4*U*I*t1*t2 )/ (rho*mh.pi*d**2*(t1*h2-t2*h1))**2  )*Dh

def dQdt1():
    return ( ( 4*U*I*t2*rho*mh.pi*d**2*(t1*h2 - t2*h1) - rho*mh.pi*d**2*h2*4*U*I*t1*t2 ) / (rho*mh.pi*d**2*(t1*h2-t2*h1))**2  )*Dt

def dQdt2():
    return ( ( 4*U*I*t1*rho*mh.pi*d**2*(t1*h2 - t2*h1) + rho*mh.pi*d**2*h1*4*U*I*t1*t2 ) / (rho*mh.pi*d**2*(t1*h2-t2*h1))**2  )*Dt

def dQdd():
    return (  2*rho*mh.pi*d*(t1*h2-t2*h1)*4*U*I*t1*t2 / (rho*mh.pi*d**2*(t1*h2-t2*h1))  )*Dd

print("Przyczynki do niepewności")
print("dQ/dU = " + str(round(dQdU() , 5)))
print("dQ/dI = " + str(round(dQdI() , 5)))
print("dQ/dh = " + str(round(dQdh() , 5)))
print("dQ/dt1 = " + str(round(dQdt1() , 5)))
print("dQ/dt2 = " + str(round(dQdt2() , 5)))
print("dQ/dd = " + str(round(dQdd() , 5)))

u = ( (dQdU())**2 + (dQdI())**2 + 2*(dQdh())**2 + (dQdt1())**2 + (dQdt2())**2 + (dQdd())**2 )**0.5

print("Niepewnośc Q = " + str(round(u , 5)))

plt.show()
