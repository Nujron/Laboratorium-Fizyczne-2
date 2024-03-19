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
degFis = 4
# DOKŁADNOŚĆ WYKRESU
acc = 1000
mult = 100000 # przelicznik bar na Pa
#===============================================

print()
# funkcje
# temperatura od oporu
A = 3.9083e-3
B = -5.775e-7
C = 4.183e-12
R0 = 100

# wielomian
def pol(fit, x, y, deg):
    for i, val in enumerate(fit):
        y += val*x**(deg-i)
    return y

# pochodna wielomianu (tablica współczynników, x-tablica argumentów/argument, y-tablica o wymiarze jak x/dowolna zmienna)
def der(fit, x, y):
    for i, val in enumerate(fit):
        y += (deg-i)*val*x**(deg-i-1)
    return y


# Ciepło w punkcie B
def heat(U, I, t1, t2, h1, h2, d):
    return 4*U*I*t1*t2/rho/mh.pi/d**2/(t1*h2-t2*h1)



# pobieranie danych
excel = pd.read_excel('BYKz04c113v0.xlsx')
#print(data)


# wartości tablicowe
# gestosc ciekłego azotu
rho = 808


# BADANIE DOPASOWANIA WIELOMIANU DO DANYCH Z TABELIKI WFIIS
#tablica oporów
resTab = np.zeros(140)
resTab[0] = 11.
for i in range(len(resTab)):
    if i == 0:
        continue
    resTab[i] = resTab[i-1] + 0.1
# tablica temperatur   
tempTab = np.array([54.8, 55.0, 55.3, 55.5, 55.8, 56.0, 56.3, 56.5, 56.8, 57.0,
57.4, 57.6, 57.9, 58.1, 58.4, 58.6, 58.9, 59.1, 59.4, 59.6,
59.9, 60.1, 60.4, 60.6, 60.9, 61.1, 61.4, 61.6, 61.9, 62.1,
62.4, 62.6, 62.8, 63.1, 63.3, 63.6, 63.8, 64.1, 64.3, 64.5,
64.8, 65.0, 65.3, 65.5, 65.8, 66.0, 66.2, 66.5, 66.7, 67.0,
67.2, 67.4, 67.7, 67.9, 68.2, 68.4, 68.6, 68.9, 69.1, 69.4,
69.6, 69.8, 70.1, 70.3, 70.6, 70.8, 71.0, 71.3, 71.5, 71.7,
72.0, 72.2, 72.5, 72.7, 72.9, 73.2, 73.4, 73.6, 73.9, 74.1,
74.4, 74.6, 74.8, 75.1, 75.3, 75.5, 75.8, 76.0, 76.2, 76.5,
76.7, 77.0, 77.2, 77.4, 77.7, 77.9, 78.1, 78.4, 78.6, 78.8,
79.1, 79.3, 79.5, 79.8, 80.0, 80.2, 80.5, 80.7, 80.9, 81.2,
81.4, 81.6, 81.9, 82.1, 82.4, 82.6, 82.8, 83.1, 83.3, 83.5,
83.8, 84.0, 84.2, 84.5, 84.7, 84.9, 85.2, 85.4, 85.6, 85.9,
86.1, 86.3, 86.6, 86.8, 87.0, 87.3, 87.5, 87.7, 88.0, 88.2])

# zależność wielomianowa
fitFis = np.polyfit(resTab, tempTab, degFis)
xRes = np.linspace(resTab[0], resTab[-1], acc)
yTemp = np.zeros(acc)
yTemp = pol(fitFis, xRes, yTemp, degFis)

#plt.scatter(resTab, tempTab)
#plt.plot(xRes, yTemp)
#plt.show()

tempRes = np.stack([resTab, tempTab])



# punkt A

# ciśnienie atmosferyczne
P0 = 1.020 * mult
DP0 = 0.001 * mult
DPM = ( (1.6*0.25*mult/100)**2 + (0.05*mult)**2 )**0.5
DP = ( DP0**2 + DPM**2 )**0.5
print("Niepewność ciśnienia: " + str(DP) + " Pa")
data = pd.DataFrame(excel[0:40][[quantity1, quantity2]])
x = np.array(data[:][quantity1]) * mult
x = x + P0
y = np.array(data[:][quantity2]) # nie mnoże razy mult, bo to opry w ohmach
# konwersja oporu na temperaturę
y = pol(fitFis, y, np.zeros(y.size), degFis)
# drukowanie wyników dla punktów pomiarówych
x = x / mult
print("Ciśnienie, Temperatura")
for i,j in zip(x, y):
    print(str(round(i, 2)) + "   " + str(round(j, 2)))
x = x * mult

# fittowanie wielomianu
fit0 = np.polyfit(x, y, deg, cov = True)
fit = fit0[0]
fitCov = fit0[1]
for i, val in enumerate(fit):
    print("w" + str(deg-i) + " = " + str(val))

xpol = np.linspace(x[0], x[-1], acc)
ypol = np.zeros(acc)
ypol = pol(fit, xpol, ypol, deg)

# gestosc azotu
# stala gazowa
R = 8.314
m1mol = 0.028
    # ciekly - podane wyzej jako rho
T0 = pol(fit, P0, 0, deg)


# pochodna w punkcie p = 1
derP0 = der(fit, P0, 0)
derP0si = derP0 # konwersja na si
print("(dP/dT)_P0 = " + str(round(derP0si, 9)) + "K*kg/Pa")
# wspóczynnik stycznej
b = pol(fit, P0, 0, deg) - derP0*P0
# styczna
yder = derP0*xpol + b

# niepewnosć pochodnej
    # współczynniki niepewności wielomianów, fitCov - tablica z wariancjami współczynników
# wariancja pochodnej:
Vder = 0
for i in range(deg-1):
    for j in range(deg-1):
        Vder = Vder + (deg-i)*( T0**(deg-i-1) )*(deg-j)*( T0**(deg-j-1) )*fitCov[i][j]
DderP0 = Vder**0.5

print("u( (dP/dT)_P0 ) = " + str(DderP0) + "K*kg/Pa")


# gestosc azotu ciekłego
rho1 = rho

# gestosc azotu gazowego
rho2 = m1mol*P0/R/T0 #konwersja P0 na si
print("Gęstość azotu gazowego: " + str(rho2) + "kg/m^3")

Qp = T0*(1/rho2 + 1/rho1)/derP0si
print("Ciepło parowania A: " + str(round(Qp, 0))) 

DQp = Qp*DderP0/derP0
print("Niepewność ciepła parowania A: " + str(round(DQp, 0)))
print() 

# temperatura punktu potrójnego
Tp = pol(fitFis, 14.26, 0, degFis)
print("Temperatura punktu potrójnego: " + str(round(Tp, 4)) + "K")

plt.figure()
xChart = x / mult
xpolChart = xpol / mult
DPChart = DP / mult
plt.scatter(xChart, y, s=15)
plt.plot(xpolChart, ypol)
plt.plot(xpolChart, yder)
plt.errorbar(xChart, y, xerr=DPChart, capsize=3, ls = 'none', color = "black",  elinewidth=0.8)
plt.xlabel("Cisnienie [bar]")
plt.ylabel("Temperatura [K]")


print()
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
