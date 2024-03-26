import math as mh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D

# PRZEDROSTKI
mili = 1e-3
centy = 1e-2
micro = 1e-6

# WCZYTYWANIE DANYCH
# dane z excela
excel1 = pd.read_excel('BYKz04c43v0.xlsx', sheet_name='Tabela1')
excel2 = pd.read_excel('BYKz04c43v0.xlsx', sheet_name='Arkusz2')
excel3 = pd.read_excel('BYKz04c43v0.xlsx', sheet_name='Arkusz3')
# drukowanie danych
'''print("\nDANE Z EXCELA:")
print(excel1)
print(excel2)
print(excel3)'''
#
# serie pomiarowe
print("\n 1. SERIA POMIAROWA:")
# 1. seria
napiecie1 = excel1.iloc[1:5+1, 1:11+1].values * mili # npaięcia halotronu [V]
Ic1 = np.arange(0., 11.) # prąd cewki I_c [A]
Ih1 = np.array([3.5, 5., 7., -7.]) * mili # prąd halotronu [A]
# 2. seria
napiecie2 = excel2.iloc[0:18+1, 2:11+1].values * mili
z2 = np.arange(0, 9.5, 0.5) * centy # oś pozioma eksperymentu, wartości dla wierszy [m]
y2 = np.arange(-2.5, 2.5, 0.5) * centy # oś pionowa eksperymentu, wartości dla kolumn [m]
y2 = y2 + 0.25 * centy # korekta, żeby pozycje były wyśrodkowane
Ic2 = 10 # prąd cewki podczas tego eksp. - maksymalny 10 A
Ih2 = 7 * mili # prąd halotronu [mA]
# 3. seria
napiecie3 = excel3.iloc[0:5+1, 1:5+1].values * mili
z3 = (np.arange(0., 3., 0.5) + 4) * centy # oś pozioma eksperymentu, wartości dla wierszy [m]
y3 = np.arange(0., 2.5, 0.5) * centy # oś pionowa eksperymentu, wartości dla kolumn [m]
Ih3 = Ih2
# drukowanie serii pomiarowych
print("Napiecie 1:\n" + str(napiecie1))
print("Napiecie 2:\n" + str(napiecie2))
print("Napiecie 3:\n" + str(napiecie3))
# wartości dla kolumn i wierszy
#print("\nIc1: " + str(Ic1) + "\nIh1: " + str(Ih1)  + "\nz2: " + str(z2)  + "\ny2: " + str(y2)  + "\nz3: " + str(z3)  + "\ny3: " + str(y3))

# WSZELNIE STAŁE
N = 40
r = 90/2 * mili
mi0 = 12.5664e-7
c = 0 # stała halotronu - będzie obliczona
R = 0 # opór odcinka halotronu - będzie obliczony

# WSZELKIE NIEPEWNOŚCI
DUh = 0.1 * mili
DIc = 0.2
DIh = (20 * mili * 0.008 + 0.01 * mili)
Dr = 3 * mili # dokładność działek miarki [m]
Dc = 0 # będzie obliczone
dR = 0 # będzie obliczone
print("\nNiepewność pomiaru napięcia Halla: " + str(DUh/mili) + "mV")
print("Niepewność pomiaru prądu cewki: " + str(DIc) + "A")
print("Niepewność pomiaru prądu halotronu: " + str(DIh/mili) + "mA")

# OBRÓBKA DANYCH
# SERIA 1 - zaleznosc napiecia halla od natezen
print("\n### PIERWSZA SERIA")
# wagi punktów
w = 1/( DUh**2 + DIc**2 )
print("Wagi punktów: " + str(w))
plt.figure()
for i, val in enumerate(napiecie1):
    # liczenie regresji
    Ic1const = sm.add_constant(Ic1) # intercept nie na zero
    model = sm.WLS(napiecie1[i][:], Ic1const, weights=0.01)
    paramsModel = model.fit() # dopasowywanie współczynników
    regress = paramsModel.params.flatten() # robienie zwykłej tablicy z tablicy zagnieżonej
    Dregress = paramsModel.bse.flatten()
    # współczynniki oraz ich niepewności
    slope = regress[1] # współcz. nachylenia
    const = regress[0] # stała addytywna
    Dslope = Dregress[1] # niepewność współczynnika
    Dconst = Dregress[0]
    # stała cewki i jej opór
    c = 2/mi0/N * slope*r/Ih1[i]
    R = const/Ih1[i]
        # niepewności
    dcdslope = 2/mi0/N * r/Ih1[i]
    dcdIh = 2/mi0/N * slope*r/Ih1[i]**2
    Dc = ( (dcdslope * Dslope)**2 + (dcdIh * DIh)**2 )**0.5
    DR = const/Ih1[i]**2 * DIh
    
    # wyniki na ekran
    print("\nDLA NATĘŻENIA HALOTRONU " + str(Ih1[i]/mili) + "mA")
    print("Współczynnik nachylenia: " + str(round(slope/mili, 5)) + "mV*s/C\nNiepewność współczynnika nachylenia: " + str(round(Dslope/mili, 5)) + "mV*s/C")
    print("Stała addytywna: " + str(round(const/mili, 5)) + "mV\nNiepewność Stałej addytywnej: " + str(round(Dconst/mili, 5)) + "mV\n#")
    print("Stała cewki: " + str(round(c , 5)) + "m^2/C\nNiepewność stałej cewki: " + str(round(Dc , 5)) + "m^2/C")
    print("Opór halotronu: " + str(round(R , 5)) + "ohm\nNiepewność oporu halotronu: " + str(round(DR , 5)) + "ohm")
    # wykres
    plt.errorbar(Ic1, napiecie1[i][:]/mili, xerr=DIc, yerr=DUh/mili, capsize=3, ls = 'none', color = "gray",  elinewidth=0.8)
    plt.scatter(Ic1, napiecie1[i][:]/mili)
    Ic1lin = np.linspace(min(Ic1), max(Ic1), 2)
    plt.plot(Ic1lin, (slope * Ic1lin + const)/mili, label = str(Ih1[i]/mili) + "mA")
# rysowanie wykresów dla regresji
plt.legend(title = "Napięcie halotronu")
plt.xlabel("Natężenie prądu cewki [A]")
plt.ylabel("Napięcie Halla [mV]")
# żeby R było dodatnie
R = abs(R)


#SERIA 2 - modelowanie pola magnetycznego w otoczeniu cewki
print("\n 2. SERIA POMIAROWA:")
# wykres 3D 
z, y = np.meshgrid(y2/mili/10, z2/mili/10)
# tworzenie wykresu 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
# Tworzenie powierzchni
surf = ax.plot_surface(z, y, napiecie2/mili, cmap='viridis')
# Dodanie paska kolorów
fig.colorbar(surf, shrink = 0.6)
# orientacja wykresu
ax.view_init(elev=40, azim=45)
# Oznaczenie osi
ax.set_xlabel('y[cm] (oś pionowa)')
ax.set_ylabel('z[cm] (oś pozioma)')
ax.set_zlabel('Napięcie Halla [mV]')
ax.set_title("Solenoid")

# liczenie indukcji na osi z
# napięcia uśrednione na osi z
napiecie2srodek = (napiecie2[:, 4] + napiecie2[:, 5])/2
# indukcja magnetyczna teoretyczna
z2teor = np.linspace(np.min(z2), np.max(z2), 1000)
B2teor = mi0*N/2 * Ic2 /r/(1 + z2teor**2/r**2)**(3/2)
# indukcja magnetyczna doswiadczalna i jej niepewności
B2 = 1/c*(napiecie2srodek/Ih2 - R)
dBdc = 1/c**2*(napiecie2srodek/Ih2 - R)
dBdUh = 1/c/Ih2
dBdIh = napiecie2srodek/c/Ih2**2
dBdR = 1/c
DB2 = ( ( dBdc*Dc )**2 + ( dBdUh*DUh )**2 +  ( dBdIh*DIh )**2 +  ( dBdR*DR )**2 )**0.5
# wykresy
plt.figure()
plt.errorbar(z2/mili/10, B2/mili, xerr=Dr/mili/10, yerr=DB2/mili, capsize=3, ls = 'none', color = "gray",  elinewidth=0.8)
plt.scatter(z2/mili/10, B2/mili)
plt. plot(z2teor/mili/10, B2teor/mili)
plt.xlabel("Pozycja na osi cewki [cm]")
plt.ylabel("Składowa z-owa indukcji magnetycznej [mT]")
# prezentacja wyników w tabeli
print("z[cm]:  B[mT]:  u(B)[mT]")
for i,j,k in zip(z2, B2, DB2):
    print(str(round(i/mili/10, 5)) + " " + str(round(j/mili, 2)) + " " + str(round(k/mili + 0.005 - 1e-10, 2)) )


#SERIA 3 - modelowanie pola magnetycznego w otoczeniu magnesu
print("\n 3. SERIA POMIAROWA:")
# wykres 3D 
z, y = np.meshgrid(y3/mili/10, z3/mili/10)
# tworzenie wykresu 3D
ax = fig.add_subplot(122, projection='3d')
# Tworzenie powierzchni
surf = ax.plot_surface(z, y, napiecie3/mili, cmap='viridis')
# Dodanie paska kolorów
fig.colorbar(surf, shrink = 0.6)
# orientacja wykresu
ax.view_init(elev=40, azim=45)
# Oznaczenie osi
ax.set_xlabel('y[cm] (oś pionowa)')
ax.set_ylabel('z[cm] (oś pozioma)')
ax.set_zlabel('Napięcie Halla [mV]')
ax.set_title("Solenoid")
# liczenie indukcji na osi z
# napięcia uśrednione na osi z
napiecie3srodek = napiecie3[:, 0]
# indukcja magnetycznai jej niepewności
B3 = mi0/2/z3**3 * N * Ih3 * r**2
dBdz = 2*mi0*N*Ih3*r**2/z3**4
dBdIh = mi0*N*r**2/2/z3**3
DB3 = ( ( dBdz*Dr )**2 + ( dBdIh*DIh )**2 )**0.5
# wykresy
plt.figure()
plt.errorbar(z3/mili/10, B3/micro, xerr=Dr/mili/10, yerr=DB3/micro, capsize=3, ls = 'none', color = "gray",  elinewidth=0.8)
plt.scatter(z3/mili/10, B3/micro)
plt.xlabel("Pozycja na osi biegunów magnesu [cm]")
plt.ylabel("Składowa z-owa indukcji magnetycznej [μT]")
# prezentacja wyników w tabeli
print("z[cm]:  B[μT]:  u(B)[μT]")
for i,j,k in zip(z3, B3, DB3):
    print(str(round(i/mili/10, 5)) + " " + str(round(j/micro, 5)) + " " + str(round(k/micro + 0.005 - 1e-10, 5)) )




plt.show()
    


