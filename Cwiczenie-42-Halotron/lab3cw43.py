import math as mh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

#latex na wykresach
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Roman'

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
z3 = (np.arange(0., 3., 0.5) + 3.5) * centy # oś pozioma eksperymentu, wartości dla wierszy [m]
y3 = np.arange(0., 2.5, 0.5) * centy # oś pionowa eksperymentu, wartości dla kolumn [m]
Ih3 = -Ih2 # prąd był ujemny bo bez pola magnetycznego napięcie było ujemne
# drukowanie serii pomiarowych
print("Napiecie 1:\n" + str(napiecie1))
print("Napiecie 2:\n" + str(napiecie2))
print("Napiecie 3:\n" + str(napiecie3))
# wartości dla kolumn i wierszy
#print("\nIc1: " + str(Ic1) + "\nIh1: " + str(Ih1)  + "\nz2: " + str(z2)  + "\ny2: " + str(y2)  + "\nz3: " + str(z3)  + "\ny3: " + str(y3))

# WSZELNIE STAŁE
N = 40
d1 = np.array([94, 96, 95.5])*mili
d2 = np.array([120.4, 120.7, 120.25])*mili
Dd = 0.05 * mili
d = (np.mean(d2) + np.mean(d1))/2
r = d/2
mu0 = 12.5664e-7
c = 0 # stała halotronu - będzie obliczona
R = 0 # opór odcinka halotronu - będzie obliczony

# WSZELKIE NIEPEWNOŚCI
DUh = 0.1 * mili
DIc = 0.2
DIh = (20 * mili * 0.008 + 0.01 * mili)
Dr = 3 * mili # dokładność działek miarki [m]
Dc = 0 # będzie obliczone
DR = 0 # będzie obliczone
print("\nNiepewność pomiaru napięcia Halla: " + str(DUh/mili) + "mV")
print("Niepewność pomiaru prądu cewki: " + str(DIc) + "A")
print("Niepewność pomiaru prądu halotronu: " + str(DIh/mili) + "mA")
print('Średinca cewki: ' + str(round(d, 6)) + ' +/- ' + str(round(Dd, 6)))

# OBRÓBKA DANYCH
# SERIA 1 - zaleznosc napiecia halla od natezen
print("\n### PIERWSZA SERIA")
# wagi punktów
plt.figure()
plt.subplot(121)
print('T_h[mA], a u(a)[mV*s/C], b u(b)[mV], c u(c)[m^2/C], R u(R)[mOhm]:')
for i, val in enumerate(napiecie1):
    # liczenie regresji
    Ic1const = sm.add_constant(Ic1) # intercept nie na zero
    model = sm.WLS(napiecie1[i][:], Ic1const, weights=1)
    paramsModel = model.fit() # dopasowywanie współczynników
    regress = paramsModel.params.flatten() # robienie zwykłej tablicy z tablicy zagnieżonej
    Dregress = paramsModel.bse.flatten()
    # współczynniki oraz ich niepewności
    slope = regress[1] # współcz. nachylenia
    const = regress[0] # stała addytywna
    Dslope = Dregress[1] # niepewność współczynnika
    Dconst = Dregress[0]
    # stała cewki i jej opór
    c = 2/mu0/N * slope*r/Ih1[i]
    R = const/Ih1[i]
        # niepewności
    dcdslope = 2/mu0/N * r/Ih1[i]
    dcdIh = 2/mu0/N * slope*r/Ih1[i]**2
    Dc = ( (dcdslope * Dslope)**2 + (dcdIh * DIh)**2 )**0.5
    DR = const/Ih1[i]**2 * DIh
    
    # wyniki na ekran
    print(str(Ih1[i]/mili) + ', ', end = '')
    print(str(round(slope/mili, 3)) + " " + str(round(Dslope/mili, 4)) + ', ', end = '')
    print(str(round(const/mili, 3)) + " " + str(round(Dconst/mili, 4)) + ', ', end = '')
    print(str(round(c , 0)) + " " + str(round(Dc , 1)) + ', ', end = '')
    print(str(round(R/mili , 0)) + " " + str(round(DR/mili , 1)))
    if i == np.size(napiecie1[: , 0]) - 1:
        plt.legend(title = "Napiecie halotronu", fontsize = 16)
        plt.xlabel("Natezenie pradu cewki [A]", fontsize = 16)
        plt.ylabel("Napiecie Halla [mV]", fontsize = 16)
        plt.subplot(122)
    # wykres
    plt.errorbar(Ic1, napiecie1[i][:]/mili, xerr=DIc, yerr=DUh/mili, capsize=3, ls = 'none', color = "gray",  elinewidth=0.8)
    plt.scatter(Ic1, napiecie1[i][:]/mili)
    Ic1lin = np.linspace(min(Ic1), max(Ic1), 2)
    plt.plot(Ic1lin, (slope * Ic1lin + const)/mili, label = str(Ih1[i]/mili) + "mA")
# rysowanie wykresów dla regresji
plt.legend(title = "Napiecie halotronu", fontsize = 16)
plt.xlabel("Natezenie pradu cewki [A]", fontsize = 16)
plt.ylabel("Napiecie Halla [mV]", fontsize = 16)
# żeby R było dodatnie
R = abs(R)


#SERIA 2 - modelowanie pola magnetycznego w otoczeniu cewki
print("\n 2. SERIA POMIAROWA:")
# wykres 3D 
B2 = 1/c*(napiecie2/Ih2 - R)
z, y = np.meshgrid(y2/centy, z2/centy)
# tworzenie wykresu 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
# Tworzenie powierzchni
surf = ax.plot_surface(z, y, B2/mili, cmap='viridis')
# Dodanie paska kolorów
fig.colorbar(surf, shrink = 0.6)
# orientacja wykresu
ax.view_init(elev=40, azim=45)
# Oznaczenie osi
ax.set_xlabel('y[cm] (oś pionowa)', fontsize = 20)
ax.set_ylabel('z[cm] (oś pozioma)', fontsize = 20)
ax.set_zlabel('Składowa z-owa indukcji magnetycznej [mT]', fontsize = 20)
ax.set_title("Cewka", fontsize = 40)

# liczenie indukcji na osi z
# napięcia uśrednione na osi z
napiecie2srodek = (napiecie2[:, 4] + napiecie2[:, 4])/2
napiecie2naosi = (napiecie2[:, 2])
# indukcja magnetyczna teoretyczna
z2teor = np.linspace(np.min(z2), np.max(z2), 1000)
B2teor = mu0*N/2 * Ic2 /r/(1 + (z2teor)**2/r**2)**(3/2)
B2test = mu0*N/2 * Ic2 /r/(1 + (z2)**2/r**2)**(3/2)
# indukcja magnetyczna doswiadczalna i jej niepewności
    # przyjęte do zbadanie
B2naosi = 1/c*(napiecie2naosi/Ih2 - R)
dBdc = 1/c**2*(napiecie2naosi/Ih2 - R)
dBdUh = 1/c/Ih2
dBdIh = napiecie2naosi/c/Ih2**2
dBdR = 1/c
DB2naosi = ( ( dBdc*Dc )**2 + ( dBdUh*DUh )**2 +  ( dBdIh*DIh )**2 +  ( dBdR*DR )**2 )**0.5
    # faktycznie na środku
B2srodek = 1/c*(napiecie2srodek/Ih2 - R)
dBdc = 1/c**2*(napiecie2srodek/Ih2 - R)
dBdUh = 1/c/Ih2
dBdIh = napiecie2srodek/c/Ih2**2
dBdR = 1/c
DB2srodek = ( ( dBdc*Dc )**2 + ( dBdUh*DUh )**2 +  ( dBdIh*DIh )**2 +  ( dBdR*DR )**2 )**0.5
    # printy
print('z[cm], U_śr[mV], B_śr[mT], u(B_śr[mT]), U_oś[mv], B_oś[mT], u(B_oś[mT]), B_teor[mT]')
for i, j, k, l, m, n, o, p in zip(z2/centy, napiecie2srodek/mili, B2srodek/mili, DB2srodek/mili, napiecie2naosi/mili, B2naosi/mili, DB2naosi/mili, B2test/mili):
    print(str(round(i, 2)) + ', ' + str(round(j, 2)) + ', ' + str(round(k, 2)) + ', ' + str(round(l, 2)) + ', ' + str(round(m, 2)) + ', ' + str(round(n, 2)) + ', ' + str(round(o, 2)) + ', ' + str(round(p, 2)))


# wykresy
plt.figure()
plt.errorbar(z2/centy, B2naosi/mili, xerr=Dr/centy, yerr=DB2naosi/mili, capsize=3, ls = 'none', color = "magenta",  elinewidth=0.8)
plt.scatter(z2/centy, B2naosi/mili, color = 'purple')
plt.plot(z2teor/centy, B2teor/mili, color = 'red')
plt.xlabel("Pozycja na osi cewki [cm]", fontsize = 16)
plt.ylabel("Składowa z-owa indukcji magnetycznej [mT]", fontsize = 16)
# prezentacja wyników w tabeli
print("z[cm]:  B[mT]:  u(B)[mT]")
for i,j,k in zip(z2, B2naosi, DB2naosi):
    print(str(round(i/centy, 5)) + " " + str(round(j/mili, 2)) + " " + str(round(k/mili + 0.005 - 1e-10, 2)) )
print('\nZMIERZONA INDUKCJA POLA\nz\\x', end = '')
for i in range(np.size(B2[0, :])):
    print(str(i*0.5-2.25) + ' ', end = '')
for k, i in enumerate(B2):
    print('\n' + str(k*0.5), end = ' ')
    for j in i:
        print(str(round(j/mili, 2)) + ' ', end = '')


#SERIA 3 - modelowanie pola magnetycznego w otoczeniu magnesu
print("\n 3. SERIA POMIAROWA:")
B3_3D = 1/c*(napiecie3/Ih3 - R)*(-1)
# wykres 3D 
z, y = np.meshgrid(y3/centy, z3/centy)
# tworzenie wykresu 3D
ax = fig.add_subplot(122, projection='3d')
# Tworzenie powierzchni
surf = ax.plot_surface(z, y, B3_3D/mili, cmap='viridis')
# Dodanie paska kolorów
fig.colorbar(surf, shrink = 0.6)
# orientacja wykresu
ax.view_init(elev=40, azim=45)
# Oznaczenie osi
ax.set_xlabel('y[cm] (oś pionowa)', fontsize = 20)
ax.set_ylabel('z[cm] (oś pozioma)', fontsize = 20)
ax.set_zlabel('Składowa z-owa indukcji magnetycznej [mT]', fontsize = 20)
ax.set_title("Magnes", fontsize = 40)
# liczenie indukcji na osi z
# napięcia uśrednione na osi z
napiecie3srodek = napiecie3[:, 0]
# indukcja magnetycznai jej niepewności
# wyznaczanie momentu dipolowego mu na podstawie 1. punktu
mu = abs( 1/c * (napiecie3srodek[0]/Ih3 - R) * 2*mh.pi*z3[0]**3/mu0 )
dmudc = 1/c**2 * (napiecie3srodek[0]/Ih3 - R) * 2*mh.pi*z3[0]**3/mu0
dmudU = 1/c/Ih3 * 2*mh.pi*z3[0]**3/mu0
dmudIh = napiecie3srodek[0]/c/Ih3**2 * 2*mh.pi*z3[0]**3/mu0
dmuDR = 1/c * 2*mh.pi*z3[0]**3/mu0
dmudz = 1/c * 2*mh.pi*z3[0]**3/mu0
Dmu = ( ( dmudc*Dc )**2 + ( dmudU*DUh )**2 + ( dmudIh*DIh )**2 + ( dmuDR*DR )**2 + ( dmudz*Dr )**2 )**0.5
# punkty pomiarowe i niepewności (niep. wg takiej samej formuły jak dla punktu 2)
B3 = np.abs( 1/c * (napiecie3srodek/Ih3 - R) )
DB3 = np.zeros(np.size(B3))
dBdc = 1/c**2*(napiecie3srodek/Ih3 - R)
dBdUh = 1/c/Ih3
dBdIh = napiecie3srodek/c/Ih3**2
dBdR = 1/c
DB3 = ( ( dBdc*Dc )**2 + ( dBdUh*DUh )**2 +  ( dBdIh*DIh )**2 +  ( dBdR*DR )**2 )**0.5
# krzywa teoretyczna
z3teor = np.linspace(np.min(z3), np.max(z3), 1000)
B3teor = mu0*mu/2/mh.pi/z3teor**3
B3test =  mu0*mu/2/mh.pi/z3**3
print("Moment dipolowy magnesu: " + str(round(mu, 4)) + "A*s")
print("Niepewność omentu dipolowego magnesu: " + str(round(Dmu, 4)) + "A*s")
# wykresy
plt.figure()
plt.plot(z3teor/centy, B3teor/mili, color = 'red')
plt.errorbar(z3/centy, B3/mili, xerr=Dr/centy, yerr=DB3/mili, capsize=3, ls = 'none', color = "magenta",  elinewidth=0.8)
plt.scatter(z3/centy, B3/mili, color = 'purple')
plt.xlabel("Pozycja na osi biegunów magnesu [cm]", fontsize = 16)
plt.ylabel("Składowa z-owa indukcji magnetycznej [mT]", fontsize = 16)
# prezentacja wyników w tabeli
print("z[cm]:  B[mT]:  u(B)[mT]: B_teor[mT]")
for i,j,k,l in zip(z3, B3, DB3, B3test):
    print(str(round(i/centy, 5)) + " " + str(round(j/mili, 2)) + " " + str(round(k/mili + 0.005 - 1e-10, 2)) + " " + str(round(l/mili + 0.005 - 1e-10, 2)) )
    

# badanie jakości dopasowania
pearsonCewka = pearsonr(B2naosi, B2test)[0]
pearsonMagnes = pearsonr(B3, B3test)[0]
# tablice reszt unormowanych
B2rest = (B2naosi - B2test)/DB2srodek
B3rest = (B3 - B3test)/DB3

print('\nWspółczynnik korelacji Pearsona dla cewki: ' + str(round(pearsonCewka, 7)))
print('\nŚrednia reszt unormowanych cewka: ' + str(round(np.mean(B2rest), 4)))
print('Odchylnie std reszt unormowanych cewka: ' + str(round(np.std(B2rest), 4)))

print('Współczynnik korelacji Pearsona dla magnesu: ' + str(round(pearsonMagnes, 7)))
print('\nŚrednia reszt unormowanych magnes: ' + str(round(np.mean(B3rest), 4)))
print('Odchylnie std reszt unormowanych magnes: ' + str(round(np.std(B3rest), 4)))

    # wykresy
plt.figure()
plt.subplot(121)
plt.scatter(z2/centy, B2rest, color = 'purple', marker = 'D')
plt.plot(z2/centy, B2rest, color = 'red')
plt.plot([np.min(z2/centy), np.max(z2/centy)], [0,0], color = 'black', linestyle = '--')
plt.plot([np.min(z2/centy), np.max(z2/centy)], [np.mean(B2rest), np.mean(B2rest)], color = 'magenta')
plt.ylabel(r'Reszty unormowane $\frac{B_i - B(x_i)}{u(B_i)}$', fontsize = 20)
plt.xlabel(r'Odległość od płaszczyzny cewki $z[cm]$', fontsize = 20)
plt.title('Cewka', fontsize = 40)
plt.subplot(122)
plt.scatter(z3/centy, B3rest, color = 'purple', marker = 'D')
plt.plot(z3/centy, B3rest, color = 'red')
plt.plot([np.min(z3/centy), np.max(z3/centy)], [0,0], color = 'black', linestyle = '--')
plt.plot([np.min(z3/centy), np.max(z3/centy)], [np.mean(B3rest), np.mean(B3rest)], color = 'magenta')
plt.ylabel(r'Reszty unormowane $\frac{B_i - B(x_i)}{u(B_i)}$', fontsize = 20)
plt.xlabel(r'Odległość od magnesu na osi jego biegunów $z[cm]$', fontsize = 20)
plt.title('Magnes', fontsize = 40)

plt.show()
    


