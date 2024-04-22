import numpy as np
import matplotlib.pyplot as plt
import math as mh
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Roman'

# wszelkie stałe i niepewności
Gamma = 8e-3 # stała ekspozycyjna
Dx = 0.1 # niepewność wysokości [cm]
Dd = 0.005 # niepewność grubości
const = 0.087e-7 # stała związana z przeliczeniem jednostek
rhoCu = 8.96 # [g/cm**3]

print()
# WCZYTYWANIE DANCYH
# czytanie excela
excel1 = pd.read_excel('BYKz04c96v0.xlsx', sheet_name= "tło")
excel2 = pd.read_excel('BYKz04c96v0.xlsx', sheet_name= "moc_odległość")
excel3 = pd.read_excel('BYKz04c96v0.xlsx', sheet_name= "miedź")
#print(excel1)
#print(excel2)
#print(excel3)
# zestawy danych
# tło
background = excel1.iloc[0:10, 0:2].values
backgroundPower = background[:, 1]
backgroundPowerMean = np.mean(background[:, 1])
# osłabienie
power1Dist = excel2.iloc[0:16, 0].values
power1 = excel2.iloc[0:16, 1:6].values
power1Mean0 = np.zeros(np.size(power1[:, 0])) # tablica mocy bez odjęcia tła
for i in range(np.size(power1Mean0)):
    power1Mean0[i] = np.mean(power1[i])
power1Mean = power1Mean0 - backgroundPowerMean # odjęcie tła
Dpower1Mean0 = power1Mean0*0.2 # niepewność w tym zakresie to 20%
# miedz
powerCuThic0 = excel3.iloc[1:12, 0:3].values*0.1 # tablica nieuśrenionych pojedyńczych pomiarów grubosci
powerCuThicMean = np.zeros(np.size(powerCuThic0[:, 0])) # tu są grubości pojedynczych płytek
for i in range(np.size(powerCuThic0[:, 0])):
    powerCuThicMean[i] = np.mean(powerCuThic0[i, :])
powerCuThic = np.array(powerCuThicMean) # to jest suma wszystkich poprzednich grubości
DpowerCuThic = np.zeros(np.size(powerCuThic))
DpowerCuThic[0] = Dd
for i in range(1, np.size(powerCuThic)):
    powerCuThic[i] = powerCuThic[i] + powerCuThic[i-1]
    DpowerCuThic[i] = (i+1)*Dd**2
    DpowerCuThic[i] = mh.sqrt(DpowerCuThic[i])
powerCu = excel3.iloc[1:12, 5:10].values
powerCuMean0 = np.zeros(np.size(powerCu[:, 0]))
for i in range(np.size(powerCuMean0)):
    powerCuMean0[i] = np.mean(powerCu[i, :])
powerCuMean = powerCuMean0 - backgroundPowerMean
DpowerCuMean0 = powerCuMean0*0.2
# wyświetlanie danych   
    # moc tła
print("PRZEDSTAWIENIE DANYCH")
print("TŁO\nNr, Moc[muSv/h]")
for i in background:
    print(str(i[0]) + ", " + str(i[1]))
    # moc od odległości
print("\nMOC(ODLEGŁOŚĆ)")
print("Odległość[cm], Pomiary mocy[muSv/h], Srednia[muSv/h], Niepewność[muSv/h]")
for i in range(np.size(power1Dist)):
    print(str(power1Dist[i]) + ", ", end = "")
    for j, val in enumerate(power1[i]):
        if j != 4:
            print(str(val) + "  ", end="")
        else:
                print(str(val) + ", ", end="")
    print(str(round(power1Mean0[i], 2)) + ', ' + str(round(Dpower1Mean0[i], 3)))
    # osłabienie przez miedź
print("\nMOC(GRUBOŚĆ MIEDZI)")
print('Nr płytki, Pomiary grubości[cm], Średnia grubość[cm]')
for i, vali in enumerate(powerCuThic0):
    print(str(i+1) + ', ', end='')
    for j, valj in enumerate(vali):
        print(str(round(valj, 2)), end = '')
        if j != 2:
            print(' ', end = '')
        else:
            print(', ', end = '')
    print(str(round(powerCuThicMean[i], 2)))
print('\nLiczba płytek, Grubosć płytek[cm], Pomiary mocy[muSv/h], Średnia[muSv/h]')
for i in range(np.size(powerCuMean0)):
    print(str(i+1) + ', ' + str(round(powerCuThic[i], 2)) + ', ', end = '')
    for j, val in enumerate(powerCu[i]):
        print(str(val), end = '')
        if j != 4:
            print(' ', end = '')
        else:
            print(', ', end = '')
    print(str(round(powerCuMean0[i], 2)))
    


# OBRÓBKA DANYCH
# pomiary tła
N = np.size(backgroundPower)
    # estymator odchylenia standardowego
Dbackground = 1/mh.sqrt(N-1)*np.std(backgroundPower)
    # printy
DbackgroundRelative = Dbackground/backgroundPowerMean*100
print("\nANALIZA TŁA")
print("Średnia moc: " + str(round(backgroundPowerMean, 3)) + "muSv/h")
print("Niepewność mocy: " + str(round(Dbackground+0.5e-3-1e-10, 3)) +  ', ' + str(round(2*Dbackground+0.5e-3-1e-10, 3)) + "muSv/h")
print("Niepewność względna mocy: " + str(round(DbackgroundRelative+0.5e-2-1e-10, 2)) + "%")
Dpower1Mean = ( Dpower1Mean0**2 + Dbackground**2 )**0.5 # niepewnosć mocy po odjęciu tła


# osłabienie(odległość)
    # dopasowanie funkcji 1/(x-x0)**2
        # niepewności
def f(x, a, x0):
    return a/(x+x0)**2
power1fit = curve_fit(f, power1Dist, power1Mean, sigma = Dpower1Mean)
power1Params = power1fit[0]
power1Cov = power1fit[1]
    # współczynniki dopasowania
a = power1Params[0]
x0 = power1Params[1]
    # niepewności współczyniików
Da = power1Cov[0][0] ** 0.5
Dx0 = power1Cov[1][1] ** 0.5
Dxr = ( Dx**2 + Dx0**2 )**0.5
A = a/Gamma*const
DA = Da/Gamma*const
    # reszty do oceny jakości dopasowania
power1Theor = f(power1Dist, *power1Params)
DyFit = (power1Mean - power1Theor)/Dpower1Mean
DyFitMean = np.mean(DyFit)
sigmaDyFitMean = np.std(DyFit)
pearson = pearsonr(power1Mean,power1Theor)[0]
print('\nWspółczynnik korelacji Pearsona: ' + str(round(pearson, 5)))
    # printy
print("\nWspółczynniki dopasowania funkcji:")
print("a = " + str(round(a, 0)) + 'W*m^2\nu(a) = ' + str(round(Da+0.5e-0-1e-10, 0)) +  ', ' + str(round(2*Da+0.5e-0-1e-10, 0)) + 'W*m^2')
print("x0 = " + str(round(x0, 2)) + 'cm\nu(x0) = ' + str(round(Dx0+0.5e-2-1e-10, 2)) +  ', ' + str(round(2*Dx0+0.5e-2-1e-10, 2)) + 'cm^2')
print("Niepewność odległości rzeczywistej u(xr) = " + str(round(Dxr+0.5e-2-1e-10, 2)) +  ', ' + str(round(2*Dxr+0.5e-2-1e-10, 2)) + 'cm')
print('Odległość rzeczywista[cm, Moc średnia po podjęciu tła [muSv/h], Niepewność mocy [muSv/h]')
print('\nSrednia reszt unormowanych: ' + str(round(DyFitMean, 3)) + '\nOdchylenie średniej reszt: ' + str(round(sigmaDyFitMean, 3)))
print('Aktywność próbki: ' + str(round(A, 6)) + 'GBq')
print('niepewność aktywności próbki: ' + str(round(DA+0.5e-6-1e-16, 6)) +  ', ' + str(round(2*DA+0.5e-6-1e-10, 6)) + 'GBq')
print('\nKOŃCOWE REZULTATY\nGr. rzeczywista [cm], Moc [muSc/h], Niepewnosć mocy [muSc/h]')
for i,j,k in zip(power1Dist + x0, power1Mean, Dpower1Mean):
    print(str(round(i, 2)) + ', ' + str(round(j, 2)) + ', ' + str(round(k, 3)) )
    # wykres
plt.subplot(121)
linspExt = (np.max(power1Dist) - np.min(power1Dist))
linsp = np.linspace(np.min(power1Dist) - linspExt*0.013, np.max(power1Dist) + linspExt*0.05, 1000)
plt.plot(linsp + x0, f(linsp, *power1Params), color = 'red')
plt.scatter(power1Dist + x0, power1Mean, color = 'purple')
plt.errorbar(power1Dist + x0, power1Mean, yerr=Dpower1Mean, capsize=3, ls = 'none', color = "magenta",  elinewidth=0.8)
plt.ylabel(r'Moc dawki skutecznej $P [\frac{\mu Sv}{h}]$', fontsize = 20)
plt.xlabel(r'Odleglosc rzeczywista $x [cm]$', fontsize = 20)
    # drugi wykres
plt.subplot(122)
power1Dist_ = np.zeros(np.size(power1Dist)-4) + x0
for i in range(0, np.size(power1Dist_)):
    power1Dist_[i] = power1Dist[i+4]
power1Mean_ = np.zeros(np.size(power1Mean)-4)
for i in range(0, np.size(power1Mean_)):
    power1Mean_[i] = power1Mean[i+4]
Dpower1Mean_ = np.zeros(np.size(Dpower1Mean)-4)
for i in range(0, np.size(Dpower1Mean_)):
    Dpower1Mean_[i] = Dpower1Mean[i+4]
linspExt = 0.05*(np.max(power1Dist_) - np.min(power1Dist_))
linsp = np.linspace(np.min(power1Dist_) - linspExt, np.max(power1Dist_) + linspExt, 1000)
plt.plot(linsp + x0, f(linsp, *power1Params), color = 'red')
plt.scatter(power1Dist_ + x0, power1Mean_, color = 'purple')
plt.errorbar(power1Dist_ + x0, power1Mean_, yerr=Dpower1Mean_, capsize=3, ls = 'none', color = "magenta",  elinewidth=0.8)
#plt.ylabel(r'Moc dawki skutecznej $P [\frac{\mu Sv}{h}]$')
plt.xlabel(r'Odleglosc rzeczywista $x [cm]$', fontsize = 20)
    # różnice dopasowania
plt.figure()
plt.plot(power1Dist + x0, DyFit, color = 'red')
plt.plot(power1Dist + x0, np.zeros(np.size(power1Dist)), color = 'black', linestyle = '--')
plt.scatter(power1Dist + x0, DyFit, color = 'purple', marker = 'D')
plt.ylabel(r'Reszty dopasowania $\frac{P_i - P(x_i)}{u(P_i)}$', fontsize = 20)
plt.xlabel(r'Odleglosc rzeczywista $x [cm]$', fontsize = 20)


# osłabienie w miedzi
print('\n\nMOC(GRUBOŚC MIEDZI)')
DpowerCuMean = (DpowerCuMean0**2 + Dbackground**2)**0.5
# logarytm z mocy osłabionej wiązki
powerCuMeanLn = np.array(powerCuMean)
for i in range(np.size(powerCuMeanLn)):
    powerCuMeanLn[i] = mh.log(powerCuMeanLn[i])
DpowerCuMeanLn = DpowerCuMean/powerCuMean
# regresja
modelConst = sm.add_constant(powerCuThic)
w = 1/( (DpowerCuThic)**2 + (DpowerCuMeanLn)**2 )
model = sm.WLS(powerCuMeanLn, modelConst, weights = w)
modelFit = model.fit()
params = modelFit.params
Dparams = modelFit.bse
a = params[1] # współczynnik nachylenia
b = params[0] # stała addytywna 
Da = Dparams[1] # model zwaraca od razu pierwiastek z wariancji
Db = Dparams[0]
# współczynnik osłabienia liniowego
mum = -a/rhoCu
Dmum = Da/rhoCu
mu = -a
Dmu = Da
    # printy
print("\nWspółczynniki dopasowania prostej")
print('a = ' + str(round(a, 3)) + ' 1/cm')
print('u(a) = ' + str(round(Da+5e-4-1e-10, 3)) +  ', ' + str(round(2*Da+0.5e-3-1e-10, 3)) + ' 1/cm')
print('b = ' + str(round(b, 2)) + ' [1]')
print('u(b) = ' + str(round(Db+5e-3-1e-10, 2)) +  ', ' + str(round(2*Db+0.5e-2-1e-10, 2)) + ' [1]')
print('Moc bez osłabienia: ' + str(round(mh.e**b, 2)) + 'muSv/h')
print('Niepewność mocy bez osłabienia: ' + str(round(Db*mh.e**b+0.5e-2+1e-10, 2)) +  ', ' + str(round(2*Db*mh.e**b+0.5e-2-1e-10, 2)) + 'muSv/h')
print('\nMasowy współczynnik osłabienia: ' + str(round(mum, 4)) + 'cm^2/g')
print('Niepewność masowego współczynnika osłabienia: ' + str(round(Dmum+5e-5-1e-10, 4)) +  ', ' + str(round(2*Dmum+0.5e-4-1e-10, 4)) + 'cm^2/g')
print('\nKOŃCOWE REZULTATY')
print('Grubość [mm], Niepewność [mm], Moc [muSv/h], Niepewność [muSv/h]')
for i,j,k,l in zip(powerCuThic, DpowerCuThic, powerCuMean, DpowerCuMean):
    print(str(round(i, 2)) + ', ' + str(round(j+0.005-1e-10, 2)) + ', ' + str(round(k, 2)) + ', ' + str(round(l+0.005-1e-10, 2)))
    # wykres  
#plt.subplot(122)
plt.figure()
linspExt = 0.05*(np.max(powerCuThic) - np.min(powerCuThic))
linsp = np.linspace(np.min(powerCuThic) - linspExt, np.max(powerCuThic) + linspExt, 1000)
plt.plot(linsp, a*linsp + b, color = 'red')
#plt.plot(linsp, (a-Da)*linsp + b + Db)
#plt.plot(linsp, (a+Da)*linsp + b - Db)
plt.scatter(powerCuThic, powerCuMeanLn, color = 'purple', s = 15)
plt.errorbar(powerCuThic, powerCuMeanLn, yerr=DpowerCuMeanLn, xerr = DpowerCuThic, capsize=3, ls = 'none', color = 'magenta',  elinewidth=0.8)
plt.xlabel("x [mm]", fontsize = 20)
plt.ylabel("ln(P) [1]", fontsize = 20)

print()
plt.show()
