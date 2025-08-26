#############################################################################
# Program Title: 
# Creation Date: 
# Description: 
#
##### Imports
import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import scipy.linalg as LA
import statistics as stat
warnings.filterwarnings('ignore')
##### Functions
def rearrange(lis):			#Reorder the dates in chronological order
	new = []
	denied = 0
	for i in lis:
		#date = str(i)[7:9] + "-" + str(i)[2:6]
		date = str(i)[2:9]
		try:						#Check for possible incomplete records
			int(date[2:4])
		except ValueError:	#Store number of incomplete records
			denied = denied + 1
		else:
			new.append(date)
	new.sort(key=lambda date: datetime.strptime(date, "%Y-%m"))
	return new, denied
	
def count(lis):				#Convert dates to number of months since first patent			
	new = []
	start = int(lis[0][0:4]) * 12 + int(lis[0][5:8])
	for i in lis:
		new.append((int(i[0:4])*12 + int(i[5:8])) - start)
	return new
	
def makax(lis):				#Count number of patents for each month to create axese
	x = np.linspace(0, np.max(lis), np.max(lis))
	y = []
	ycum = []
	for i in x:
		y.append(lis.count(int(i)))
	for i in x:
		ycum.append(sum(y[:int(i)]))
	return np.array(x, dtype=np.float128), np.array(y, dtype=np.float128), np.array(ycum, dtype=np.float128)
	
def f(x, a, b, c):			#Exponential function for optimization
	return a**(x-b) + c
#####
#Analogue Watch
awatch = pd.read_csv("analoge_watch.csv", usecols=['publication date'])		#Read publication date from csv file
awatch = list(np.array(awatch))		#Convert dates to np array then store as a list
new, denied = rearrange(awatch)		#Create new list in chronological order and save number of incomplete records
print("There were {} patent(s) without a publication date listed for 'Analogue Watch'".format(denied))		#Print number of incomplete records for product
new = count(new)		#Convert dates to number of elapsed months
x1, y1, ycum1 = makax(new)		#Store axese for plotting

#Plot data
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))		#Create Figure with 2 subplots for number of patents and cumulative patents
fig1.suptitle("Analogue Watch Patents", fontsize=20)			
ax1.scatter(x1, y1, s=4, label="Data Points")					#Plot Data points on subplot 1
ax1.set_xlabel('Elapsed Months', fontsize=15)					#Label the axese
ax1.set_ylabel('Patents Published per Month', fontsize=15)
ax1.grid(True, which='both')											#Turn on the grid for subplot
ax2.scatter(x1, ycum1, s=4, label="Data Points")				#Plot Cumulative Data Points on subplot 2
ax2.set_xlabel('Elapsed Months', fontsize=15)					#Label subplot 2's axese
ax2.set_ylabel('Cumulative Patents Published', fontsize=15)
ax2.grid(True, which='both')											#Turn on grid
plt.gca().tick_params(labelsize=10)									#Set font size of axis intervals
plt.minorticks_on()														#Turn on axis ticks

#Digital Watch
dwatch = pd.read_csv("digital_watch.csv", usecols=['publication date'])
dwatch = list(np.array(dwatch))
new, denied = rearrange(dwatch)
print("There were {} patent(s) without a publication date listed for 'Digital Watch'".format(denied))
new = count(new)
x2, y2, ycum2 = makax(new)
#Plot Data
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Digital Watch Patents", fontsize=20)
ax3.scatter(x2, y2, s=4, label="Data Points")
ax3.set_xlabel('Elapsed Months', fontsize=15)
ax3.set_ylabel('Patents Published per Month', fontsize=15)
ax3.grid(True, which='both')
ax4.scatter(x2, ycum2, s=4, label="Data Points")
ax4.set_xlabel('Elapsed Months', fontsize=15)
ax4.set_ylabel('Cumulative Patents Published', fontsize=15)
ax4.grid(True, which='both')
plt.gca().tick_params(labelsize=10)
plt.minorticks_on()

#Caster Wheel
cwheel = pd.read_csv("caster_wheel.csv", usecols=['publication date'])
cwheel = list(np.array(cwheel))
new, denied = rearrange(cwheel)
print("There were {} patent(s) without a publication date listed for 'Caster Wheel'".format(denied))
new = count(new)
x3, y3, ycum3 = makax(new)
#Plot Data
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("Caster Wheel Patents", fontsize=20)
ax5.scatter(x3, y3, s=4, label="Data Points")
ax5.set_xlabel('Elapsed Months', fontsize=15)
ax5.set_ylabel('Patents Published per Month', fontsize=15)
ax5.grid(True, which='both')
ax6.scatter(x3, ycum3, s=4, label="Data Points")
ax6.set_xlabel('Elapsed Months', fontsize=15)
ax6.set_ylabel('Cumulative Patents Published', fontsize=15)
ax6.grid(True, which='both')
plt.gca().tick_params(labelsize=10)
plt.minorticks_on()
#Doorknob
knob = pd.read_csv("doorknob.csv", usecols=['publication date'])
knob = list(np.array(knob))
new, denied = rearrange(knob)
print("There were {} patent(s) without a publication date listed for 'Doorknob'".format(denied))
new = count(new)
x4, y4, ycum4 = makax(new)
#Plot Data
fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 5))
fig4.suptitle("Doorknob Patents", fontsize=20)
ax7.scatter(x4, y4, s=4, label="Data Points")
ax7.set_xlabel('Elapsed Months', fontsize=15)
ax7.set_ylabel('Patents Published per Month', fontsize=15)
ax7.grid(True, which='both')
ax8.scatter(x4, ycum4, s=4, label="Data Points")
ax8.set_xlabel('Elapsed Months', fontsize=15)
ax8.set_ylabel('Cumulative Patents Published', fontsize=15)
ax8.grid(True, which='both')
plt.gca().tick_params(labelsize=10)
plt.minorticks_on()

#Lightswitch
switch = pd.read_csv("lightswitch.csv", usecols=['publication date'])
switch = list(np.array(switch))
new, denied = rearrange(switch)
print("There were {} patent(s) without a publication date listed for 'Light Switch'".format(denied))
new = count(new)
x5, y5, ycum5 = makax(new)
#Plot Data
fig5, (ax9, ax10) = plt.subplots(1, 2, figsize=(12, 5))
fig5.suptitle("Light Switch Patents", fontsize=20)
ax9.scatter(x5, y5, s=4, label="Data Points")
ax9.set_xlabel('Elapsed Months', fontsize=15)
ax9.set_ylabel('Patents Published per Month', fontsize=15)
ax9.grid(True, which='both')
ax10.scatter(x5, ycum5, s=4, label="Data Points")
ax10.set_xlabel('Elapsed Months', fontsize=15)
ax10.set_ylabel('Cumulative Patents Published', fontsize=15)
ax10.grid(True, which='both')
plt.gca().tick_params(labelsize=10)
plt.minorticks_on()

#Computer mouse
mouse = pd.read_csv("mouse.csv", usecols=['publication date'])
mouse = list(np.array(mouse))
new, denied = rearrange(mouse)
print("There were {} patent(s) without a publication date listed for 'Computer Mouse'".format(denied))
new = count(new)
x6, y6, ycum6 = makax(new)
#Plot Data
fig6, (ax11, ax12) = plt.subplots(1, 2, figsize=(12, 5))
fig6.suptitle("Computer Mouse Patents", fontsize=20)
ax11.scatter(x6, y6, s=4, label="Data Points")
ax11.set_xlabel('Elapsed Months', fontsize=15)
ax11.set_ylabel('Patents Published per Month', fontsize=15)
ax11.grid(True, which='both')
ax12.scatter(x6, ycum6, s=4, label="Data Points")
ax12.set_xlabel('Elapsed Months', fontsize=15)
ax12.set_ylabel('Cumulative Patents Published', fontsize=15)
ax12.grid(True, which='both')
plt.gca().tick_params(labelsize=10)
plt.minorticks_on()


#Tissue Paper
tp = pd.read_csv("tp.csv", usecols=['publication date'])
tp = list(np.array(tp))
new, denied = rearrange(tp)
print("There were {} patent(s) without a publication date listed for 'Tissue Paper'".format(denied))
new = count(new)
x7, y7, ycum7 = makax(new)
#Plot Data
fig7, (ax13, ax14) = plt.subplots(1, 2, figsize=(12, 5))
fig7.suptitle("Tissue Paper Patents", fontsize=20)
ax13.scatter(x7, y7, s=4, label="Data Points")
ax13.set_xlabel('Elapsed Months', fontsize=15)
ax13.set_ylabel('Patents Published per Month', fontsize=15)
ax13.grid(True, which='both')
ax14.scatter(x7, ycum7, s=4, label="Data Points")
ax14.set_xlabel('Elapsed Months', fontsize=15)
ax14.set_ylabel('Cumulative Patents Published', fontsize=15)
ax14.grid(True, which='both')
plt.gca().tick_params(labelsize=10)
plt.minorticks_on()

#Save image of plots without regressions
fig1.savefig("Analogue_Watch2.eps", format='eps')
fig2.savefig("Digital_Watch2.eps", format='eps')
fig3.savefig("Caster_Wheel2.eps", format='eps')
fig4.savefig("Doorknob2.eps", format='eps')
fig5.savefig("Lightswitch2.eps", format='eps')
fig6.savefig("Mouse2.eps", format='eps')
fig7.savefig("TP2.eps", format='eps')

#Regression Data
m = 7

#Analogue Watch Regressions
#Vandermonde
A = np.vander(x1, increasing=True)		#Create Vandermonde matrix for m degree polynomial
A = A[:, :(m+1)]
a = LA.lstsq(A, y1)[0]						#Solve for coefficients of polynomial using least squares
reg = np.polyval(np.flip(a), x1)			#Create list of y values for each x according to polynomial
res = np.sum((y1-reg)**2)					#Calculate residuals
dev = np.sum((y1-stat.mean(y1))**2)		#Calculate deviation of y points
R1 = 1 - res/dev								#Calculate R^2 Value
ax1.plot(x1, reg, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R1))	#Plot regression with legend entry
a2 = LA.lstsq(A, ycum1)[0]					#Solve for coefficients of polynomail for cumulative data
reg2 = np.polyval(np.flip(a2), x1)		#List of y values for cumulative regression
res = np.sum((ycum1-reg2)**2)				#Calculate residuals
dev = np.sum((ycum1-stat.mean(ycum1))**2)	#Calculate deviation of ycum values
R2 = 1 - res/dev								#Calculate R^2 value
ax2.plot(x1, reg2, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R2))	#Plot cumulative regression with legend entry

#Optimization
o, s = scipy.optimize.curve_fit(f, x1, y1, maxfev=len(x1))		#Find coefficients of given function f
fit = f(x1, o[0], o[1], o[2])		#Calculate list of y values using function f
res = np.sum((y1-fit)**2)			#Calculate residuals
dev = np.sum((y1-stat.mean(y1))**2)		#Calculate deviation of y values
R1 = 1 - res/dev						#Calculate R^2 value
ax1.plot(x1, fit, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R1))	#Plot regression with legend entry
o, s = scipy.optimize.curve_fit(f, x1, ycum1, maxfev=len(x1))	#Find coefficients for ycum regression function
fit2 = f(x1, o[0], o[1], o[2])		#Calculate list of y values using function
res = np.sum((ycum1-fit2)**2)			#Calculate residuals
dev = np.sum((ycum1-stat.mean(ycum1))**2)	#Calculate deviation of ycum values
R2 = 1 - res/dev		#Calculate R^2 value
ax2.plot(x1, fit2, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R2))	#Plot regression with legend entry
ax1.legend()		#Show legends
ax2.legend()

#Digital Watch
#Vandermonde
A = np.vander(x2, increasing=True)
A = A[:, :(m+1)]
a = LA.lstsq(A, y2)[0]
reg = np.polyval(np.flip(a), x2)
res = np.sum((y2-reg)**2)
dev = np.sum((y2-stat.mean(y2))**2)
R1 = 1 - res/dev
ax3.plot(x2, reg, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R1))
a2 = LA.lstsq(A, ycum2)[0]
reg2 = np.polyval(np.flip(a2), x2)
res = np.sum((ycum2-reg2)**2)
dev = np.sum((ycum2-stat.mean(ycum2))**2)
R2 = 1 - res/dev
ax4.plot(x2, reg2, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R2))
#Optimization did not produce coefficients for function f
#o, s = scipy.optimize.curve_fit(f, x2, y2, maxfev=len(x2))
#fit = f(x2, o[0], o[1], o[2])
#ax3.plot(x2, fit)
#o, s = scipy.optimize.curve_fit(f, x2, ycum2, maxfev=len(x2))
#fit = f(x2, o[0], o[1], o[2])
#ax4.plot(x2, fit)
ax3.legend()
ax4.legend()

#Caster Wheels
#Vandermonde
A = np.vander(x3, increasing=True)
A = A[:, :(m+1)]
a = LA.lstsq(A, y3)[0]
reg = np.polyval(np.flip(a), x3)
res = np.sum((y3-reg)**2)
dev = np.sum((y3-stat.mean(y3))**2)
R1 = 1 - res/dev
ax5.plot(x3, reg, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R1))
a2 = LA.lstsq(A, ycum3)[0]
reg2 = np.polyval(np.flip(a2), x3)
res = np.sum((ycum3-reg2)**2)
dev = np.sum((ycum3-stat.mean(ycum3))**2)
R2 = 1 - res/dev
ax6.plot(x3, reg2, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R2))

#Optimization
o, s = scipy.optimize.curve_fit(f, x3, y3, maxfev=len(x3))
fit = f(x3, o[0], o[1], o[2])
res = np.sum((y3-fit)**2)
dev = np.sum((y3-stat.mean(y3))**2)
R1 = 1 - res/dev
ax5.plot(x3, fit, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R1))
o, s = scipy.optimize.curve_fit(f, x3, ycum3, maxfev=len(x3))
fit2 = f(x3, o[0], o[1], o[2])
res = np.sum((ycum3-fit2)**2)
dev = np.sum((ycum3-stat.mean(ycum3))**2)
R2 = 1 - res/dev
ax6.plot(x3, fit2, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R2))
ax5.legend()
ax6.legend()

#Doorknob
#Vandermonde
A = np.vander(x4, increasing=True)
A = A[:, :(m+1)]
a = LA.lstsq(A, y4)[0]
reg = np.polyval(np.flip(a), x4)
res = np.sum((y4-reg)**2)
dev = np.sum((y4-stat.mean(y4))**2)
R1 = 1 - res/dev
ax7.plot(x4, reg, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R1))
a2 = LA.lstsq(A, ycum4)[0]
reg2 = np.polyval(np.flip(a2), x4)
res = np.sum((ycum4-reg2)**2)
dev = np.sum((ycum4-stat.mean(ycum4))**2)
R2 = 1 - res/dev
ax8.plot(x4, reg2, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R2))

#Optimization
o, s = scipy.optimize.curve_fit(f, x4, y4, maxfev=len(x4))
fit = f(x4, o[0], o[1], o[2])
res = np.sum((y4-fit)**2)
dev = np.sum((y4-stat.mean(y4))**2)
R1 = 1 - res/dev
ax7.plot(x4, fit, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R1))
o, s = scipy.optimize.curve_fit(f, x4, ycum4, maxfev=len(x4))
fit2 = f(x4, o[0], o[1], o[2])
res = np.sum((ycum4-fit2)**2)
dev = np.sum((ycum4-stat.mean(ycum4))**2)
R2 = 1 - res/dev
ax8.plot(x4, fit2, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R2))
ax7.legend()
ax8.legend()

#Lightswitch
#Vandermonde
A = np.vander(x5, increasing=True)
A = A[:, :(m+1)]
a = LA.lstsq(A, y5)[0]
reg = np.polyval(np.flip(a), x5)
res = np.sum((y5-reg)**2)
dev = np.sum((y5-stat.mean(y5))**2)
R1 = 1 - res/dev
ax9.plot(x5, reg, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R1))
a2 = LA.lstsq(A, ycum5)[0]
reg2 = np.polyval(np.flip(a2), x5)
res = np.sum((ycum5-reg2)**2)
dev = np.sum((ycum5-stat.mean(ycum5))**2)
R2 = 1 - res/dev
ax10.plot(x5, reg2, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R2))

#Optimization
o, s = scipy.optimize.curve_fit(f, x5, y5, maxfev=len(x5))
fit = f(x5, o[0], o[1], o[2])
res = np.sum((y5-fit)**2)
dev = np.sum((y5-stat.mean(y5))**2)
R1 = 1 - res/dev
ax9.plot(x5, fit, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R1))
o, s = scipy.optimize.curve_fit(f, x5, ycum5, maxfev=len(x5))
fit2 = f(x5, o[0], o[1], o[2])
res = np.sum((ycum5-fit2)**2)
dev = np.sum((ycum5-stat.mean(ycum5))**2)
R2 = 1 - res/dev
ax10.plot(x5, fit2, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R2))
ax9.legend()
ax10.legend()

#Computer mouse
#Vandermonde
A = np.vander(x6, increasing=True)
A = A[:, :(m+1)]
a = LA.lstsq(A, y6)[0]
reg = np.polyval(np.flip(a), x6)
res = np.sum((y6-reg)**2)
dev = np.sum((y6-stat.mean(y6))**2)
R1 = 1 - res/dev
ax11.plot(x6, reg, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R1))
a2 = LA.lstsq(A, ycum6)[0]
reg2 = np.polyval(np.flip(a2), x6)
res = np.sum((ycum6-reg2)**2)
dev = np.sum((ycum6-stat.mean(ycum6))**2)
R2 = 1 - res/dev
ax12.plot(x6, reg2, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R2))

#Optimization was not able to produce coefficients for function f
#o, s = scipy.optimize.curve_fit(f, x6, y6, maxfev=len(x6))
#fit = f(x6, o[0], o[1], o[2])
#ax11.plot(x6, fit)
#o, s = scipy.optimize.curve_fit(f, x6, ycum6, maxfev=len(x6))
#fit = f(x6, o[0], o[1], o[2])
#ax12.plot(x6, fit)
ax11.legend()
ax12.legend()

#Tissue paper
#Vandermonde
A = np.vander(x7, increasing=True)
A = A[:, :(m+1)]
a = LA.lstsq(A, y7)[0]
reg = np.polyval(np.flip(a), x7)
res = np.sum((y7-reg)**2)
dev = np.sum((y7-stat.mean(y7))**2)
R1 = 1 - res/dev
ax13.plot(x7, reg, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R1))
a2 = LA.lstsq(A, ycum7)[0]
reg2 = np.polyval(np.flip(a2), x7)
res = np.sum((ycum7-reg2)**2)
dev = np.sum((ycum7-stat.mean(ycum7))**2)
R2 = 1 - res/dev
ax14.plot(x7, reg2, linewidth=2, linestyle='-.', color='k', label="{}{} degree polynomial fit R{} = {:.3f}".format(str(m), '\u1D57\u02B0', '\u00B2', R2))

#Optimization
o, s = scipy.optimize.curve_fit(f, x7, y7, maxfev=len(x7))
fit = f(x7, o[0], o[1], o[2])
res = np.sum((y7-fit)**2)
dev = np.sum((y7-stat.mean(y7))**2)
R1 = 1 - res/dev
ax13.plot(x7, fit, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R1))
o, s = scipy.optimize.curve_fit(f, x7, ycum7, maxfev=len(x7))
fit2 = f(x7, o[0], o[1], o[2])
res = np.sum((ycum7-fit2)**2)
dev = np.sum((ycum7-stat.mean(ycum7))**2)
R2 = 1 - res/dev
ax14.plot(x7, fit2, linewidth=2, linestyle='-.', color='r', label="Exponential curve fit R{} = {:.3f}".format('\u00B2', R2))
ax13.legend()
ax14.legend()

#Save image of plots with regressions
fig1.savefig("Analogue_Watch.eps", format='eps')
fig2.savefig("Digital_Watch.eps", format='eps')
fig3.savefig("Caster_Wheel.eps", format='eps')
fig4.savefig("Doorknob.eps", format='eps')
fig5.savefig("Lightswitch.eps", format='eps')
fig6.savefig("Mouse.eps", format='eps')
fig7.savefig("TP.eps", format='eps')

plt.show()
#Last Updated: 


