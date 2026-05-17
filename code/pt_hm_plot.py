#!/usr/bin/env python
#
# Compare N-body to the halo model and 1-loop PT for
# scale free models.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
#
# Read the Abacus data.
abacus = []
for enn in ['150','200','225']:
    nb = np.loadtxt('abacus_n'+enn+'.txt')
    abacus.append(nb)
#
# Make the figure.
fig,ax = plt.subplots(1,1,figsize=(6,3.25))
kap = np.geomspace(1e-3,1e2,100)
kR  = [1.5290,1.8850,2.675]
aPT = [0.2052,1.3850,2.582]
aHM = [0.2816,0.9282,6.534]
k10 = [0.617 ,0.072, 0.013]
enn = [-1.5,-2.0,-2.25]
kf  = 0.75 # Where to match counterterm to N-body
# Put in some "fake" points for the legend.
ax.plot([1e-5,1e-4],[1,1],'k:' ,label='EFT')
ax.plot([1e-5,1e-4],[1,1],'k--',label='HM')
ax.plot([1e-5,1e-4],[1,1],'k-' ,label='N-body',alpha=0.5)
# Now plot the models.
for i in range(3):
    lbl = r'$n='+f"{enn[i]:.2f}"+r'$'
    col = 'C'+str(i)
    # Plot the N-body
    x,y = abacus[i][:,0]/kR[i],abacus[i][:,1]
    ax.plot(x,y/x,'-',color=col,alpha=0.5)
    # Now the analytic models.
    lin = kap**(3+enn[i])
    hm  = lin + aHM[i]*kap**3
    pt  = lin + aPT[i]*kap**(2*(3+enn[i]))
    alp = (Spline(x,y)(kf)-Spline(kap,pt)(kf))/kf**(5+enn[i])
    print(f"n={enn[i]:.2f}, alpha={alp:.2f}")
    ct  = pt  + alp*kap**(5+enn[i])
    sf  = 1/kap
    # Each of the analytic models
    ax.plot(kap,sf*ct ,':' ,color=col,label=lbl)
    ax.plot(kap,sf*hm ,'--',color=col)
#
ax.legend(ncol=2)
#
ax.set_xlim(3e-2,2)
ax.set_ylim(1e-1,10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\kappa=k/k_\star$')
ax.set_ylabel(r'$\kappa^{-1}\ \Delta^2(\kappa)$')
#
fig.tight_layout()
plt.savefig('pt_hm_plot.pdf')
#
