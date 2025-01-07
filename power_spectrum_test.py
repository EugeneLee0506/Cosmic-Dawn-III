#Trial code for the power spectrum analysis that I worked over the summer (not complete)

from powerbox import PowerBox
from powerbox import get_power
import time
import numpy as np
import astropy
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

snap=118
i_chunk_in=1

rho=np.load('/scratch/08389/tg876886/2048_data/output_'+str(snap).zfill(6)+'_temp/rho_'+str(i_chunk_in).zfill(2)+'.npy')
xion=np.load('/scratch/08389/tg876886/2048_data/output_'+str(snap).zfill(6)+'_temp/xion_'+str(i_chunk_in).zfill(2)+'.npy')
temp=np.load('/scratch/08389/tg876886/2048_data/output_'+str(snap).zfill(6)+'_temp/temp_'+str(i_chunk_in).zfill(2)+'.npy')

H0 = 0.677699966430664E+02
G = 6.67408E-8
c = 2.99792458e10
omega_b=0.450000017881393E-01
omega_m=0.307
omega_l=0.693
hv_alpha = 3.028113999E-12 
m_p = 1.6726E-24 
k_b = 1.3807E-16
e = 2.71828182845904
E13 = 1.936e-11 
E14 = 2.04e-11
H_alpha = 6.5646e-5
Mpc = 3.086e24


start = time.time()


rho = np.load('rho.npy')[:500,:500,:500].astype(np.float64)
x_ion = np.load('xion.npy')[:500,:500,:500].astype(np.float64)
temp = np.load('temp.npy')[:500,:500,:500].astype(np.float64)
unit_d = np.load('unit_d.npy').astype(np.float64)
redshift = np.load('redshift.npy').astype(np.float64)
scale = np.load('scale_factor.npy').astype(np.float64)


ha_wave = (H_alpha)*redshift[117]


def L_REC (T, rho, x_ion):
    l = hv_alpha * 1.17E-13*(T/10E3)**(-0.942-0.031*np.log(T/10E3))*((rho*unit_d[117]/m_p)*x_ion)**2 *0.76**2
    return l  
def L_col (T, rho, x_ion):
    col = (3.57E-17/T**0.5)*e**(-140360/T)*(1+7.8/(1+5E5/T))*(rho*unit_d[117]/m_p)**2*x_ion*(1-x_ion) *0.76**2
    return col


print('start computing L', time.time() - start)
Rec3D=L_REC(temp, rho, x_ion).astype(np.float64)
Col3D=L_col(temp, rho, x_ion).astype(np.float64)



snapnum = 118
ALL = (Col3D + Rec3D) * scale[snapnum-1]**3 # erg/s/cm^3, cm in comoving
a = (10**23)*(ha_wave)\
/(((4*np.pi*(1+redshift[snapnum-1])**4)*H0*(omega_m*(1+redshift[snapnum-1])**3+(omega_m-omega_l)*(1+redshift[snapnum-1])**2+omega_l)**0.5) * 1e5/Mpc)


I_nu = a*ALL
proj_I_nu = I_nu.sum(axis=2)


print('start get_power', time.time() - start)

p,k = get_power(I_nu, (64/8192)*500, remove_shotnoise=True, dimensionless=True, log_bins=False, bins=700)
p_proj,k_proj = get_power(proj_I_nu, (64/8192)*500, remove_shotnoise=True, dimensionless=True, log_bins=False, bins=500)


print('start plotting', time.time() - start)

plt.plot(k, (p*k**3) / 2*(np.pi)**2, label = "500^3 Cells in 3D")
plt.plot(k_proj, (p_proj*k_proj**3) / 2*(np.pi)**2, label = "500^3 Cells in 2D projection")
plt.ylabel('$Δ^2$(k) [$k^3$P(k)/2π$^2$]', fontsize = 18)
plt.xlabel("wave number k [h/cMpc]", fontsize = 18)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0,1e3)
plt.grid()
plt.savefig('./test.jpg')

print('done', time.time() - start)