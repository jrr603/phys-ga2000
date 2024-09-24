import numpy as np
import matplotlib.pyplot as plt
import random

#Initialize for 10000 Bi213 atoms
NBi_213 = 10000
NBi_209 = 0
NTl = 0
NPb = 0

#halflives for unstable atoms
tau_Bi_213 = 46*60
tau_Tl = 2.2*60
tau_Pb = 3.3*60

#timesettings
tmax = 20000 #sec
h=1.0 #sec
tpoints = np.arange(0.0,tmax,h)

#probablility of decay depending on each halflife
p_Bi_213 = 1 - 2**(-h/tau_Bi_213)
p_Tl = 1 - 2**(-h/tau_Tl)
p_Pb = 1 - 2**(-h/tau_Pb)

# Main loop
Tlpoints = [] 
Pbpoints = []
Bi_209points = []
Bi_213points = []

for t in tpoints: #loop for all time
#update values that we will plot based on number of atoms present at time t
  Bi_209points.append(NBi_209)
  Pbpoints.append(NPb)
  Tlpoints.append(NTl)
  Bi_213points.append(NBi_213)

#count the decays from bi213 to pb209 and from bi213 to thallium
  decayBiPb = 0
  decayBiTl = 0
  for i in range (NBi_213) :
    if random.random () <p_Bi_213: #probability that Bi213 will decay at all 
      if random.random () <0.0209:  #if it does decay there is 2 percent chance itll decay into thallium
        decayBiTl += 1
       # print('yuh')
      else: #and a 97 percent chance of lead
        decayBiPb += 1
  NBi_213 -= decayBiPb 
  NPb += decayBiPb
  NBi_213 -= decayBiTl 
  NTl += decayBiTl

#count the decays from thallium to lead
  decayTlPb = 0
  for i in range (NTl) :
    if random.random () <p_Tl:
      decayTlPb += 1
  NTl -= decayTlPb 
  NPb += decayTlPb

#count the decays from lead to bi209
  decayPbBi = 0
  for i in range (NPb) :
    if random.random () <p_Pb:
      decayPbBi += 1
  NPb -= decayPbBi 
  NBi_209 += decayPbBi
 


plt.plot (tpoints, Tlpoints)
plt.plot (tpoints, Pbpoints )
plt.plot (tpoints, Bi_209points)
plt.plot (tpoints, Bi_213points)
plt.xlabel ("Time")
plt.ylabel ("Number of atoms") 
plt.legend(['Tl','Pb','Bi_209','Bi_214'])
plt.show()