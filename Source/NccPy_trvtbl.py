# This program calculates the travel-time table to be used by the NccPy package.

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)


import numpy as np
import obspy.taup
import time
import datetime
import os

# Time the calculation
start_time = time.time()

print(f'\n>>>> This is NccPy_trvtbl.py  (Travel-time table calculation) <<<<')

# ## I. Parameters  ########################################
vtbdistini = 0           # Distance: start    (degree)
vtbdistend = 10          # Distance: end      (degree)
vtbdiststep = 0.001      # Distance: interval (degree)

vtbdepini = 0             # Depth: start    (km)
vtbdepend = 300           # Depth: end      (km)
vtbdepstep = 0.1          # Depth: interval (km)

velmodel = obspy.taup.TauPyModel(model="ak135")   # Set velocity model

# Set path to output directory:
dirname = f'Travel_table'
fnametag = f'AK135_near'

# ## II. Calculation  ######################################
# The size of the travel-time table array
vtbdistdur = int(round((vtbdistend - vtbdistini) / vtbdiststep + 1))
vtbdepdur = int(round((vtbdepend - vtbdepini) / vtbdepstep + 1))

# Declare saving arrays
vtbarrp = np.array(np.zeros([vtbdistdur, vtbdepdur]))          # P arrival in table
vtbarrs = np.array(np.zeros([vtbdistdur, vtbdepdur]))          # S arrival in table
# vtbarrdist = np.array(np.zeros([vtbdistdur, vtbdepdur]))     # Distance table
# vtbarrdep = np.array(np.zeros([vtbdistdur, vtbdepdur]))      # Depth table

for jj1 in range(vtbdistdur):
    for jj2 in range(vtbdepdur):
        disttmp = vtbdistini + jj1 * vtbdiststep
        deptmp = vtbdepini + jj2 * vtbdepstep

        # vtbarrdist[jj1, jj2] = disttmp
        # vtbarrdep[jj1, jj2] = deptmp

        # Default output to -1, for those without arrival time
        vtbarrp[jj1, jj2] = vtbarrs[jj1, jj2] = -1

        # The travel-time calculation
        travellll = velmodel.get_travel_times(source_depth_in_km=deptmp, distance_in_degree=disttmp,
                                              phase_list=['P', 'p', 'S', 's'])

        # Look for first arrival of P and S
        for jj3 in range(len(travellll)):
            if travellll[jj3].name == 'P' or travellll[jj3].name == 'p':
                vtbarrp[jj1, jj2] = travellll[jj3].time
                break

        for jj4 in range(len(travellll)):
            if travellll[jj4].name == 'S' or travellll[jj4].name == 's':
                vtbarrs[jj1, jj2] = travellll[jj4].time
                break

    print(f'Finish distance {disttmp:.4f}')

    end_time = time.time()
    total_time = end_time - start_time
    print("Time till now: ", str(datetime.timedelta(seconds=total_time)))

# Write them into files!
if os.path.isdir(dirname) == 0:
    os.mkdir(dirname)

ff1 = open(f'{dirname}/{fnametag}.P.txt', 'w')
ff2 = open(f'{dirname}/{fnametag}.S.txt', 'w')
# ff3 = open(f'{dirname}/{fnametag}.dist.txt', 'w')
# ff4 = open(f'{dirname}/{fnametag}.dep.txt', 'w')
ff5 = open(f'{dirname}/{fnametag}.para.txt', 'w')

np.savetxt(ff1, vtbarrp, fmt='%11.4f')
np.savetxt(ff2, vtbarrs, fmt='%11.4f')
# np.savetxt(ff3, vtbarrdist, fmt='%10.4f')
# np.savetxt(ff4, vtbarrdep, fmt='%10.4f')
print(f'{vtbdistini:.4f} {vtbdistend:.4f} {vtbdiststep:.4f} {vtbdepini:.4f} {vtbdepend:.4f} {vtbdepstep:.4f}', file=ff5)

ff1.close()
ff2.close()
# ff3.close()
# ff4.close()
ff5.close()

# Time the execution
end_time = time.time()
total_time = end_time - start_time

print("Time of execution", str(datetime.timedelta(seconds=total_time)))
