#-------------------------------------------------------------------------------
# Purpose:     Plot how the simulation time and error varies with the
#              evaluations per orbit

# BGN:	       6946S
# Created:     14/04/2021
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Palatino Linotype'
rcParams['font.size'] = 15

# Data
min_eval_p_o = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 , 15, 20, 25, 30, 40,
                50, 60, 70]
time = [4.4, 4.1, 4.4, 4.7, 4.8, 5.4, 6.1, 6.4, 6.7, 7.7, 7.8, 8.3, 9.8, 12.4,
        14.7, 17.2, 22.7, 27.4, 33.3, 40.0]
error = [1.4e-2, 6.0e-3, 4.6e-3, 1.7e-3, 2.9e-4, 6.2e-5, 1.7e-5, 5.2e-6, 1.8e-6,
            7.3e-7, 3.1e-7, 1.5e-7, 2.1e-8, 1.5e-9, 2.0e-10, 3.9e-11, 3.1e-12,
            4.4e-13, 2.6e-13, 2.3e-13]


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()

# Plot time data.
colour = 'tab:blue'
ax1.plot(min_eval_p_o,time,'-', color=colour)
ax1.tick_params(axis='y', labelcolor=colour)

# Plot on log scale.
colour = 'tab:orange'
ax2.plot(min_eval_p_o, error,'-', color=colour)
ax2.tick_params(axis='y', labelcolor=colour)
ax2.set_yscale('log')

# Plot linear scale.
colour = 'tab:red'
ax3.spines['right'].set_position(('outward', 40))
ax3.plot(min_eval_p_o, error,'-', color=colour)
ax3.tick_params(axis='y', labelcolor=colour)

ax1.set_xlabel('Minimum evaluations per orbit')
ax1.set_ylabel('Time / s')
ax3.set_ylabel('% error in the Hamiltonian', rotation=270, labelpad=15)

plt.show()