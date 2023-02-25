#-------------------------------------------------------------------------------
# Purpose:     Calculate and plot wander distance for different masses of
#               planet

# BGN:	       6946S
# Created:     01/04/2021
#-------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import griddata
import csv

import matplotlib.pyplot as plt
from matplotlib import rcParams

FILE_NAME = '--.csv'
WD_LIM = 8      # Wander distance where an asteroid is not Trojan.
GRAPH_RES = 80  # Resolution of the grid contour plots is plotted to.


def extract_data_from_file():
    """
    Extract the data from the file in FILE_NAME.

    Returns:
        (np.array(N_mass, N_ast, 5)): Array of the initial positions and
            corresponding wander distances of the asteroids saved in FILE_NAME.
            N_mass is the number of different planet masses simulated. N_ast
            is the number of asteroids simulated per mass.
        (np.array(N_mass, 5)): Array of the parameters that each simulation was
            run using. Each row corresponds to the parameters for a different
            planet mass. Each row is in the order This is in the order
            [R, r_sun, omega, L4[0], L4[1]].
    """
    i = -1
    wD = np.empty((0, 5))
    with open(FILE_NAME, newline='') as csvfile:
        for n, row in enumerate(csvfile):
            split_data = list(map(str, row.split(',')))

            if n == 0:
                # Extract the first row as the number of masses and
                # number of asteroids.
                n_mass, n_asteroid = map(int, split_data)

                # Make arrays to save the data to.
                wDwM = np.empty((n_mass, n_asteroid, 5))
                sim_parameters = np.empty((n_mass, 5))

            # If the row is the header for a new planet mass.
            elif split_data[0] == '--':
                # Extract simulation parameters.
                sim_parameters[i+1] = list(map(float, split_data[1:]))

                # Once found the start of the data for the next planet mass
                # add the old data to the wDwM. So that the asteroid
                # data for each mass is grouped into one element of the array.
                if i >= 0:
                    wDwM[i] = wD
                    wD = np.empty((0, 5))  # Empty for the next mass
                i += 1
            else:
                # If asteroid data then add data to the current wD
                wD = np.append(wD, [list(map(float, split_data))], axis=0)

    wDwM[i] = wD
    return wDwM, sim_parameters


def plot_wd_with_masses(ax, wDwM, masses):
    """
    Plots a histogram showing how the number of asteroids with each wander
    distance changes with mass.

    Parameters:

        ax (matplotlib axis): The matplotlib axis to plot onto.

        wDwM(np.array(N_mass, N_ast, 5)): The initial positions and wander
            wander distances for each asteroid for each of the planet masses.

        masses(np.array(N_mass)) : List of the masses that each of the elements
            in the 0th axis of wDwM corresponds to.

    Returns:

        (matplotlib.collections.Collection) : The return value of the pcolor
            plotting function.

    """
    wD_bins = 25  # Arbitary for # of bins to put wander distance into
    wD_res = (wD_bins-1) / WD_LIM
    mass_res = (len(masses)-1)/max(masses)
    density = np.zeros((len(masses), wD_bins))

    # Find number of stable asteroids at each wander distance and mass
    # Here iterate through each mass and then each asteroid
    for mass_n, mass_wDs in enumerate(wDwM.swapaxes(0, -1)[-1].swapaxes(0, 1)):

        mass_index = round(masses[mass_n] * (mass_res))

        for asteroid_wD in mass_wDs:

            if asteroid_wD < WD_LIM:  # Check stable

                wD_index = round(asteroid_wD * wD_res)-1
                density[mass_index][wD_index] += 1

    # Create mesh to plot
    x, y = np.mgrid[0:max(masses)+1/mass_res:1/mass_res,
                    0:WD_LIM+1/wD_res:1/wD_res]

    cs = ax.pcolor(x, y, density, shading='auto')
    return cs


def plot_mass_contour(ax, wD, r_L4):
    """
    Plot the wander distance contour plot for a specific mass.

    Parameters:

        ax (matplotlib axis): The matplotlib axis to plot onto.

        wD(np.array(N_ast, 5)): The initial positions and wander
            wander distances for each asteroid for a given planet mass.

        r_L4 (float) : The distance from the barycentre to L4 for a specific
            mass.

    Returns:

        (matplotlib.contour.QuadContourSet) : The return value of the contourf
            plotting function.


    """
    r = (wD[0]**2+wD[1]**2)**0.5 - r_L4
    ri = np.linspace(min(r), max(r), GRAPH_RES)

    v_r = (wD[0]*wD[2]+wD[1]*wD[3]) / (wD[0]**2+wD[1]**2)**0.5
    vri = np.linspace(min(v_r), max(v_r), GRAPH_RES)

    [Xi, Yi] = np.meshgrid(ri, vri)
    rvr = griddata((r, v_r), wD[4], (Xi, Yi))  # Calculate grid to plot

    cs = ax.contourf(ri,
                     vri,
                     rvr,
                     np.linspace(0, WD_LIM, WD_LIM*4+1),
                     extend='both')

    ax.set_ylim(vri.min(), vri.max())
    ax.set_xlabel('Radial displacement / au')
    ax.set_ylabel('Radial velocity / au y^-1')

    return cs


def plot():
    """
    Creates plots of how the wander distance is influenced by the planet mass
    Plots:
        - Histogram of how planet mass changes the distribution of wander
          distances
        - A plot of 6 sub-plots showing the wander distance in r r_dot space
          for different masses.
    """
    rcParams['font.family'] = 'Palatino Linotype'
    rcParams['font.size'] = 12

    wDwM, sim_params = extract_data_from_file()
    masses = sim_params.T[4]

    #  Plot histogram of density of asteroids against wD and mass of planet
    fig, ax = plt.subplots()
    cs = plot_wd_with_masses(ax, wDwM, masses)
    cbar = fig.colorbar(cs)
    cbar.set_label('Number of stable asteroids', rotation=270, labelpad=13)
    ax.set_xlabel('Mass / Solar Masses')
    ax.set_ylabel('Wander distance / au')

    fig, axes_6 = plt.subplots(nrows=2, ncols=3)
    subplot_index = 0

    # For approximating the area of stability
    no_stable = np.zeros(len(masses))

    # Itterate through each mass
    for i, wD in enumerate(wDwM):

        # Prepare data that is needed for plotting
        L_4 = [0, 0]
        R, omega, L_4[0], L_4[1], mass = sim_params[i]
        r_L4 = (L_4[0]**2 + L_4[1]**2)**0.5

        wD = wD.T

        # Calculate number of stable asteroids
        for distance in wD[4]:
            if distance < WD_LIM:
                no_stable[i] += 1
        '''
        # Just plot for single mass
        if mass == 0.001:
            fig, axes_1 = plt.subplots()
            plot_mass_contour(axes_1,
                              wD,
                              r_L4)
        '''
        # Generate plot of 6 subplots from the masses below
        if mass in [0, 0.0001, 0.001, 0.006, 0.01, 0.02]:
            # Complicated index just to move the plot along the grid
            cs = plot_mass_contour(axes_6[subplot_index//3, subplot_index % 3],
                                   wD,
                                   r_L4)
            subplot_index += 1

    # Add colour bar to the 6 axis plot
    cbar = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cs, cax=cbar)
    cbar.tick_params(labelsize=10)
    cbar.set_ylabel('Wander distance / au', rotation=270, labelpad=15)

    # Plot how area of stability varies with mass
    fig, ax = plt.subplots()
    ax.plot(masses, no_stable, '-+', linewidth=0.4)
    ax.set_xlabel('Mass / Solar Masses')
    ax.set_ylabel('Number of stable asteroids')

    plt.show()

if __name__ == '__main__':
    plot()
