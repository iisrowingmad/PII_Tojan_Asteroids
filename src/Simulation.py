#-------------------------------------------------------------------------------
# Purpose:     To calculate the paths of asteroids in the CR3BP

# BGN:	       6946S
# Created:     21/02/2021
#-------------------------------------------------------------------------------

# Scientific Python
import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp

# Optimisation Libraries
import cProfile
import multiprocessing as mp


class AsteroidPath:
    """A struct to contain information on the path of each asteroid."""
    def __init__(self, t, y):
        self.t = t
        self.rx = y[0]
        self.ry = y[1]
        self.vx = y[2]
        self.vy = y[3]
        self.v = y[2:].T
        self.r = y[:2].T

    def appened(self, path):
        """
        Appends the paths of one AsteroidPath instance to the end of another

        Parameters:

            path (AsteroidPath): path to add to end of the path contained in
                this instance of AsteroidPath.
        Returns:

            (AsteroidPath): The new asteroid path instance with the paths of
                the asteroid in the paths parameter added to the end.
        """
        return AsteroidPath(np.append(self.t, path.t),
                            np.array([np.append(self.rx, path.rx),
                                      np.append(self.ry, path.ry),
                                      np.append(self.vx, path.vx),
                                      np.append(self.vy, path.vy)]))


class CelestialBody:
    """A struct containing the information on a celestial body."""
    def __init__(self, position, mass):
        self.position = np.array(position)
        self.mass = mass


class Simulation:
    """
    A class to simulate the motion of asteroids in a sun and planet system in
    the planar circular restricted three-body problem (CR3BP).

    This class also includes functions to calculate other parameters of the
    simulation such as the Hamiltonian and the maximum wander distance of
    asteroids.

    Coordinates used here are cartesian with the origin at the barycentre of
    the planet and sun. The length unit is au and masses are in solar masses.
    Time is measured in earth years and therefore, G = 4*pi^2.

    The planet is placed along the positive y which is in the upwards direction
    in the drawing plane. Positive x is to the right. Angles are measured from
    the planet (hence from the y axis with positive angles being clockwise).
    Hence to convert from a coordinate to an angle one should use arctan(x, y).

    Methods:

        __init__: Constructor for the simulation. This sets up the parameters
            of the simulation and generates the asteroids to be simulated.

        run: Runs the simulation. This generates the paths of the asteroids.

        hamiltonian: Calculates the Hamiltonian of the asteroids in the
            rotating frame. The Hamiltonian is calculated as is a conserved
            quantity of the simulation.

        wander_distance: Calculates the maximum distance that individual
            asteroids have wandered from a chosen Lagrange point.

        remove_asteroid: Removes an asteroid from the system. Often useful if
            an asteroid has left the area of interest and therefore does not
            need to be simulated further.

        rotation_transformation: Preforms a transformation to move into a
            different rotating reference frame than the one of the planet. This
            transformation is NOT saved within the instance of the Simulation.

        _equations_of_motion: The equations of motion of the system used to
            calculate asteroid paths.

        _gravity_equation: Calculates the acceleration due to gravity at a
            point in space due to a gravitational body. Note the result needs
            to be multiplied by the gravitational constant (G) to get the true
            force on the body.

        _potential_energy: Calculates the potential energy at a point in space
            due to a gravitational body. Note the result needs to be multiplied
            by the gravitational constant (G) to get the true force on the
            body.

        _ODE: Uses SciPy integrator to calculate the paths of a given
            asteroid(s).

        _initial_position_to_index: Converts the initial position of an
            asteroid to the index of that asteroid in the other arrays in the
            simulation.

        _check_if_sim_performed: Check if a simulation has been performed or
            not.
    """
    # Physical Constants
    G = 4 * pi**2
    SUN_MASS = 1

    def __init__(self, asteroid_generator, N, *args, planet_mass=0.001, R=5.2):
        """
        Creates and initialises the parameters of the simulation for use
        later. This also generates the asteroids for the simulation.

        Parameters:

            asteroidGenerator (function): The function used to generate the
                initial position and velocities of the asteroids. The calling
                signature is asteroidGenerator(N, Simulation, *args). Where N
                is the number of asteroids, Simulation an instance of the
                Simulation class - This is used to extract specific
                parameters of each simulation. *args are any additional
                arguments required, for example the maximum velocity permitted.
                asteroidGenerator must return np.array of size (4,N).
                The rows must represent [rx, ry, vx, vy] for the initial
                positions and velocities of each asteroid in cartesian
                coordinates. If N asteroids are not generated, then the
                simulation will run with the number of asteroids generated but
                an error message will be displayed.

            N (int): Number of asteroids to attempt to generate.

            *args: Additional parameters (if needed) for the asteroid
                generation function.

            planet_mass (float): Mass of the planet. The default value is 0.001
                for Jupiter.

            R(float): Radius of the orbiting planet from the sun. The default
                value is 5.2 for Jupiter.
        """
        if N <= 0:
            raise Exception('The simulation must have at least 1 asteroid.')
        if int(N) != N:
            raise Exception('N must be an integer.')
        if planet_mass < 0:
            raise Exception('The planet mass must be positive.')
        if R < 0.005:
            raise Exception('R must be larger than the radius of the sun.')

        self.R = R
        self.planet_mass = planet_mass
        self.omega = (Simulation.G*(Simulation.SUN_MASS+planet_mass)/R**3)**0.5
        self.T = 2*pi / self.omega  # Period of oscillation

        # Move barycentre to origin.
        # r_sun is negative corresponding to the sun being below the barycentre
        self.r_sun = -R * planet_mass / (planet_mass + Simulation.SUN_MASS)
        self.r_planet = self.r_sun + R

        # Positions of the Lagrange points.
        self.L4 = [-R*np.sin(pi/3), R*np.cos(pi/3)+self.r_sun]
        self.L5 = [R*np.sin(pi/3), R*np.cos(pi/3)+self.r_sun]

        # For convenience when generating asteroids
        self.r_L4 = (self.L4[0]**2 + self.L4[1]**2)**0.5
        self.theta_L4 = np.arctan2(self.L4[0], self.L4[1])
        self.r_L5 = (self.L5[0]**2 + self.L5[1]**2)**0.5
        self.theta_L5 = np.arctan2(self.L5[0], self.L5[1])

        # Create the celestial bodies with a gravitational field.
        sun = CelestialBody([0, self.r_sun], Simulation.SUN_MASS)
        planet = CelestialBody([0, self.r_planet], planet_mass)
        self._bodies = np.array([sun, planet])

        # Generate asteroids.
        self.initial_positions = np.array(asteroid_generator(N, self, *args))

        if len(self.initial_positions) != N:
            print('Failed to produce', N, 'asteroids. '
                  'Simulation will run with', len(self.initial_positions),
                  'asteroids instead.')

        # If asteroidGenerator does not return N asteroids.
        self.N = len(self.initial_positions)

        # Initialise for later
        self.paths = None
        self._t_start = None
        self._t_end = None
        self._max_step = None

    def run(self, periods, extend=False,
            min_steps_per_orbit=12, threaded=True):
        """
        Runs the simulation.

        This function uses the scipy.integrate.solve_ivp ODE solver to solve
        for the path of each asteroid for the number of periods specified.

        Parameters:

            periods (float): The number of orbital periods of the planet to run
                the simulation for.

            extend (bool): Whether to run the simulation as an extension to
                the simulation if the simulation has already been run. This
                will start the simulation at the time where it previously
                finished. The default value is False.

            min_steps_per_orbit (int): The minimum number of evaluations per
                orbit used when calculating the paths of each asteroid. The
                default value is 12 which is provides a good balance for speed
                and accuracy.

            threaded (bool): Whether to run the simulation using
                multithreading. This significantly improves speed for large
                data sets although maybe slower for small data sets. The
                default value is True.

        Reutrns:

            (A list of class AsteroidPath dim(Simulation.N)): A list of the
                paths of each asteroid. The path information is contained
                within the class AsteroidPath for ease of extraction and
                plotting. This is also saved within the class so can be
                accessed later if required.
        """
        self._max_step = self.T / min_steps_per_orbit

        if extend:
            # Get end positions for each asteroid.
            initial_positions = [[asteroid.rx[-1],
                                  asteroid.ry[-1],
                                  asteroid.vx[-1],
                                  asteroid.vy[-1]] for asteroid in self.paths]
            self._t_start = self._t_end

        else:
            initial_positions = self.initial_positions
            self._t_start = 0

        self._t_end = periods * self.T + self._t_start

        # Run the simulation for each asteroid.
        if threaded:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                paths = np.array(pool.map(self._ODE, initial_positions))

                # Handle memory issues.
                pool.close()
                pool.join()
        else:
            paths = np.array([self._ODE(asteroid)
                              for asteroid in initial_positions])

        # Reformat result to make plotting more intuitive.
        paths = [AsteroidPath(asteroid.t, asteroid.y) for asteroid in paths]

        # If extended then add paths to the end of the old ones.
        if extend:
            self.paths = [self.paths[n].appened(asteroid_path)
                          for n, asteroid_path in enumerate(paths)]
        else:
            self.paths = paths

        return self.paths

    def remove_asteroid(self, initial_position):
        """
        Function to remove an asteroid from the simulation.

        Parameters:
            initial_position (np.array dim(4)): The initial position
                (in format [x, y, vx, vy]) of the asteroid to be removed.
                If two asteroids have the same initial position, then only the
                first will be removed.
        """
        self._check_if_sim_performed()
        '''
        Cannot rely on the positions within the array remaining constant
        throughout operation therefore cannot provide the index so need to use
        the initial position to find the asteroid. If two asteroids had the
        same initial position then they would have evolved the same way so it
        should not matter which is removed.
        '''
        n = self._initial_position_to_index(initial_position)

        # Delete the asteroid
        self.initial_positions = np.delete(self.initial_positions, n, 0)
        self.paths = np.delete(self.paths, n, 0)

    def hamiltonian(self, initial_position):
        """
        Calculates the Hamiltonian of the asteroids in the simulation.
        The Hamiltonian is calculated instead of the energy because in the
        rotating frame energy is not conserved whereas the Hamiltonian is.

            H = E - 1/2 (w x r)^2
        where:
            E = 1/2 v^2 + U(r)

        Parameters:

            initial_position (np.array dim(4)): The initial position
                (in format [x, y, vx, vy]) of the asteroid to calculate the
                Hamiltonian for.

        Returns:

            (np.array dim(2, n)): First column corresponds to times in years.
                The second column corresponds to the Hamiltonian of the
                asteroid at the time in the first column.
        """
        self._check_if_sim_performed()
        '''
        The ODE solver can take variable step sizes.
        This would mean that different asteroids would take a different
        number of steps to reach t_max. It is therefore difficult to sum
        the individual asteroid's energies over time. Or even return all the
        energies as lists as these lists would be of different lengths
        therefore this is implemented to only look at one asteroid at a time.
        '''
        # Find index of asteroid in question
        n = self._initial_position_to_index(initial_position)
        asteroid = self.paths[n]

        H = np.zeros(len(asteroid.t))

        for dt in range(len(asteroid.t)):

            v, r = asteroid.v[dt], asteroid.r[dt]

            # Correction for rotating frame (tangential velocity).
            v_T = np.cross([0, 0, self.omega], r)

            PE = - Simulation.G * sum(map(self._potential_energy,
                                          (r, r), self._bodies))

            H[dt] = 1/2 * v.dot(v) + PE - 1/2 * v_T.dot(v_T)

        return np.array([asteroid.t, H])

    def wander_distance(self, L5=False):
        """
        Calculates the maximum distance that each asteroid has travelled
        from a given Lagrange point during the simulation.

        Parameters:

            L5 (bool): A parameter to determine which Lagrange point to measure
                distances from. False means to calculate from L4 (the Lagrange
                point on the leading side of the planet, this is on the left
                for this simulation). True means calculating distances from L5
                (on the trailing side of the planet). The default value is
                False.

        Returns:

            (np.array dim(5,N)): A list of the initial positions and the
                maximum wander distance for each of the asteroids in the
                simulation. The format is: [rx, ry, vx, vy, wd], with wd being
                the wander distance. The other parameters are the initial
                displacement and velocity of the asteroid.
        """
        self._check_if_sim_performed()

        if L5:
            L = self.L5
        else:
            L = self.L4
        '''
        Loop over each asteroid and find maximum distance from L
        then add this distance and the initial position of that asteroid
        to the array.
        '''
        return np.array([[self.initial_positions[n][0],
                          self.initial_positions[n][1],
                          self.initial_positions[n][2],
                          self.initial_positions[n][3],
                          max(np.sqrt(np.sum((ast.r-L)**2, axis=-1)))]
                         for n, ast in enumerate(self.paths)])

    def rotation_transformation(self, omega):
        """
        Transforms the result of the simulation into a different rotating
        frame.

        The result of this is NOT saved to the instance of the simulation.
        Therefore the Simulation.paths parameter will always be in the planet's
        rotating frame. This is done to avoid confusion.

        Parameters:

            omega (float): Angular velocity of the output frame. This angular
                velocity should be given relative to an inertial -
                non-rotating frame. Therefore using an angular velocity of the
                planet in the simulation (self.omega) is therefore the same as
                no transformation and the function will simply return the same
                as self.paths.

        Returns:

            (list of class (AsteroidPath) dim(Simulation.N)): The paths of the
                asteroids in the new rotating frame formatted as a list of
                AsteroidPath classes.
        """
        self._check_if_sim_performed()

        # Subtract the current rotation of the simulation.
        omega -= self.omega

        # For simplicity
        array = np.array

        '''
        Performs the transformation for each time and each asteroid the
        resultant transformed coordinates are then used to create new
        AsteroidPath classes with the transformed data. This is simply a
        standard rotating frame transformation for velocity and position.
        '''
        return [AsteroidPath(ast.t,  # No change in time coordinate.
                             array([[ast.rx[n] * np.cos(-omega * t) -
                                     ast.ry[n] * np.sin(-omega * t),
                                     ast.rx[n] * np.sin(-omega * t) +
                                     ast.ry[n] * np.cos(-omega * t),
                                     ast.vx[n] +
                                     np.cross([0, 0, omega], ast.r[n])[0],
                                     ast.vy[n] +
                                     np.cross([0, 0, omega], ast.r[n])[1]]
                                    for n, t in enumerate(ast.t)]).T)
                for ast in self.paths]

    def _equations_of_motion(self, t, y):
        """
        Calculates the right-hand side of the equations of motion of the
        in a frame rotating at the velocity of the planet.

        This is solely intended to be used as the fun parameter for
        scipy.integration.solve_ivp.

        Parameters:

            t (float): Time of evaluation, required by solve_ivp.

            y (np.array(4)): Position to evaluate equations at. This is a
                vector with components [rx,ry,vx,vy] corresponding to both
                displacement and velocity of an asteroid.

        Returns:

            (list(4)): The derivatives of y in the rotating frame.

        """
        r = y[:2]
        vx, vy = y[2:]
        '''
        Calculate the acceleration in the rotating frame. This is the sum
        of the gravitational force and centripetal force. The Coriolis force
        is excluded here for increased performance.
        G is also multiplied here as is faster compared to multiplying
        within the gravity function.
        '''
        accel = (Simulation.G * sum(map(self._gravity_equation,
                                        (r, r), self._bodies)) +
                 self.omega**2*r)

        return [vx,                          # drx / dt
                vy,                          # dry / dt
                accel[0] + 2*self.omega*vy,  # dvx / dt, coriolis force added
                accel[1] - 2*self.omega*vx]  # dvy / dt

    def _gravity_equation(self, r, body):
        """
        Calculates the gravitational acceleration at position r (np.array(2))
        created by body (CelestialBody class). The gravitational constant
        (G) is omitted for performance reasons.
        """
        r = body.position - r  # Displacement from body
        return body.mass * r / r.dot(r)**1.5

    def _potential_energy(self, r, body):
        """
        Calculates the gravitational potential energy at position r
        (np.array(2)) created by body (CelestialBody class). The gravitational
        constant (G) is omitted for performance reasons.
        """
        r = body.position - r  # Displacement from body
        return body.mass / r.dot(r)**0.5

    def _ODE(self, asteroid):
        """
        Runs the scipy.integrate.solve_ivp using the DOP853 algorithm to
        calculate the path of asteroid. This is separated from Simulation.run
        to allow multithreading.

        Parameters:

          asteroid (np.array dim(4)): The initial position of the asteroid to
              be simulated. The components of asteroid are [x, y, vx, vy].
        """
        return solve_ivp(self._equations_of_motion,
                         [self._t_start, self._t_end],
                         asteroid,
                         method='DOP853',
                         max_step=self._max_step,
                         dense_output=True)

    def _initial_position_to_index(self, ip):
        """
        Finds the index of an asteroid in the initial positions and hence the
        paths arrays. This is done using the the initial position of the
        asteroid given by the ip (np.array(4)) parameter. The components of
        ip are [x, y, vx, vy].
        """
        return np.nonzero(np.all(self.initial_positions == ip, axis=-1))[0][0]

    def _check_if_sim_performed(self):
        """
        Check if a Simulation.run has been performed and raise exception if
        not.
        """
        if self.paths is None:
            raise Exception('No simulation performed. Use Simulation.run() '
                            'first.')
