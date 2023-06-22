import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
import pandas as pd
import plotly.express as px


class ChargedParticle:
    def __init__(self, location, charge, velocity, mass):
        self.location = np.array(location)
        self.charge = charge
        self.velocity = np.array(velocity)
        self.mass = mass
        self.force_on_self = np.array([])

    def calc_electric_field(self, target_location):
        """Calculate the electric field a particle projects onto a location"""
        return K * self.charge * (np.array(target_location) - np.array(self.location)) / (
            np.linalg.norm((np.array(target_location) - np.array(self.location)))) ** 3


# constants
e_m = 9.11e-31  # electron mass in Kg
e_q = -1.6e-19  # electron charge in C
K = 8.99e9  # colon force constant N*m^2*C^-2
t = 1e-3  # step time is seconds
electrons = []
x = []
y = []
z = []
x_step = []
y_step = []
z_step = []
time = [0]
for i in range(200):  # creation of 200 random vectors inside the disk
    vector = np.random.rand(2)
    vector = np.append(vector, 0)
    while np.linalg.norm(vector) > 1:
        vector = np.random.random(2)
        vector = np.append(vector, 0)
    for k in range(2):
        vector[k] = vector[k] * rand.choice([1, -1])
    electrons.append(ChargedParticle(vector, e_q, [0, 0, 0], e_m))  # Creation a list of 200 electrons
for electron in electrons:  # Recording of initial conditions
    x_step.append(electron.location[0])
    y_step.append(electron.location[1])
    z_step.append(electron.location[2])
x.append(x_step)
y.append(y_step)
z.append(z_step)
x_step = []
y_step = []
z_step = []

total_electric_field = 0
timeframe = range(1, 3000)
for timestep in timeframe:  # Making different steps of movement in the chosen timeframe of movement
    for electron in electrons:
        for other_electron in electrons:
            if electron != other_electron:
                total_electric_field += other_electron.calc_electric_field(electron.location)  # Calculating the
                # total electric field(as a vector) on a point by all the electrons but the one in the target location
        electron.force_on_self = electron.charge * total_electric_field  # Calculating then force as a vector on the
        # electron on the target location
        electron.location = electron.location + (electron.force_on_self / electron.mass) * (t ** 2) / 2
        # Calculating the new electron location by the work of the force
        total_electric_field = 0
        if np.linalg.norm(electron.location) > 1:  # Making sure the electrons don't go outside the disk
            electron.location = electron.location / np.linalg.norm(electron.location)
        x_step.append(electron.location[0])  # Recording of locations in each time step
        y_step.append(electron.location[1])
        z_step.append(electron.location[2])
    time.append(timestep * t)
    print(timestep)
    x.append(x_step)
    y.append(y_step)
    z.append(z_step)
    x_step = []
    y_step = []
    z_step = []

# Convert to NumPy arrays
x = np.array(x)
y = np.array(y)
z = np.zeros_like(x)

# Calculate distances from origin
distances = np.sqrt(x ** 2 + y ** 2 + z ** 2)

# Combine all the data into a single DataFrame
data = []
for i in range(len(time)):
    data.extend([{
        'Time': time[i],
        'X': x[i][j],
        'Y': y[i][j],
        'Z': z[i][j],
        'Distance': distances[i][j]
    } for j in range(len(x[i]))])
df = pd.DataFrame(data)

# Create the 3D scatter plot with time slider
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Distance', color_continuous_scale='viridis',
                    animation_frame='Time', range_color=(0, np.max(distances)))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='X [m]',
        yaxis_title='Y [m]',
        zaxis_title='Z [m]'
    )
)

# Show the graph
fig.show()

r_lst = []
electrons_density_lst = []
num_of_electrons_temp = 0
for i in range(5, 21):  # Creating graphs made from 4 to 19 rings at size dr
    section_points = np.linspace(0, 1, i)
    r_lst = []
    electrons_density_lst = []
    num_of_electrons_temp = 0
    for k in range(i-1):
        section_start = section_points[k]
        section_end = section_points[k+1]
        r_lst.append((section_start + section_end) / 2)  # Defining the location of the ring r as the middle of the ring
        for electron in electrons:
            if section_start < np.linalg.norm(electron.location) < section_end:
                # Counting how meny electrons are in each ring
                num_of_electrons_temp += 1
        if num_of_electrons_temp == 0:
            electrons_density_lst.append(0)
        else:
            electrons_density_lst.append((num_of_electrons_temp * e_q) / (2 * math.pi * (section_end - section_start)))
            # Calculating of charge density for each ring
        num_of_electrons_temp = 0

    plt.figure()
    plt.plot(r_lst, electrons_density_lst, 'b-', label='Lines')
    plt.scatter(r_lst, electrons_density_lst, c='red', label='Points')
    plt.title(f"Charge distribution λdr by {i-1} rings")
    plt.ylabel("σ [C/m]")
    plt.xlabel("r [m]")
plt.show()

r_lst = []
electrons_density_lst = []
num_of_electrons_temp = 0
for i in range(5, 21):  # creating graphs with 5 to 20 points - 5 to 20 discs of charge density
    section_points = np.linspace(0, 1, i)
    r_lst = []
    electrons_density_lst = []
    num_of_electrons_temp = 0
    for k in range(i):
        for electron in electrons:
            if np.linalg.norm(electron.location) < section_points[k]:
                # Counting how meny electrons are in each disk
                num_of_electrons_temp += 1
        if num_of_electrons_temp == 0:
            electrons_density_lst.append(0)
        else:
            electrons_density_lst.append((num_of_electrons_temp * e_q) / (2 * math.pi * (section_points[k])))
            # Calculating of charge density for each disk
        num_of_electrons_temp = 0
        r_lst.append(section_points[k])

    plt.figure()
    plt.plot(r_lst, electrons_density_lst, 'b-', label='Lines')
    plt.scatter(r_lst, electrons_density_lst, c='red', label='Points')
    plt.title(f"Charge distribution σ cut to {i} points")
    plt.ylabel("σ [C/m^2]")
    plt.xlabel("r [m]")
plt.show()
