import numpy as np
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
num_vectors = 200  # Number of random vectors to generate
vectors = np.random.rand(num_vectors, 2) * 2 - 1  # Scale and shift to fit in the square
for vector in vectors:
    vector = np.append(vector, 0)
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
        if abs(electron.location[0]) > 1:  # Making sure the electron don't go out of the square at the x axi
            electron.location[0] = np.sign(electron.location[0])
        if abs(electron.location[1]) > 1:  # Making sure the electron don't go out of the square at the y axi
            electron.location[1] = np.sign(electron.location[1])
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
