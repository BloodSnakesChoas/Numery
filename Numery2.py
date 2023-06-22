import numpy as np
import matplotlib.pyplot as plt
import random as rand
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
in_on_ball_ratio = []
on_ball_counter = 0
time = [0]

for i in range(200):  # creation of 200 random vectors inside the sphere
    vector = np.random.rand(3)
    while np.linalg.norm(vector) > 1:
        vector = np.random.random(3)
    for k in range(3):
        vector[k] = vector[k] * rand.choice([1, -1])
    electrons.append(ChargedParticle(vector, e_q, [0, 0, 0], e_m))  # Creation a list of 200 electrons
for electron in electrons:  # Recording of initial conditions
    x_step.append(electron.location[0])
    y_step.append(electron.location[1])
    z_step.append(electron.location[2])
    if abs(1 - np.linalg.norm(electron.location)) < 1e-5:  # Counting number of electrons in the sphere
        on_ball_counter += 1
if on_ball_counter != 0:
    in_on_ball_ratio.append((200 - on_ball_counter) / on_ball_counter)  # calculating Ratio of electrons inside to on
    # the sphere
    on_ball_counter = 0
else:
    in_on_ball_ratio.append(0)
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
        if np.linalg.norm(electron.location) > 1:  # Making sure the electrons don't go outside the sphere
            electron.location = electron.location / np.linalg.norm(electron.location)
        x_step.append(electron.location[0])  # Recording of locations in each time step
        y_step.append(electron.location[1])
        z_step.append(electron.location[2])
        if abs(1 - np.linalg.norm(electron.location)) < 1e-5:  # Counting number of electrons in the sphere
            on_ball_counter += 1
    if on_ball_counter != 0:
        in_on_ball_ratio.append((200 - on_ball_counter) / on_ball_counter)  # calculating Ratio of electrons inside
        # to on the sphere
        on_ball_counter = 0
    else:
        in_on_ball_ratio.append(0)
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
z = np.array(z)

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
# סעיף 6
fig1 = plt.figure()
plt.plot(time, in_on_ball_ratio)
plt.title("Ratio of particels in and on of ball to time")
plt.ylabel("Ratio of particles in to on the ball")
plt.xlabel("t[sec]")
plt.show()


# סעיף 7
def calc_electric_potential(r):
    """Function calculating of the sum of electric potential in the location r by all electrons"""
    K = 8.99e9  # colon force constant N*m^2*C^-2
    sum_p = 0
    for electron in electrons:
        sum_p += electron.charge / (np.linalg.norm((np.array(r) - np.array(electron.location))))
    electric_potential = K * sum_p
    return electric_potential


number_of_steps = 1000  # number of points for the potential graph
x_for_r = np.linspace(0, 3, number_of_steps)
electric_pot_lst = []
r_lst = []
for x_value in x_for_r:  # creation of potential and r vectors for the graph
    r = np.array([x_value, 0, 0])
    electric_pot_lst.append(calc_electric_potential(r))
    r_lst.append(np.linalg.norm(r))

plt.figure()
plt.plot(r_lst, electric_pot_lst)
plt.title("Electric Potential as a function of r")
plt.ylabel("electric_pot_lst [volt]")
plt.xlabel("r [m]")
plt.show()
