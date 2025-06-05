# Import stuff
import sys, os, time
import pandas as pd
import numpy as np

# Use plotly for plotting
import plotly
import plotly.graph_objs as go
import plotly.offline as pltlyoff
from IPython.display import display, HTML

# Custom files
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))) # Add parent folder to sys.path
import golf_ballstics

# Setup
pltlyoff.init_notebook_mode(connected=True)
print(sys.version)
print('Plotly version', plotly.__version__)

# Declare ballistics model
golf_m = golf_ballstics.golf_ballstics()
# Example of how to run the model

##########################################################
########### Simulate entire ball trajectory ##############

# Hit the ball
golf_m.initiate_hit(velocity = (185.5) * 0.44704,  # Convert from mph to mps
                    launch_angle_deg = 11.5, off_center_angle_deg = 0, 
                    spin_rpm = 2312, spin_angle_deg = 0,
                    windspeed = 0, windheading_deg = 100)  # Convert windspeed from mph to mps
# Simulation results stored in dataframe df
df=golf_m.df_simres 

display(df.head())

##########################################################
########### Calculate landing position ###################

# If we just want to calculate the landing position of the ball 

y2, y1 = golf_m.get_landingpos(velocity = (185.5) * 0.44704,  # Convert from mph to mps
                    launch_angle_deg = 11.5, off_center_angle_deg = 0, 
                    spin_rpm = 2312, spin_angle_deg = 0,
                    windspeed = 0, windheading_deg = 100)

print('Ball lands at ({},{})'.format(y1 * 1.09361, y2 * 1.09361))  # Convert to yards
### Default data used for all plotting ###
##########################################

# Default input case
dict_default_input = {
    'velocity':70, 
    'launch_angle_deg':12, 
    'spin_rpm':5000, 
    'windspeed':8, 
    'windheading_deg':100, 
    'spin_angle_deg':-10, 
    'off_center_angle_deg':-8}

# Ranges 
dict_ranges = {
    'velocity':(10, 90), 
    'launch_angle_deg':(6, 30), 
    'spin_rpm':(1000, 9000), 
    'windspeed':(0, 8), 
    'windheading_deg':(-90, 270), # [-90, 90] = tail wind
    'spin_angle_deg':(-45, 45), 
    'off_center_angle_deg':(-45, 45)}
# Run model
golf_m.initiate_hit(**dict_default_input)
df=golf_m.df_simres

# Remove points where z<0
trace0 = go.Scatter3d(x=df[df['z']>=0]['x'] * 1.09361, y=df[df['z']>=0]['y'] * 1.09361, z=df[df['z']>=0]['z'] * 3.28084,  # Convert to yards and feet
                        mode='markers',
                        marker = dict(
                            color = 'rgb(255, 255, 255)',
                            size = 3,
                            line = dict(
                                color = 'rgb(150, 150, 150)',
                                width = 1
                            )
                        ),
                        name='trajectory'
                     )

data = [trace0]

abs_x = max(abs(df[df['z']>=0]['x'])) # For plotting
scene=dict(
    xaxis=dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(165, 210, 247)',
        range = [-abs_x - 5, abs_x + 5],
        #range = [abs_x + 5, -abs_x - 5],
        title = 'x'
    ),
    yaxis=dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(165, 210, 247)',
        range = [0,max(df[df['z']>=0]['y'])*1.1],
        title = 'y'
    ),
    zaxis=dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(148, 206, 161)',
        range = [0, max(max(df['z']), 1)],
        title = ''
    ),
    aspectratio = dict( x= 1, y= 2.5, z = 1 ),
    #aspectratio = dict( x= 3, y= 1, z = 1 ),
    aspectmode = 'manual',
    camera = {'eye':{'x':-2.2, 'y':0.2, 'z':0.3}}
)


layout=go.Layout(title='', showlegend=False, margin={'t':10}, scene=scene)


fig=go.Figure(data=data, layout=layout)
pltlyoff.iplot(fig, filename='tmp', show_link=False)

y2, y1, err = golf_m.get_landingpos(check = True, **dict_default_input)
print('Input:', dict_default_input)
print('Landing position: ({},{}). Length = {}'.format(y1, y2, np.linalg.norm([y1, y2])))
# Input variable
x_name = 'velocity'

# Number of points
n_pts = 100

# Create x values
x_vals = np.linspace(dict_ranges[x_name][0], dict_ranges[x_name][1], n_pts)

# Copy default input
dict_input = dict_default_input.copy()

# Calculate landing position for each x in x_vals
y1_vals = np.zeros(n_pts)
y2_vals = np.zeros(n_pts)
for i in range(n_pts):
    dict_input[x_name] = x_vals[i]
    
    y2, y1 = golf_m.get_landingpos(**dict_input)
    y1_vals[i] = y1
    y2_vals[i] = y2

# Create plots
trace_y1 = go.Scatter(x = x_vals, y = y1_vals, mode = 'lines', name = 'y1')
layout = go.Layout(title = 'Plot of landing position y1', xaxis = dict(title = x_name), yaxis = dict(title = 'y1'))
fig_y1 = go.Figure(data=[trace_y1], layout=layout)

trace_y2 = go.Scatter(x = x_vals, y = y2_vals, mode = 'lines', name = 'y2')
layout = go.Layout(title = 'Plot of landing position y2', xaxis = dict(title = x_name), yaxis = dict(title = 'y2'))
fig_y2 = go.Figure(data=[trace_y2], layout=layout)

pltlyoff.iplot(fig_y1, filename='tmp', show_link=False)
pltlyoff.iplot(fig_y2, filename='tmp', show_link=False)