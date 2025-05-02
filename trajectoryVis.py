import numpy
import matplotlib.pyplot as pyplot


def simpleTrajectory(u, theta_deg):
    '''define a function to find the path of a projectile without drag'''
    
    '''define the value for the acceleration due to gravity'''
    g = 9.81
    
    '''convert the angle to radians'''
    theta_rad = numpy.radians(theta_deg)
  
    '''use the angle to find the components of initial velocity'''
    u_x = u * numpy.cos(theta_rad)
    u_y = u * numpy.sin(theta_rad)
    
    '''find the time when the ball will hit the floor'''
    t_end = (2 * u_y) / g  
    
    '''create an array of points to sample'''
    t = numpy.linspace(0, t_end, 30)
    
    '''find the x, y, and z co-ordinates at each point'''
    s_x = u_x * t
    s_y = u_y * t - 0.5 * g * t ** 2
    
    '''return the output arrays of points'''
    return s_x, s_y
    

def forceDrag(rho, v, C_D, area):
    '''define a function to calculate the force of drag on an object'''

    
    force = - 0.5 * rho * v ** 2 * C_D * area
    
    return force


def dragTrajectory(u, theta_deg, C_D):
    '''define a fucntion to find the path of a projectile with drag'''
    
    '''initialise variables'''
    interval = 0.01
    g = - 9.81
    rho = 1.225
    area = 0.00138
    
    '''convert angles to radians'''
    theta_rad = numpy.radians(theta_deg)
    
    s_x = 0
    s_y = 0
    
    v_x = u * numpy.cos(theta_rad)
    v_y = u * numpy.sin(theta_rad)
    
    
    '''mass in kg'''
    m = 0.045
    
    a_x = forceDrag(rho, v_x, C_D, area)/m
    a_y = (forceDrag(rho, v_y, C_D, area) )/m +g

    
    '''intialise arrays'''
    x_array = numpy.array([0])
    y_array = numpy.array([0])
    t_array = numpy.array([0])
    
    count = 0 
    
    while s_y >= 0:
        
        count = count + 1
        time = count * interval
        
        '''find the next point's position '''
        s_x = s_x + v_x * interval + 0.5 * a_x * interval ** 2
        s_y = s_y + v_y * interval + 0.5 * a_y * interval ** 2
        
        '''find the new velocity at the next point'''
        v_x = v_x + a_x * interval
        v_y = v_y + a_y * interval
        
        '''find the new acceleration'''
        a_x = forceDrag(rho, v_x, C_D, area)/m
        a_y = (forceDrag(rho, v_y, C_D, area) )/m +g
        
        '''append data to arrays'''
        x_array = numpy.append(x_array, s_x)
        y_array = numpy.append(y_array, s_y)
        t_array = numpy.append(t_array, time)
        
    return x_array, y_array, t_array
    
    
def liftTrajectory(u, theta_deg, C_D, spin_RPM):
    '''define a function to find the path of a ball with drag and lift caused
    by its rotation and the Magnus effect'''
    
    '''initialise variables'''
    interval = 0.01
    g = - 9.81
    rho = 1.225
    area = 0.00138
    
    '''convert angles to radians'''
    theta_rad = numpy.radians(theta_deg)
    
    '''find the magnitude of the lift force'''
    lift_mag = 0.285 * (1-numpy.exp(-0.00026*spin_RPM))
    
    
    
    s_x = 0
    s_y = 0
    
    v_x = u * numpy.cos(theta_rad)
    v_y = u * numpy.sin(theta_rad)
    
    
    '''mass in kg'''
    m = 0.045
    
    a_x = forceDrag(rho, v_x, C_D, area)/m
    a_y = (forceDrag(rho, v_y, C_D, area) )/m +g 

    
    '''intialise arrays'''
    x_array = numpy.array([0])
    y_array = numpy.array([0])
    t_array = numpy.array([0])
    
    count = 0 
    
    while s_y >= 0:
        
        count = count + 1
        time = count * interval
        
        '''find the next point's position '''
        s_x = s_x + v_x * interval + 0.5 * a_x * interval ** 2
        s_y = s_y + v_y * interval + 0.5 * a_y * interval ** 2
        
        '''find the new velocity at the next point'''
        v_x = v_x + a_x * interval
        v_y = v_y + a_y * interval
        
        '''find the components of lift'''
        theta_i = numpy.arctan(v_y/v_x)
        l_x = lift_mag * numpy.sin(theta_i)
        l_y = lift_mag * numpy.cos(theta_i)
        
        '''find the new acceleration'''
        a_x = forceDrag(rho, v_x, C_D, area)/m + l_x/m
        a_y = (forceDrag(rho, v_y, C_D, area) )/m +g +l_y/m
        
        '''append data to arrays'''
        x_array = numpy.append(x_array, s_x)
        y_array = numpy.append(y_array, s_y)
        t_array = numpy.append(t_array, time)
        
    return x_array, y_array, t_array 
    

'''give values for the intial conditions and graph the different models'''
ball_speed_mph = 150
initial_velocity = ball_speed_mph * 0.447
launch_angle = 12
spin = 12000
CD_dimpled = 0.25

x_values = simpleTrajectory(initial_velocity, launch_angle)[0]
y_values = simpleTrajectory(initial_velocity, launch_angle)[1]


x_values_drag_dimpled = dragTrajectory(initial_velocity, launch_angle, CD_dimpled)[0]
y_values_drag_dimpled = dragTrajectory(initial_velocity, launch_angle, CD_dimpled)[1]

x_values_lift_dimpled = liftTrajectory(initial_velocity, launch_angle, CD_dimpled, spin)[0]
y_values_lift_dimpled = liftTrajectory(initial_velocity, launch_angle, CD_dimpled, spin)[1]

pyplot.xlim(0,200)
pyplot.ylim(0,30)
pyplot.gca().set_aspect('equal', adjustable='box')
pyplot.plot(x_values, y_values, 'wo')
pyplot.plot(x_values_lift_dimpled, y_values_lift_dimpled, 'b')
pyplot.plot(x_values_drag_dimpled, y_values_drag_dimpled, 'g')

pyplot.xlabel('Distance travelled (m)')
pyplot.ylabel('Height of the ball (m)')
pyplot.title('Simple trajectory model with and without drag')
pyplot.show()

print("Distance travelled by ball with no drag :", x_values[-1] * 1.0936, "yards")
print("Distance travelled by dimpled ball with drag :", x_values_drag_dimpled[-1] * 1.0936, "yards")
print("Distance travelled by dimpled ball with drag and lift :", x_values_lift_dimpled[-1] * 1.0936, "yards")

    
    