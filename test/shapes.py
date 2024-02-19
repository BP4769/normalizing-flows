import numpy as np
import matplotlib.pyplot as plt
import torch

###########################################################################
###   UTILS   #############################################################
###########################################################################

# used for testing principal manifold flows and visualising the contours

def generate_continuously_colored_samples(num_samples):
    """
    Generate a set of points sampled from a bivariate standard normal distribution, with a continuously changing color 
    assigned to each point based on the y-value, using a colormap analogous to a rainbow.

    :param num_samples: Number of samples to generate.
    :return: A tuple (samples, colors), where samples is a 2D array of shape (num_samples, 2), each row representing
             a sample [x, y], and colors is a list of colors corresponding to each sample.
    """
    # Sample x and y coordinates from standard normal distributions
    x_samples = torch.randn(num_samples)
    y_samples = torch.randn(num_samples)
    samples = torch.stack((x_samples, y_samples), dim=1)
    # x_samples = np.random.normal(0, 1, num_samples)
    # y_samples = np.random.normal(0, 1, num_samples)
    # samples = np.column_stack((x_samples, y_samples))


    # Use a colormap to map y-values to colors
    colormap = plt.cm.rainbow
    color_norm = plt.Normalize(y_samples.min(), y_samples.max())
    colors = colormap(color_norm(y_samples))

    return samples, colors

def generate_grid_data(x_range, y_range, x_tick, y_tick):
    """
    Generate a grid of points in the x-y plane.

    :param x_range: A tuple (x_min, x_max) specifying the range of x values.
    :param y_range: A tuple (y_min, y_max) specifying the range of y values.
    :param x_tick: Number of points to generate along the x-axis.
    :param y_tick: Number of points to generate along the y-axis.
    :return: A 2D tensor of shape (x_tick * y_tick, 2), each row representing a point [x, y].
    """
    # Generate x and y values for the grid
    x_values = torch.linspace(x_range[0], x_range[1], x_tick)
    y_values = torch.linspace(y_range[0], y_range[1], y_tick)
    
    # Create grid points
    grid_points = torch.cartesian_prod(x_values, y_values)
    
    return grid_points



###########################################################################
###   SHAPES   ############################################################
###########################################################################

# the shapes used for testing principal manifold flows

def caret(num_points, line_length, angle, noise_std, seed=None):
    """
    Generate points on a caret shape with the specified parameters.

    :param line_length: length of the lines (float)
    :param angle: angle of the lines (float)
    :param num_points: number of points to generate (int)
    :param std_dev: standard deviation of the Gaussian noise (float)
    :param seed: random seed (int)
    :return: x and y coordinates of the points
    """
    if seed is not None:
        np.random.seed(seed)
    # Calculate the slopes of the lines
    slope = np.tan(angle)

    # Calculate the number of points for each line
    num_points_per_line_1 = num_points // 2
    num_points_per_line_2 = num_points - num_points_per_line_1

    # Generate points for the first line
    x1 = np.linspace(0, line_length / 2, num_points_per_line_1)
    y1 = -slope * x1 + (line_length / 2) 

    # Generate points for the second line (mirror image)
    x2 = np.linspace(-line_length / 2, 0, num_points_per_line_2)
    y2 = slope * x2 + line_length / 2


    # Concatenate the x and y coordinates of the points from both lines
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    # Add Gaussian noise to the x and y coordinates
    # add seed for reproducibility
    np.random.seed(seed)
    x_noisy = x + np.random.normal(0, noise_std, x.shape)
    y_noisy = y + np.random.normal(0, noise_std, y.shape)

    return x_noisy, y_noisy



def circle(num_points, radius, noise_std, seed=None):
    """
    Generate points on a circle with radius `radius` and add Gaussian noise.

    :param radius: radius of the circle (float)
    :param num_points: number of points to generate (int)
    :param noise_std: standard deviation of the Gaussian noise (float)
    :param seed: random seed (int)
    :return: x and y coordinates of the points
    """
    if seed is not None:
        np.random.seed(seed)
    # Generate angles uniformly
    angles = np.linspace(0, 2*np.pi, num_points)
    
    # Create points in polar coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # Add noise to the points
    np.random.seed(seed)
    x += np.random.normal(0, noise_std, num_points)
    y += np.random.normal(0, noise_std, num_points)
    
    return x, y



def double_circle(num_points, radius1, radius2, noise_std, seed=None):
    """
    Generate points on 2 circles with radius `radius1` and `radius2` and add Gaussian noise.

    :param radius1: radius of the first circle (float)
    :param radius2: radius of the second circle (float)
    :param num_points: number of points to generate (int)
    :param noise_std: standard deviation of the Gaussian noise (float)
    :param seed: random seed (int)
    :return: x and y coordinates of the points
    """
    if seed is not None:
        np.random.seed(seed)
    # create 2 circles with different radius
    circle1 = circle(num_points, radius1, noise_std, seed)
    circle2 = circle(num_points, radius2, noise_std, seed)

    # concatenate the 2 circles
    x = np.concatenate((circle1[0], circle2[0]))
    y = np.concatenate((circle1[1], circle2[1]))

    return x, y


def n_regular_polygon(num_points, n, radius, noise_std, angle=0, arc_height=None, seed=None):
    """
    Generate points on a regular polygon with `n` sides and add Gaussian noise, with the 
    option to generate an arc instead of a straight line on the sides of the polygon when 
    arc_height is specified.

    :param n: number of sides of the polygon (int)
    :param radius: radius of the polygon - from center to any vertex (float)
    :param num_points: number of points to generate (int)
    :param noise_std: standard deviation of the Gaussian noise (float)
    :param angle: angle of rotation of the polygon (float)
    :param arc_height: height of the arc - if positive it arc towards the center of the polygon, 
                        and away from it otherwise. (float)
    :param seed: random seed (int)
    :return: x and y coordinates of the points, x and y coordinates of the vertices of the polygon,
    """
    if seed is not None:
        np.random.seed(seed)

    # calculate coordinates of the vertices of the polygon
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1] + (angle / 180 * np.pi)
    x_vertices = radius * np.cos(angles)
    y_vertices = radius * np.sin(angles)
    # add first vertex to the end to close the polygon
    x_vertices = np.append(x_vertices, x_vertices[0])
    y_vertices = np.append(y_vertices, y_vertices[0])

    # calculate number of points per side
    num_points_per_side = [num_points // n] * n
    num_points_per_side = [num_points_per_side[i] + 1 if i < num_points % n else num_points_per_side[i] for i in range(n)]

    # generate points
    x = np.array([])
    y = np.array([])
    if arc_height is not None:
        # calculate side angles (relative to x-axis)
        side_angles = np.arctan2(y_vertices[1:] - y_vertices[:-1], x_vertices[1:] - x_vertices[:-1])

        # calculate arc radius (r = d^2/8h + h/2)
        d = np.sqrt((x_vertices[1] - x_vertices[0])**2 + (y_vertices[1] - y_vertices[0])**2)
        arc_radius = d**2 / (8 * arc_height) + arc_height / 2

        # calculate arc centers
        arc_angle = np.arcsin(d / (arc_radius * 2))
        beta = np.arccos(d / (arc_radius * 2))
        gammas = side_angles - beta
        arc_x_centers = [np.cos(gammas[i]) * arc_radius + x_vertices[i] for i in range(n)]
        arc_y_centers = [np.sin(gammas[i]) * arc_radius + y_vertices[i] for i in range(n)]

        for i in range(n):
            side_angle_perpendicular = side_angles[i] + np.pi / 2
            angles = np.linspace(side_angle_perpendicular - arc_angle, side_angle_perpendicular + arc_angle, num_points_per_side[i])
            x = np.append(x, np.cos(angles) * arc_radius + arc_x_centers[i])
            y = np.append(y, np.sin(angles) * arc_radius + arc_y_centers[i])

        # apply noise
        x = x + np.random.normal(0, noise_std, len(x))
        y = y + np.random.normal(0, noise_std, len(y))
        
        return x, y
        # return x, y, x_vertices, y_vertices, arc_x_centers, arc_y_centers # for debugging

    else:
        for i in range(n):
            x = np.append(x, np.linspace(x_vertices[i], x_vertices[i + 1], num_points_per_side[i]))
            y = np.append(y, np.linspace(y_vertices[i], y_vertices[i + 1], num_points_per_side[i]))

        # apply noise
        x = x + np.random.normal(0, noise_std, len(x))
        y = y + np.random.normal(0, noise_std, len(y))

        return x, y
        # return x, y, x_vertices, y_vertices # for debugging


def grid(num_points, n, dist, noise_std, seed=None):
    """
    Generate points on a grid with `n` vertices in each direction and add Gaussian noise. It also
    accepts a tuple or list for `n` and `dist` to specify different number of vertices and distance
    between them in each axis.

    :param n: number of vertices in each direction (int or tuple/list of 2 ints)
    :param dist: distance between vertices in each direction (float or tuple/list of 2 floats)
    :param num_points: number of points to generate (int)
    :param noise_std: standard deviation of the Gaussian noise (float)
    :param seed: random seed (int)
    :return: x and y coordinates of the points
    """
    if seed is not None:
        np.random.seed(seed)

    # if n is a tuple or list, unpack it
    if isinstance(n, int):
        n_x = n_y = n
    else:
        try:
            n_x, n_y = n
        except ValueError as e:
            raise ValueError('n must be an int or a tuple/list of 2 ints') from e

    # if dist is a tuple or list, unpack it
    if isinstance(dist, (float, int)):
        dist_x = dist_y = dist
    else:
        try:
            dist_x, dist_y = dist
        except ValueError as e:
            raise ValueError('dist must be a float/int or a tuple/list of 2 floats/ints') from e
        
    
    # calculate number of points per vertex
    n_per_vertex = [num_points // (n_x*n_y)] * (n_x*n_y)
    # distribute the remaining points
    n_per_vertex = [n_per_vertex[i] + 1 for i in range(num_points % (n_x*n_y))] + n_per_vertex
    # generate the grid
    x = np.array([])
    y = np.array([])
    for i in range(n_x*n_y):
        # calculate the vertex position around (0, 0)
        x_vertex = dist_x * (i % n_x - n_x/2 + 0.5)
        y_vertex = dist_y * (i // n_x - n_y/2 + 0.5)
        # use 2d gaussian to distribute points around the vertex
        x = np.append(x, np.random.normal(x_vertex, noise_std, n_per_vertex[i]))
        y = np.append(y, np.random.normal(y_vertex, noise_std, n_per_vertex[i]))

    return x, y


def moons(num_points, radius, noise_std, angle=0, shift = None, seed=None):
    """
    Generate points on a double moon shape with the specified parameters.

    """
    if seed is not None:
        np.random.seed(seed)

    if shift is None:
        shift_x = shift_y = radius/2
    elif isinstance(shift, (int, float)):
        shift_x = shift_y = shift
    else: 
        try:
            shift_x, shift_y = shift
        except ValueError as e:
            print("shift must be a float/int or a tuple of two floats/ints")
            raise e

    # compute number of points in each half circle
    n_points_out = int(num_points / 2)
    n_points_inn = num_points - n_points_out
    
    outer_circ_x = radius * np.cos(np.linspace(0, np.pi, n_points_out)) - shift_x
    outer_circ_y = radius * np.sin(np.linspace(0, np.pi, n_points_out)) - shift_y/2
    inner_circ_x = radius * np.cos(np.linspace(np.pi, 2*np.pi, n_points_out)) + shift_x
    inner_circ_y = radius * np.sin(np.linspace(np.pi, 2*np.pi, n_points_out)) + shift_y/2

    # append the two half circles and merge them into moon shapes
    x = np.vstack([np.append(outer_circ_x, inner_circ_x)])
    y = np.vstack([np.append(outer_circ_y, inner_circ_y)])

    # add noise
    x += np.random.normal(0, noise_std, size=x.shape)
    y += np.random.normal(0, noise_std, size=y.shape)

    if angle != 0:
        raise NotImplementedError("Rotation not implemented yet")
        # convert angle to radians
        angle = angle * np.pi / 180
        # rotate the moons
        x = x * np.cos(angle) - y * np.sin(angle)
        y = x * np.sin(angle) + y * np.cos(angle)
        
    return x, y


def bad_swiss_roll(num_points, noise_std, hole=True, seed=None):
    """
    This is a reimplementation of the swiss roll from scikit-learn, but we 
    didn't like it so we made one from scratch below.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 1.5 * np.pi, num_points)
    
    if not hole:
        y = 21 * np.random.rand(num_points)
    else:
        corners = np.array([[np.pi * (1.5 + i), j * 7] for i in range(3) for j in range(3)])
        corners = np.delete(corners, 4, axis=0)
        corner_index = np.random.choice(8, num_points)
        parameters = np.random.rand(2, num_points) * np.array([[np.pi], [7]])
        t, y = corners[corner_index].T + parameters
    
    x = t * np.cos(t)
    y = t * np.sin(t)

    # add noise
    x += np.random.normal(0, noise_std, size=x.shape)
    y += np.random.normal(0, noise_std, size=y.shape)

    return x, y

def swiss_roll(num_points, radius1, radius2, num_revolutions, noise_std, angle=0, seed=None):
    """
    Generate points on a swiss roll shape with the specified parameters.

    :param radius1: radius of the first circle - outer starting point (float)
    :param radius2: radius of the second circle - inner starting point (float)
    note: you can control the rotation of the swiss roll by changing the ratio between 
        radius1 and radius2. If radius1 > radius2, the swiss roll will have a positive 
        rotation (counterclockwise) and vice versa.
    :param num_revolutions: number of revolutions (float)
    :param num_points: number of points to generate (int)
    :param noise_std: standard deviation of the Gaussian noise (float)
    :param angle: angle of rotation of the swiss roll (float)
    :param seed: random seed (int)
    :return: x and y coordinates of the points
    """
    if seed is not None:
        np.random.seed(seed)

    angles = np.linspace(0, num_revolutions * 2 * np.pi, num_points)
    radius_decay = (radius1 - radius2)*angles / (num_revolutions * 2 * np.pi)
    x = (radius1 - radius_decay) * np.cos(angles)
    y = (radius1 - radius_decay) * np.sin(angles)

    # add noise
    x += np.random.normal(0, noise_std, size=x.shape)
    y += np.random.normal(0, noise_std, size=y.shape)

    if angle != 0:
        raise NotImplementedError("Rotation not implemented yet")
        # convert angle to radians
        angle = angle * np.pi / 180
        # rotate the swiss roll
        x = x * np.cos(angle) - y * np.sin(angle)
        y = x * np.sin(angle) + y * np.cos(angle)

    return x, y



def swirl(num_points, radius, noise_std, angle=0, seed=None):
    """
    Generate points on a swirl shape with the specified parameters.

    :param radius: radius of the swirl (float)
    :param num_points: number of points to generate (int)
    :param noise_std: standard deviation of the Gaussian noise (float)
    :param angle: angle of rotation of the swirl (float)
    :param seed: random seed (int)
    :return: x and y coordinates of the points
    """
    if seed is not None:
        np.random.seed(seed)

    # generate 2 swiss rolls with opposite rotations
    num_points1 = int(num_points / 2)
    num_points2 = num_points - num_points1
    x1, y1 = swiss_roll(radius1=radius, radius2=0, num_revolutions=1, num_points=num_points1, noise_std=noise_std, seed=seed)
    x2, y2 = swiss_roll(radius1=0, radius2=radius, num_revolutions=1, num_points=num_points2, noise_std=noise_std, seed=seed)

    # flip the second swiss roll accross the x-axis
    x2 = -x2

    # concatenate the 2 swiss rolls
    x = np.vstack([x1, x2]).flatten()
    y = np.vstack([y1, y2]).flatten()

    return x, y


def hline(num_points, noise_std, length=1, seed=None):
    """
    Generate a centered horizontal line with Gaussian noise in the vertical direction, with adjustable length.

    :param num_points: Number of points to generate (int)
    :param length: Length of the line (float)
    :param std: Standard deviation of the Gaussian noise (float)
    :param seed: Random seed for reproducibility (int, optional)
    :return: x and y numpy arrays representing the point cloud
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate x coordinates evenly spaced, centered at 0
    x = np.linspace(-length / 2, length / 2, num_points)
    
    # Generate y coordinates as Gaussian noise around 0
    y = np.random.normal(0, std, num_points)
    
    return x, y
