"""
Cieran: Designing Sequential Colormaps with a Teachable Robot
Copyright (C) 2023 Matt-Heun Hong

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np



def mean_and_slope_chroma(trajectory):
    """Returns the mean and slope of the chroma values of the trajectory via linear regression."""
    chroma = []
    for point in trajectory:
        chroma.append(np.sqrt(point[1]**2 + point[2]**2))

    x = np.array([l for l,a,b in trajectory])
    y = np.array(chroma)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])
    denominator = sum([(x[i] - x_mean)**2 for i in range(len(x))])
    slope = numerator / denominator

    return y_mean/100, slope/0.6


def distance(trajectory, degree, norm):
    """Returns the distance of the trajectory from the corner of the color space, normalized by the distance from the corner to the nearest visible color."""
    ab_values = [[point[1], point[2]] for point in trajectory]
    # Rotate ab_values by an amount of degree (in radians)
    ab_values = np.array(ab_values)
    ab_values = np.dot(ab_values, np.array([[np.cos(degree), -np.sin(degree)], [np.sin(degree), np.cos(degree)]]))
    
    corner = (-128, 127)
    dist = []
    for point in ab_values:
        dist.append(np.sqrt((corner[0] - point[0])**2 + (corner[1] - point[1])**2))
    return (min(dist) - norm) / (180 - norm)



def feature_func(trajectory):
    mean_c, slope_c = mean_and_slope_chroma(trajectory)

    corner1 = distance(trajectory, 0, 84.54)
    corner2 = distance(trajectory, np.pi/4, 120.63)
    corner3 = distance(trajectory, np.pi/2, 138.68)
    corner4 = distance(trajectory, 3*np.pi/4, 99.41)
    corner5 = distance(trajectory, np.pi, 66.66)
    corner6 = distance(trajectory, 5*np.pi/4, 102.23)
    corner7 = distance(trajectory, 3*np.pi/2, 90.01)
    corner8 = distance(trajectory, 7*np.pi/4, 97.90)

    return np.array([slope_c, corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8])
