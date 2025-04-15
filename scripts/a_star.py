# ================================ #
#  A* Search for TurtleBot3 Waffle
#  with Non-Holonomic Constraints
#  Author: [Your Name]
# ================================ #

import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
import cv2
from matplotlib import colors
import csv

# ------------------------------- #
#      User Input Parameters
# ------------------------------- #
clear = int(input("Enter robot clearance (in mm) = "))
s1 = int(input("Enter first angular velocity = "))
s2 = int(input("Enter second angular velocity = "))

# ------------------------------- #
#      Map and Robot Settings
# ------------------------------- #
scale = 200  # Pixels per meter
actual_width = 5.4  # meters
actual_height = 3.0  # meters

canvas_width = int(actual_width * scale)
canvas_height = int(actual_height * scale)

robot_radius_real = 0.22  # meters
robot_radius = robot_radius_real * scale
clear_process = clear * scale / 1000  # convert mm to pixels
c = int(robot_radius + clear_process)

# Resolution settings
XY_RESOLUTION = 30
THETA_RES = 30
NUM_THETA = 360 // THETA_RES

# Matplotlib colormap setup
cmap = colors.ListedColormap(['white', 'black', 'orange'])
bounds = [0, 0.5, 1.5, 2.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# ------------------------------- #
#      Obstacle Definitions
# ------------------------------- #
def point_inside_obstacle(x, y):
    """Check whether point lies inside defined obstacles."""
    if x < 0.1*scale or x > canvas_width - 0.1*scale or y < 0.1*scale or y > canvas_height - 0.1*scale:
        return True  # Border walls

    # Define walls as rectangles
    if (scale*1.0 <= x <= scale*1.1) and (0 <= y <= scale*2.4):
        return True
    if (scale*2.1 <= x <= scale*2.2) and (scale*0.6 <= y <= scale*3.0):
        return True
    if (scale*3.2 <= x <= scale*3.3):
        if (0 <= y <= scale*1.25) or (scale*1.75 <= y <= scale*3.0):
            return True
    if (scale*4.3 <= x <= scale*4.4) and (0 <= y <= scale*2.4):
        return True

    return False

def create_obstacles():
    """Generate map with defined obstacles."""
    map_img = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)
    obstacle_map = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    for y in range(canvas_height):
        for x in range(canvas_width):
            if point_inside_obstacle(x, y):
                obstacle_map[y, x] = 1
                map_img[y, x] = (0, 0, 0)

    return map_img, obstacle_map

def clearance_obstacles(clearance):
    """Generate map with obstacle clearance."""
    _, obstacle_mask = create_obstacles()
    kernel = np.ones((2 * clearance, 2 * clearance), np.uint8)
    clearance_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

    # Mark dilated space (clearance) as 2
    clearance_mask[(clearance_mask == 1) & (obstacle_mask == 0)] = 2
    clearance_mask[obstacle_mask == 1] = 1  # Ensure obstacles remain

    return clearance_mask

# ------------------------------- #
#        Robot Motion Model
# ------------------------------- #
def cost(Xi, Yi, Thetai, UL, UR, clearance_map):
    """Simulate differential drive motion and check for collisions."""
    r = 6.7   # wheel radius in mm
    L = 32    # wheel distance in mm
    dt = 0.1  # time step
    t = 0

    Xn = Xi
    Yn = Yi
    Thetan = math.radians(Thetai)
    D = 0

    path_x, path_y = [], []

    while t < 1:
        t += dt
        Delta_Xn = 0.5 * r * (UL + UR) * math.cos(Thetan) * dt
        Delta_Yn = 0.5 * r * (UL + UR) * math.sin(Thetan) * dt
        Xn += Delta_Xn
        Yn += Delta_Yn
        Thetan += (r / L) * (UR - UL) * dt
        D += math.hypot(Delta_Xn, Delta_Yn)
        path_x.append(Xn)
        path_y.append(Yn)

        x_map = int(Xn)
        y_map = int(Yn)

        if x_map < 0 or x_map >= canvas_width or y_map < 0 or y_map >= canvas_height:
            return 3
        if clearance_map[y_map, x_map] in [1, 2]:
            return 3

    Thetan = math.degrees(Thetan) % 360
    return Xn, Yn, Thetan, D, path_x, path_y

# ------------------------------- #
#        A* Heuristic
# ------------------------------- #
def heuristic(x, y, goal):
    """Euclidean distance as heuristic."""
    return math.hypot(goal[0] - x, goal[1] - y)

# ------------------------------- #
#           Node Class
# ------------------------------- #
class Node:
    def __init__(self, x, y, theta, g, h, parent=None, path_x=None, path_y=None, action=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.path_x = path_x or []
        self.path_y = path_y or []
        self.action = action

    def __lt__(self, other):
        return self.f < other.f

def state_to_index(x, y, theta):
    """Quantize the state for visited check."""
    x_idx = int(x / XY_RESOLUTION)
    y_idx = int(y / XY_RESOLUTION)
    theta_idx = int(theta // THETA_RES) % NUM_THETA
    return x_idx, y_idx, theta_idx

# ------------------------------- #
#         Path Reconstruction
# ------------------------------- #
def backtrack_path(node, ax):
    """Backtrack from goal to start."""
    path_x = []
    path_y = []

    while node.parent:
        ax.plot(node.path_x, node.path_y, color='red', linewidth=2)
        path_x = node.path_x + path_x
        path_y = node.path_y + path_y
        node = node.parent

    # Save the full path into CSV
    with open("path.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for x, y in zip(path_x, path_y):
            writer.writerow([x, y])

    return path_x, path_y

# ------------------------------- #
#         A* Search Function
# ------------------------------- #
def astar_with_matrix(start, goal, actions):
    """A* algorithm implementation with motion primitives."""
    clearance_map = clearance_obstacles(clearance=c)
    x_size = int(canvas_width / XY_RESOLUTION)
    y_size = int(canvas_height / XY_RESOLUTION)
    V = np.zeros((x_size, y_size, NUM_THETA), dtype=bool)

    open_list = []
    fig, ax = plt.subplots()
    ax.set_title("A* Non-Holonomic Search")
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)

    ax.imshow(clearance_map, cmap=cmap, norm=norm, origin='lower',
              extent=[0, canvas_width, 0, canvas_height], alpha=1.0)
    goal_radius = 50
    goal_circle = plt.Circle((goal[0], goal[1]), goal_radius, color='red', fill=False, linewidth=2)
    ax.add_patch(goal_circle)

    start_node = Node(start[0], start[1], start[2], 0, heuristic(start[0], start[1], goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current = heapq.heappop(open_list)

        if heuristic(current.x, current.y, goal) <= goal_radius:
            path_x, path_y = backtrack_path(current, ax)
            ax.plot(path_x, path_y, color='blue', linewidth=3, label="Final Path")
            ax.scatter(*goal, color='green', marker='X', s=100, label="Goal")
            ax.legend()
            plt.show()
            return current, ax

        x_idx, y_idx, theta_idx = state_to_index(current.x, current.y, current.theta)
        if V[x_idx, y_idx, theta_idx]:
            continue
        V[x_idx, y_idx, theta_idx] = True

        for action in actions:
            result = cost(current.x, current.y, current.theta, action[0], action[1], clearance_map)
            if result == 3:
                continue

            xn, yn, thetan, cost_to_come, path_x, path_y = result

            if not (0 <= xn < canvas_width and 0 <= yn < canvas_height):
                continue
            if clearance_map[int(yn), int(xn)] == 1:
                continue

            g_new = current.g + cost_to_come
            h_new = heuristic(xn, yn, goal)

            new_node = Node(xn, yn, thetan, g_new, h_new,
                            parent=current, path_x=path_x, path_y=path_y, action=action)
            heapq.heappush(open_list, new_node)

            ax.plot(path_x, path_y, color='gray', linewidth=0.4)

    print("Path not found")
    return None, ax

# ------------------------------- #
#           Main Routine
# ------------------------------- #
if __name__ == "__main__":
    actions = [[0, s1], [s1, 0], [s1, s1], [s2, 0],
               [0, s2], [s2, s2], [s2, s1], [s1, s2]]
    start = (100, 100, 0)  # Start position (x, y, theta)
    goal = (800, 400)      # Goal position (x, y)

    result, ax = astar_with_matrix(start, goal, actions)
