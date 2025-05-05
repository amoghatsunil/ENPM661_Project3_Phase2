import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
import cv2
from matplotlib import colors
import csv

# Define the colors: 0 = Free (white), 1 = Obstacle (black), 2 = Clearance (orange)
cmap = colors.ListedColormap(['white', 'black', 'orange'])
bounds = [0, 0.5, 1.5, 2.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

canvas_width = 1080
canvas_height = 600

XY_RESOLUTION = 30
THETA_RES = 30
NUM_THETA = 360 // THETA_RES

def create_obstacles():
    scale = 200
    map_img = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)
    obstacle_map = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    cv2.rectangle(map_img, (10, 10), (canvas_width - 10, canvas_height - 10), (0, 0, 0), 20)
    cv2.rectangle(obstacle_map, (10, 10), (canvas_width - 10, canvas_height - 10), 1, 20)

    x_starts_m = [1.0, 2.1, 3.2, 4.3]
    for x_m in x_starts_m:
        x_px = int(x_m * scale)
        wall_width = int(0.1 * scale)
        if x_m == 1.0 or x_m == 4.3:
            y_top = int(20)
            y_bottom = int(3.0 * scale - 0.5 * scale - 20)
            cv2.rectangle(map_img, (x_px, y_top), (x_px + wall_width, y_bottom), (0, 0, 0), -1)
            cv2.rectangle(obstacle_map, (x_px, y_top), (x_px + wall_width, y_bottom), 1, -1)
        if x_m == 2.1:
            y_top = int(0.5 * scale + 20)
            y_bottom = int(3.0 * scale)
            cv2.rectangle(map_img, (x_px, y_top), (x_px + wall_width, y_bottom), (0, 0, 0), -1)
            cv2.rectangle(obstacle_map, (x_px, y_top), (x_px + wall_width, y_bottom), 1, -1)
        if x_m == 3.2:
            y_top = int(20)
            y_bottom = int(250)
            cv2.rectangle(map_img, (x_px, y_top), (x_px + wall_width, y_bottom), (0, 0, 0), -1)
            cv2.rectangle(obstacle_map, (x_px, y_top), (x_px + wall_width, y_bottom), 1, -1)
            y_top2 = int(3.0 * scale - 1.25 * scale)
            y_bottom2 = int(3.0 * scale)
            cv2.rectangle(map_img, (x_px, y_top2), (x_px + wall_width, y_bottom2), (0, 0, 0), -1)
            cv2.rectangle(obstacle_map, (x_px, y_top2), (x_px + wall_width, y_bottom2), 1, -1)

    return map_img, obstacle_map

def clearance_obstacles(clearance=20):
    _, obstacle_mask = create_obstacles()
    kernel = np.ones((clearance, clearance), np.uint8)
    clearance_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
    clearance_mask[obstacle_mask == 1] = 1
    clearance_mask[0:clearance, :] = 1
    clearance_mask[-clearance:, :] = 1
    clearance_mask[:, 0:clearance] = 1
    clearance_mask[:, -clearance:] = 1
    return clearance_mask

def cost(Xi, Yi, Thetai, UL, UR, clearance_map):
    t = 0
    r = 8
    L = 70
    dt = 0.1
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
        if (x_map < 0 or x_map >= canvas_width or y_map < 0 or y_map >= canvas_height or
            clearance_map[y_map, x_map] == 1):
            return 3

    Thetan = math.degrees(Thetan) % 360
    return Xn, Yn, Thetan, D, path_x, path_y

def heuristic(x, y, goal):
    return math.hypot(goal[0] - x, goal[1] - y)

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
    x_idx = int(x / XY_RESOLUTION)
    y_idx = int(y / XY_RESOLUTION)
    theta_idx = int(theta // THETA_RES) % NUM_THETA
    return x_idx, y_idx, theta_idx

def backtrack_path(node, ax):
    path_x = []
    path_y = [] 

    while node.parent:
        ax.plot(node.path_x, node.path_y, color='red', linewidth=2)
        path_x = node.path_x + path_x
        path_y = node.path_y + path_y 
        node = node.parent

    # Write to CSV
    with open("path.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for x, y in zip(path_x, path_y):
            writer.writerow([x/200, y/200])

    return path_x, path_y

def astar_with_matrix(start, goal, actions):
    clearance_map = clearance_obstacles(clearance=20)
    x_size = int(canvas_width / XY_RESOLUTION)
    y_size = int(canvas_height / XY_RESOLUTION)
    V = np.zeros((x_size, y_size, NUM_THETA), dtype=bool)

    open_list = []
    fig, ax = plt.subplots()
    ax.set_title("A* Non-Holonomic Search")
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.grid(True)

    ax.imshow(clearance_map, cmap=cmap, norm=norm, origin='lower', extent=[0, canvas_width, 0, canvas_height], alpha=1.0)
    goal_radius = 50
    goal_circle = plt.Circle((goal[0], goal[1]), goal_radius, color='red', fill=False, linewidth=2)
    ax.add_patch(goal_circle)

    start_node = Node(start[0], start[1], start[2], 0, heuristic(start[0], start[1], goal))
    heapq.heappush(open_list, start_node)
    gem = 0

    while open_list:
        current = heapq.heappop(open_list)
        gem += 1

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
            if cost(current.x, current.y, current.theta, action[0], action[1], clearance_map) == 3:
                continue

            xn, yn, thetan, cost_to_come, path_x, path_y = cost(current.x, current.y, current.theta, action[0], action[1], clearance_map)

            if not (0 <= xn < canvas_width and 0 <= yn < canvas_height):
                continue

            x_map = int(xn)
            y_map = int(yn)
            if clearance_map[y_map, x_map] == 1:
                continue

            g_new = current.g + cost_to_come
            h_new = heuristic(xn, yn, goal)

            x_idx, y_idx, theta_idx = state_to_index(xn, yn, thetan)
            found_in_open_list = False
            for node_in_open_list in open_list:
                if node_in_open_list.x == xn and node_in_open_list.y == yn and node_in_open_list.theta == thetan:
                    found_in_open_list = True
                    if node_in_open_list.g > g_new:
                        node_in_open_list.g = g_new
                        node_in_open_list.f = g_new + h_new
                        node_in_open_list.parent = current
                        node_in_open_list.path_x = path_x
                        node_in_open_list.path_y = path_y
                        node_in_open_list.action = action
                    break

            if not found_in_open_list:
                new_node = Node(xn, yn, thetan, g_new, h_new, parent=current, path_x=path_x, path_y=path_y, action=action)
                heapq.heappush(open_list, new_node)

            ax.plot(path_x, path_y, color='gray', linewidth=0.4)

    print("Path not found")
    return None, ax

if __name__ == "__main__":
    actions = [[0,20],[20,0],[20, 20], [25, 0], [0, 25], [25, 25], [25,20], [20, 25]]
    start = (100, 100, 0)
    goal = (800, 400)

    result, ax = astar_with_matrix(start, goal, actions)