import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import os

# set font to helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['helvetica']

colors = ["#D55E00", "#FF9123", "#61C2FF", "#07AB92", "#014C4C", "#FF7DBE"]

cm = 1 / 2.54  # centimeters in inches

def darker(color, amount=0.7):
    """
    darken the given color.

    parameters:
    - color: str or tuple, color in any format matplotlib accepts.
    - amount: float, amount to darken (between 0 and 1).

    returns:
    - darker_color: tuple, rgb values of the darker color.
    """
    c = np.array(mcolors.to_rgb(color))
    return tuple(c * amount)


# # example usage
# # create 'figures' directory if it doesn't exist
# os.makedirs('figures', exist_ok=True)

# # generate a random graph
# G = nx.erdos_renyi_graph(n=20, p=0.3)

# # compute positions so they are consistent across plots
# pos = nx.spring_layout(G)

# # plot the shortest path between node 0 and node 10
# plot_shortest_path(G, source=0, target=10, pos=pos)

