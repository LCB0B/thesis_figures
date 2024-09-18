import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import os
import matplotlib.colors as mcolors
import random
# set font to helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['helvetica']
#font size
plt.rcParams.update({'font.size': 12})

colors = ["#FF7DBE", "#FF9123", "#61C2FF", "#07AB92", "#FF7DBE", "#FF7DBE"]
main_color = '#DDDDDD'
second_color = colors[5]

cm = 1 / 2.54  # centimeters in inches

def darker(color, amount=0.5):
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

def lighter(color,amount=0.1):
    """
    lighten the given color."""
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    c = c + (white-c)*amount
    c = np.clip(c, 0, 1)  # Ensure RGB values are within [0, 1]
    return tuple(c)


# create 'figures' directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# generate a random graph
num_nodes = 20
G = nx.erdos_renyi_graph(n=num_nodes, p=0.3)

# plot the network
plt.figure(figsize=(10 * cm, 10 * cm))
pos = nx.spring_layout(G)

# choose one of the colors for nodes
node_color = main_color

# make a darker version of the node color for the node borders
node_edge_color = darker(node_color)

# draw nodes with specified edgecolors and node colors
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_color,
    node_size=300,
    edgecolors=node_edge_color,
    linewidths=2,
)

# draw edges in grey with increased width
nx.draw_networkx_edges(
    G,
    pos,
    edge_color='#C1C1C1',
    width=2.5
)

plt.axis('off')
plt.tight_layout()
plt.savefig('figures/network_plot.png', dpi=300)
plt.close()

#NON WEIGHTED 
# Create a black canvas
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, num_nodes)
ax.set_ylim(0, num_nodes)

# Set black background
ax.set_facecolor('black')

# Assign random weights to the edges
for (u, v, w) in G.edges(data=True):
    w['weight'] = random.random()

# Convert to a numpy array with weights
weighted_adj_matrix = nx.to_numpy_array(G, weight='weight')

# For each cell in the matrix, plot a centered white square scaled by the weight
for i in range(num_nodes):
    for j in range(num_nodes):
        weight = weighted_adj_matrix[i, j]
        if weight > 0:  # Only plot squares for edges with non-zero weight
            size = 0.9  # Scale the size by the weight (0 to 1)
            # Draw the square centered in the grid cell
            ax.add_patch(plt.Rectangle((j + 0.5 - size / 2, num_nodes - i - 1 + 0.5 - size / 2), 
                                       size, size, color='white'))

# Remove ticks and gridlines
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

plt.tight_layout()
plt.savefig('figures/adjacency_matrix.png', dpi=300)

# Create a black canvas
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, num_nodes)
ax.set_ylim(0, num_nodes)

# Set black background
ax.set_facecolor('black')
Gw = nx.erdos_renyi_graph(n=num_nodes, p=0.5)
# Assign random weights to the edges
for (u, v, w) in Gw.edges(data=True):
    w['weight'] = random.random()

# Convert to a numpy array with weights
weighted_adj_matrix = nx.to_numpy_array(Gw, weight='weight')

# For each cell in the matrix, plot a centered white square scaled by the weight
for i in range(num_nodes):
    for j in range(num_nodes):
        weight = weighted_adj_matrix[i, j]
        if weight > 0:  # Only plot squares for edges with non-zero weight
            size = weight  # Scale the size by the weight (0 to 1)
            # Draw the square centered in the grid cell
            ax.add_patch(plt.Rectangle((j + 0.5 - size / 2, num_nodes - i - 1 + 0.5 - size / 2), 
                                       size, size, color='white'))

# Remove ticks and gridlines
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

plt.tight_layout()
plt.savefig('figures/adjacency_matrix_w.png', dpi=300)

#directed
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, num_nodes)
ax.set_ylim(0, num_nodes)

# Set black background
ax.set_facecolor('black')


# Convert to a numpy array with weights
weighted_adj_matrix = nx.to_numpy_array(Gw, weight='weight')

# For each cell in the matrix, plot a centered white square scaled by the weight
for i in range(num_nodes):
    for j in range(num_nodes):
        weight = weighted_adj_matrix[i, j]
        if weight > 0:  # Only plot squares for edges with non-zero weight
            size = weight  # Scale the size by the weight (0 to 1)
            # Draw the square centered in the grid cell
            p = np.random.rand()
            if p < 0.6:
                ax.add_patch(plt.Rectangle((j + 0.5 - size / 2, num_nodes - i - 1 + 0.5 - size / 2), 
                                        size, size, color='white'))

# Remove ticks and gridlines
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

plt.tight_layout()
plt.savefig('figures/adjacency_matrix_dw.png', dpi=300)



# Plot the degree distribution using step (only outer edges)
degrees = [degree for node, degree in G.degree()]
plt.figure(figsize=(10 * cm, 7 * cm))

# Calculate histogram data
hist, bin_edges = np.histogram(degrees, bins=np.arange(1 - 0.5, max(degrees) + 1.5, 1))
hist = np.append(hist, 0)

#zero at he beginning of both arrays
bin_edges = np.insert(bin_edges, 0 , 0.5)
hist = np.insert(hist, 0, 0)
# Use step to plot the histogram outline (without filling the bars)
plt.step(bin_edges, hist, where='mid', color='black', linewidth=1)

# Set labels in lowercase
plt.xlabel('degree')
plt.ylabel('count')

# Remove the top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add all integer ticks from 1 to max degree, by step of 1
plt.xticks(range(1, max(degrees) + 1, 1))
plt.ylim(0, max(hist) * 1.1)
plt.tight_layout()
plt.savefig('figures/degree_distribution_.png', dpi=300)
plt.close()




#save the network as csv
nx.write_edgelist(G, "network.csv", delimiter=",", data=False)


def plot_shortest_path(G, source, target, pos=None, filename='figures/shortest_path.png'):
    """
    plots the shortest path between source and target nodes in the graph G.

    parameters:
    - G: networkx graph
    - source: source node
    - target: target node
    - pos: positions of nodes (optional). if not provided, spring_layout will be computed.
    - filename: filename to save the plot.
    """
    if pos is None:
        pos = nx.spring_layout(G)
    
    # find the shortest path
    try:
        path = nx.shortest_path(G, source=source, target=target)
    except nx.NetworkXNoPath:
        print(f"No path between node {source} and node {target}.")
        return
    
    # edges in the shortest path
    path_edges = list(zip(path, path[1:]))
    
    plt.figure(figsize=(10 * cm, 10 * cm))
    
    # draw all nodes
    node_color = main_color  # use the same color as before
    node_edge_color = darker(node_color)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=300,
        edgecolors=node_edge_color,
        linewidths=2,
    )
    
    # draw all edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#C1C1C1',
        width=2.5,
        alpha=0.5
    )
    
    # highlight the shortest path edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=path_edges,
        edge_color=second_color,  # highlight color
        width=3.0
    )
    
    # highlight the nodes along the shortest path
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=path,
        node_color=second_color,
        node_size=300,
        edgecolors=darker(second_color),
        linewidths=2
    )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
# example usage
plot_shortest_path(G, source=0, target=10, pos=pos)


def plot_clustering_coefficient(G, pos=None, filename='figures/clustering_coefficient.png'):
    """
    plots the network with nodes sized or colored based on their clustering coefficient.

    parameters:
    - G: networkx graph
    - pos: positions of nodes (optional). if not provided, spring_layout will be computed.
    - filename: filename to save the plot.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    # compute clustering coefficients
    clustering = nx.clustering(G)
    clustering_values = np.array(list(clustering.values()))
    
    # normalize clustering coefficients for visualization
    max_clustering = max(clustering_values)
    min_clustering = min(clustering_values)
    norm_clustering = (clustering_values - min_clustering) / (max_clustering - min_clustering + 1e-6)
    
    # map clustering coefficients to colors
    #create a cmap with the a color gradient form the list
    cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', [lighter(main_color,amount=0.7), main_color, darker(main_color)])
    node_colors = [cmap(value) for value in norm_clustering]

    plt.figure(figsize=(10 * cm, 10 * cm))

    # draw nodes with sizes proportional to clustering coefficient
    node_sizes = 300 + norm_clustering * 400  # adjust size scaling as needed
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=darker(main_color),
        linewidths=2
        
    )

    # draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#C1C1C1',
        width=1.5,
        alpha=0.7
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
plot_clustering_coefficient(G, pos=pos)


def create_lattice_network_with_figure(dimensions=(5, 5), periodic=False, filename='figures/lattice_network.png'):
    # create the lattice network
    rows, cols = dimensions
    G = nx.grid_2d_graph(rows, cols, periodic=periodic)
    # convert node labels from 2d tuples to integers (optional)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # compute positions based on grid coordinates for true lattice layout
    pos = {n: (n % cols, n // cols) for n in G.nodes()}

    # create 'figures' directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # plot the lattice network
    plt.figure(figsize=(10 * cm, 8 * cm))

    # choose one of the colors for nodes
    node_color = main_color 

    # make a darker version of the node color for the node borders
    node_edge_color = darker(node_color)

    # draw nodes with specified edgecolors and node colors
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=300,
        edgecolors=node_edge_color,
        linewidths=2,
    )

    # draw edges in grey with increased width
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#C1C1C1',
        width=2.5
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return G

create_lattice_network_with_figure(dimensions=(5, 5), periodic=False, filename='figures/lattice_network.png')


def create_small_world_network_with_figure(n=20, k=4, p=0.3, filename='figures/small_world_network.png'):
    # create the small-world network
    G = nx.watts_strogatz_graph(n, k, p)

    # compute positions using circular layout
    pos = nx.circular_layout(G)

    # create 'figures' directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # plot the small-world network
    plt.figure(figsize=(10 * cm, 10 * cm))

    # choose main color for nodes
    node_color = main_color

    # make a darker version of the node color for the node borders
    node_edge_color = darker(node_color)

    # draw nodes with specified edgecolors and node colors
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=300,
        edgecolors=node_edge_color,
        linewidths=2,
    )

    # draw edges in grey with increased width
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#C1C1C1',
        width=2.5
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return G

create_small_world_network_with_figure(n=20, k=4, p=0.3, filename='figures/small_world_network.png')


def create_ba_network_with_figure(n=30, m=2, filename='figures/ba_network.png'):
    # create the barabási-albert network
    G = nx.barabasi_albert_graph(n, m)

    # compute positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    #update POS so that the nodes are not overlapping
    pos = nx.kamada_kawai_layout(G)
    
    # create 'figures' directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # plot the ba network
    plt.figure(figsize=(10 * cm, 10 * cm))

    # choose main color for nodes
    node_color = main_color

    # make a darker version of the node color for the node borders
    node_edge_color = darker(node_color)

    # draw nodes with specified edgecolors and node colors
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=300,
        edgecolors=node_edge_color,
        linewidths=2,
    )

    # draw edges in grey with increased width
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#C1C1C1',
        width=2.5
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return G

create_ba_network_with_figure(n=30, m=1, filename='figures/ba_network.png')

def adjust_positions_no_overlap(pos, node_size, min_distance_factor=1.2, iterations=100):
    """
    Adjusts the positions of nodes to avoid overlap based on node size using pdist for efficient pairwise distance calculation.

    Parameters:
    - pos: dict, positions of nodes (node -> (x, y)).
    - node_size: float, the size of the nodes in the plot (radius).
    - min_distance_factor: float, multiplier for minimum distance between nodes (default is 1.2 for slight spacing).
    - iterations: int, number of iterations for adjustment (default is 100).

    Returns:
    - new_pos: dict, adjusted positions of nodes to avoid overlap.
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    # Convert node size to a distance measure in the position space (using radius)
    min_distance = min_distance_factor * np.sqrt(node_size)

    # Create a new copy of the positions to adjust
    new_pos = pos.copy()

    # Get the nodes and their positions
    nodes = list(new_pos.keys())
    positions = np.array(list(new_pos.values()))

    for _ in range(iterations):
        # Compute pairwise distances between nodes using pdist
        pairwise_distances = pdist(positions)
        
        # Convert to a square distance matrix
        dist_matrix = squareform(pairwise_distances)

        # Find nodes that are too close
        too_close = dist_matrix < min_distance
        
        # Iterate over node pairs that are too close
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if too_close[i, j]:  # If nodes are too close
                    # Compute the displacement vector
                    displacement = positions[i] - positions[j]
                    displacement_length = np.linalg.norm(displacement)
                    
                    # If displacement length is 0, use a random small displacement
                    if displacement_length == 0:
                        displacement = np.random.rand(2) - 0.5
                        displacement_length = np.linalg.norm(displacement)
                    
                    # Normalize the displacement vector
                    displacement_unit = displacement / displacement_length

                    # Move nodes apart equally
                    move_distance = (min_distance - displacement_length) / 2
                    positions[i] += move_distance * displacement_unit
                    positions[j] -= move_distance * displacement_unit

    # Update the node positions in the dictionary
    for idx, node in enumerate(nodes):
        new_pos[node] = positions[idx]

    return new_pos

def plot_high_modularity_network_with_communities(n=100, p_intra=0.8, p_inter=0.05, filename='figures/high_modularity_network.png'):
    """
    creates a network with high modularity (4 communities) and plots it with each block in a different color.

    parameters:
    - n: int, number of nodes in the network (divided into 4 communities).
    - p_intra: float, probability of edges within communities (high intra-community connectivity).
    - p_inter: float, probability of edges between communities (low inter-community connectivity).
    - filename: str, path to save the plotted figure.
    """
   
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Define the number of communities (4 blocks)
    communities = 4
    block_sizes = [n // communities] * communities  # Equal-sized blocks
    sizes = np.array(block_sizes)

    # Generate stochastic block model
    p_matrix = np.full((communities, communities), p_inter)  # Inter-community probability
    np.fill_diagonal(p_matrix, p_intra)  # Intra-community probability

    G = nx.stochastic_block_model(sizes, p_matrix)

    # Compute positions for plotting
    pos = nx.spring_layout(G, seed=42)
    pos = nx.kamada_kawai_layout(G)
    # make sure there is no overlapping 
    node_size = 300
    pos = adjust_positions_no_overlap(pos, node_size, min_distance_factor=int(1e-3), iterations=10)
    
    
    # Assign different colors to each block/community
    node_colors = []
    community_color_map = [colors[i % len(colors)] for i in range(communities)]
    
    # Assign colors based on the community
    for i, block_size in enumerate(block_sizes):
        node_colors.extend([community_color_map[i]] * block_size)

    # Plot the network
    plt.figure(figsize=(10 * cm, 10 * cm))

    # Draw nodes with their community colors
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors='black',
        linewidths=1.5
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#C1C1C1',
        width=1.5,
        alpha=0.7
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return G

plot_high_modularity_network_with_communities(n=50, p_intra=0.8, p_inter=0.02, filename='figures/high_modularity_network.png')


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_link_clustering_network(G, edge_communities, filename='link_clustering_network.png'):
    """
    Plots a network with edges colored according to their link communities and nodes as pie charts
    representing the proportion of their belonging to each community, with black node contours.
    
    Parameters:
    - G: networkx graph
    - edge_communities: dict, mapping of edges (tuples) to their link communities.
    - filename: str, path to save the plot.
    """
    # Set font and color scheme
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['helvetica']
    plt.rcParams.update({'font.size': 12})
    colors = ["#FF7DBE", "#FF9123", "#61C2FF", "#07AB92"]
    
    fig = plt.figure(figsize=(10*cm, 10*cm))
    # Max number of link communities
    max_communities = min(len(set(edge_communities.values())), 4)
    
    # Assign up to 4 colors to the communities
    community_colors = colors[:max_communities]
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    pos = nx.kamada_kawai_layout(G)
    # Create a dictionary to track node community membership (as pie chart proportions)
    node_community_pie = {node: np.zeros(max_communities) for node in G.nodes()}
    
    # Draw edges, coloring them based on their link community
    for edge, community in edge_communities.items():
        color = community_colors[community % max_communities]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color, width=2.5)

        # Add to the community pie chart for the nodes
        for node in edge:
            node_community_pie[node][community % max_communities] += 1
    
    # Normalize the community membership for nodes
    for node in node_community_pie:
        total = np.sum(node_community_pie[node])
        if total > 0:
            node_community_pie[node] /= total
    
    # Plot nodes as pie charts with black edges (contours)
    for node in G.nodes():
        pie_data = node_community_pie[node]
        if np.sum(pie_data) > 0:  # Avoid empty pies
            plot_pie_chart_as_node(pos[node], pie_data, community_colors)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_pie_chart_as_node(position, pie_data, colors, radius=0.09, node_edge_color=darker(main_color)):
    """
    Draws a pie chart at a given position on the plot to represent a node.
    
    Parameters:
    - position: tuple, (x, y) coordinates for the pie chart.
    - pie_data: list, proportions of each community for the node.
    - colors: list, colors corresponding to the pie segments (communities).
    - radius: float, radius of the pie chart.
    - node_edge_color: str, color of the node contour (edge).
    """
    ax = plt.gca()
    
    # Draw the pie chart with white borders
    # wedges, _ = ax.pie(pie_data, colors=colors, radius=radius)
    wedges, _ = ax.pie(pie_data, colors=colors, radius=radius, wedgeprops=dict(edgecolor='white', linewidth=1))
    
    
    # Draw a black contour around the pie chart
    circle = plt.Circle(position, radius=radius, edgecolor=node_edge_color, fill=False, lw=1.5)
    ax.add_artist(circle)
    
    # Move pie chart to the position of the node
    for w in wedges:
        w.set_center(position)

# Example of how you might generate link communities (using a placeholder method)
def generate_link_communities(G):
    """
    Placeholder function for generating link communities.
    Replace this with actual link clustering algorithm.
    
    Returns:
    - edge_communities: dict, mapping of edges to their communities.
    """
    # For example purposes, randomly assign edges to communities (replace with real algorithm)
    edge_communities = {}
    for i, edge in enumerate(G.edges()):
        edge_communities[edge] = i % 4  # Assign to one of 4 communities
    return edge_communities

# Example usage
G = nx.erdos_renyi_graph(n=20, p=0.4)
edge_communities = generate_link_communities(G)
plot_link_clustering_network(G, edge_communities, filename='figures/link_clustering_network.png')
