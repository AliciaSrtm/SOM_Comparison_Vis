import os
import numpy as np
import matplotlib.pyplot as plt
from somoclu import Somoclu
import networkx as nx
from prov.model import ProvDocument
import gzip
import holoviews as hv
from holoviews import opts

hv.extension('bokeh')


# Step 1: Load Dataset
def load_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    return data


# Step 2: Train SOM
def train_som(data, map_size=(10, 10), epochs=100, init_radius=2, learning_rate=0.1):
    som = Somoclu(map_size[0], map_size[1])
    som.train(data=data, epochs=epochs, radius0=init_radius, scale0=learning_rate)
    return som


# Step 3: Provenance Tracking
def generate_provenance(visualization, params):
    prov = ProvDocument()
    entity = prov.entity(f"ex:{visualization}", {"ex:params": str(params)})
    prov_str = prov.serialize(format='json')
    with open(f"provenance_{visualization}.json", "w") as f:
        f.write(prov_str)
    return prov_str


# Step 4: Compare SOMs
def compare_soms(som1, som2, threshold=2):
    grid_size = som1.codebook.shape[:2]
    shift_graph = nx.DiGraph()

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            shift = np.linalg.norm(som1.codebook[i, j] - som2.codebook[i, j])
            if shift > threshold:
                shift_graph.add_edge((i, j), (i, j), weight=shift)
    return shift_graph


# Step 5: Visualization of Shift Arrows
def plot_shift_arrows(shift_graph):
    pos = {node: node for node in shift_graph.nodes()}
    edges, weights = zip(*nx.get_edge_attributes(shift_graph, 'weight').items())
    nx.draw(shift_graph, pos, edges=edges, width=np.array(weights) / max(weights) * 5, with_labels=True)
    plt.show()


# Step 6: Hit Histogram
def hit_histogram(m, n, weights, idata):
    hist = np.zeros(m * n)
    for vector in idata:
        position = np.argmin(np.sqrt(np.sum(np.power(weights - vector, 2), axis=1)))
        hist[position] += 1
    return hist.reshape(m, n)


# Step 7: U-Matrix
def u_matrix(m, n, weights, dim):
    U = weights.reshape(m, n, dim)
    U = np.insert(U, np.arange(1, n), values=0, axis=1)
    U = np.insert(U, np.arange(1, m), values=0, axis=0)
    for i in range(U.shape[0]):
        if i % 2 == 0:
            for j in range(1, U.shape[1], 2):
                U[i, j][0] = np.linalg.norm(U[i, j - 1] - U[i, j + 1], axis=-1)
        else:
            for j in range(U.shape[1]):
                if j % 2 == 0:
                    U[i, j][0] = np.linalg.norm(U[i - 1, j] - U[i + 1, j], axis=-1)
                else:
                    U[i, j][0] = (np.linalg.norm(U[i - 1, j - 1] - U[i + 1, j + 1], axis=-1) + np.linalg.norm(
                        U[i + 1, j - 1] - U[i - 1, j + 1], axis=-1)) / (2 * np.sqrt(2))
    U = np.sum(U, axis=2)
    for i in range(0, U.shape[0], 2):
        for j in range(0, U.shape[1], 2):
            region = []
            if j > 0: region.append(U[i][j - 1])
            if i > 0: region.append(U[i - 1][j])
            if j < U.shape[1] - 1: region.append(U[i][j + 1])
            if i < U.shape[0] - 1: region.append(U[i + 1][j])
            U[i, j] = np.median(region)
    return U


# Step 8: SDH
def SDH(m, n, weights, idata, factor, approach):
    import heapq
    sdh_m = np.zeros(m * n)
    cs = sum(factor - i for i in range(factor))

    for vector in idata:
        dist = np.sqrt(np.sum(np.power(weights - vector, 2), axis=1))
        c = heapq.nsmallest(factor, range(len(dist)), key=dist.__getitem__)
        if approach == 0:
            for j in range(factor): sdh_m[c[j]] += (factor - j) / cs
        elif approach == 1:
            for j in range(factor): sdh_m[c[j]] += 1.0 / dist[c[j]]
        elif approach == 2:
            dmin, dmax = min(dist[c]), max(dist[c])
            for j in range(factor): sdh_m[c[j]] += 1.0 - (dist[c[j]] - dmin) / (dmax - dmin)
    return sdh_m.reshape(m, n)


# Step 9: Visualization
if __name__ == "__main__":
    dataset_path = "chainlink.txt"
    data = load_dataset(dataset_path)

    som1 = train_som(data, map_size=(10, 10))
    som2 = train_som(data, map_size=(10, 10))

    generate_provenance("SOM_Training", {"size": (10, 10), "epochs": 100, "learning_rate": 0.1})

    shift_graph = compare_soms(som1, som2)
    plot_shift_arrows(shift_graph)

    hithist = hv.Image(hit_histogram(som1.codebook.shape[0], som1.codebook.shape[1], som1.codebook, data)).opts(
        xaxis=None, yaxis=None)
    um = hv.Image(u_matrix(som1.codebook.shape[0], som1.codebook.shape[1], som1.codebook, 4)).opts(xaxis=None,
                                                                                                   yaxis=None)
    sdh = hv.Image(SDH(som1.codebook.shape[0], som1.codebook.shape[1], som1.codebook, data, 25, 0)).opts(xaxis=None,
                                                                                                         yaxis=None)

    hv.Layout([hithist.relabel('HitHist').opts(cmap='kr'),
               um.relabel('U-Matrix').opts(cmap='jet'),
               sdh.relabel('SDH').opts(cmap='viridis')])