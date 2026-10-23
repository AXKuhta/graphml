from collections import Counter

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np

seeds = [4, 56, 66, 115, 155, 156, 230, 237, 254, 255]
N = 100

G1x = [nx.erdos_renyi_graph(N, .2, seed=seed) for seed in seeds]
G2x = [nx.barabasi_albert_graph(N, N//8, seed=seed) for seed in seeds]
G3x = [nx.watts_strogatz_graph(N, 20, .5, seed=seed) for seed in seeds]

x = G1x + G2x + G3x
y = [0]*len(G1x) + [1]*len(G2x) + [2]*len(G3x)

####################################################################################################
# Average feature embeddings
####################################################################################################

embs = []

for G in x:
	f1 = nx.average_clustering(G)
	f2 = nx.average_shortest_path_length(G)
	f3 = np.mean([v for k,v in nx.average_neighbor_degree(G).items()])

	embs.append(np.array([f1, f2, f3]))

model = PCA(2)
pts = model.fit_transform(embs)

plt.scatter(*pts.T, c=y)
plt.show()

####################################################################################################
# Embeddings of virtual node
####################################################################################################

embs = []

for G in x:
	G = G.copy()
	G.add_node("virtual")
	G.add_edges_from(("virtual", existing_node) for existing_node in G.nodes)

	f1 = nx.eigenvector_centrality(G)["virtual"]
	f2 = nx.pagerank(G)["virtual"]

	embs.append(np.array([f1, f2]))

model = PCA(2)
pts = model.fit_transform(embs)

plt.scatter(*pts.T, c=y)
plt.show()

####################################################################################################
# Anonymous walks
####################################################################################################

# TODO: rewrite for nx 3.7 random_walk
def anonymous_walks_histogram(graph):
  import random

  def next_node(G, previous, current, p, q):
    neighbors = list(G.neighbors(current))
    alphas = []
    for neighbor in neighbors:
      if neighbor == previous: alpha = 1/p
      elif G.has_edge(neighbor, previous): alpha = 1
      else: alpha = 1/q
      alphas.append(alpha)
    probs = [alpha / sum(alphas) for alpha in alphas]
    next = np.random.choice(neighbors, size=1, p=probs)[0]
    return next

  def random_walk(G, start, length, p, q):
    walk = [start]
    for i in range(length):
      current = walk[-1]
      previous = walk[-2] if len(walk) > 1 else None
      next = next_node(G, previous, current, p, q)
      walk.append(next)
    return [int(x) for x in walk]

  def random_anonymous_walk(G, start, length, p, q):
    walk = [0]
    encoder,enc_max={},0
    encoder[start] = 0
    for i in range(length):
      current = walk[-1]
      previous = walk[-2] if len(walk) > 1 else None
      next = next_node(G, previous, current, p, q)
      if next not in encoder:
        enc_max+=1
        encoder[next]=enc_max
      walk.append(encoder[next])
    return [int(x) for x in walk],encoder

  def histogram(G):
    walks = []

    for node in G.nodes:
      for _ in range(80):
        walk, code = random_anonymous_walk(G, node, 2,3,2)
        walk_str = "".join([str(x) for x in walk])
        walks.append(walk_str)

    cnt = Counter(walks)
    if "000" not in cnt: cnt["000"] = 0
    return np.array([cnt[k] for k in sorted(cnt)])

    plt.figure(figsize=(15,3))
    plt.hist(walks,bins=100)
    plt.yscale("log")
    plt.show()

  return histogram(graph)

embs = []

for G in x:
	embs.append(anonymous_walks_histogram(G))

model = PCA(2)
pts = model.fit_transform(embs)

plt.scatter(*pts.T, c=y)
plt.show()
