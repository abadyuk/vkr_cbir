import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import MiniBatchKMeans
from dataset import Dataset


class VocabularyTree(object):
    def __init__(self, n_branches, depth, descriptor):
        self.n_branches = n_branches
        self.depth = depth
        self.descriptor = descriptor

        self.tree = {}
        self.nodes = {}
        self.graph = nx.DiGraph()

        self._current_index = 0
        self._propagated = set()

    def learn(self, dataset):
        features = self.extract_features(dataset)
        self.fit(features)
        return self

    def extract_features(self, dataset):
        print("Extracting features...")
        features = []
        for elem in dataset.image_paths:
            features.append(self.descriptor.embedding(dataset.read_image(elem)))
        return np.array(features)

    def fit(self, features, node=0, root=None, current_depth=0):
        if root is None:
            root = np.mean(features, axis=0)

        self.nodes[node] = root
        self.graph.add_node(node)

        if current_depth >= self.depth or len(features) < self.n_branches:
            return


        print("Computing clusters %d/%d with %d features from node %d at level %d\t\t" %
              (self._current_index, self.n_branches ** self.depth, len(features), node, current_depth),
              end="\r")
        model = MiniBatchKMeans(n_clusters=self.n_branches)
        model.fit(features)
        children = [[] for i in range(self.n_branches)]
        for i in range(len(features)):
            children[model.labels_[i]].append(features[i])


        self.tree[node] = []
        for i in range(self.n_branches):
            self._current_index += 1
            self.tree[node].append(self._current_index)
            self.graph.add_edge(node, self._current_index)
            self.fit(children[i], self._current_index,
                     model.cluster_centers_[i], current_depth + 1)
        return

    def propagate(self, image, image_id):
        if (image_id in self._propagated):
            return

        features = self.descriptor.embedding(image)
        for feature in features:
            path = self.propagate_feature(feature)
            for i in range(len(path)):
                node = path[i]
        
                if image_id not in self.graph.nodes[node]:
                    self.graph.nodes[node][image_id] = 1
                else:
                    self.graph.nodes[node][image_id] += 1
        self._propagated.add(image_id)
        return

    def propagate_feature(self, feature, node=0):
        path = [node]
        while self.graph.out_degree(node): 
            min_dist = float("inf")
            closest = None
            for child in self.graph[node]:
                distance = np.linalg.norm(
                    [self.nodes[child] - feature]) 
                if distance < min_dist:
                    min_dist = distance
                    closest = child
            path.append(closest)
            node = closest
        return path

    def embedding(self, image):
        self.propagate(image)

        image_id = self.dataset.get_image_id(image)

    
        embedding = np.array(self.graph.nodes(data=image_id, default=0))[:, 1]

        embedding = embedding / np.linalg.norm(embedding, ord=2) 

        return embedding

    def subgraph(self, image_id):
        subgraph = self.graph.subgraph(
            [k for k, v in self.graph.nodes(data=image_id, default=None) if v is not None])
        colours = ["C0"] * len(self.graph.nodes)
        for node in subgraph.nodes:
            colours[node] = "C3"
        self.draw(node_color=colours)
        return subgraph

    def save(self, path=None):
        if path is None:
            path = "data"


        nx.write_gpickle(self.graph, os.path.join(path, "graph.pickle"))

        with open(os.path.join(path, "nodes.pickle"), "wb") as f:
            pickle.dump(self.nodes, f)

        return True

    def draw(self, figsize=None, node_color=None, layout="tree", labels=None):
        figsize = (30, 10) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        layout = layout.lower()
        if "tree" in layout:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="dot")
        elif "radial" in layout:
            pos = nx.drawing.nx_agraph.graphviz_layout(
                self.graph, prog="twopi")
        else:
            pos = None
        if labels is None:
            nx.draw(self.graph, pos=pos, with_labels=True,
                    node_color=node_color)
        else:
            nx.draw(self.graph, pos=pos, labels=labels, node_color=node_color)
        return fig
