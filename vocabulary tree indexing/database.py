import os
import numpy as np
import matplotlib.pyplot as plt
from .dataset import Dataset
import pickle
from . import utils


class Database(object):
    def __init__(self, dataset, encoder):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, str):
            self.dataset = Dataset(dataset)
        else:
            raise TypeError("Invalid dataset of type %s" % type(dataset))
        self.encoder = encoder

        self._database = {}
        self._image_ids = {}
        return

    def get_image_id(self, image_path):
        image_path = os.path.abspath(image_path)

        if image_path not in self._image_ids:
            self._image_ids[image_path] = hash(image_path)

        return self._image_ids[image_path]

    def create_index(self):
        times = []
        total = len(self.dataset.all_images)
        done = 0
        for i, image_path in enumerate(self.dataset.image_paths):
            image_id = self.get_image_id(image_path)
            image = self.dataset.read_image(image_path)
            start = time.time()
            self.encoder.propagate(image, image_id)
            times.append(time.time() - start)
            avg = np.mean(times)
            eta = avg * total - avg * (i + 1)
            done += 1
            
        N = len(self.dataset)
        for node_id, files in self.encoder.graph.nodes(data=True):
            N_i = len(files)
            if N_i:  
                self.encoder.graph.nodes[node_id]["w"] = np.log(
                    N / N_i)
        return

    def is_indexed(self, image_path):
        image_id = self.get_image_id(image_path)
        return image_id in self._database

    def embedding(self, image_path):
        image_id = self.get_image_id(image_path)
        if image_id not in self._database:
            image = self.dataset.read_image(image_path)
            self._database[image_id] = self.encoder.embedding(image)
        return self._database[image_id]

    def score(self, db_image_path, query_image_path):
        
        d = self.embedding(db_image_path)
        q = self.embedding(query_image_path)
        d = d / np.linalg.norm(d, ord=2)
        q = q / np.linalg.norm(q, ord=2)
        score = np.linalg.norm(d - q, ord=2)
        return score if not np.isnan(score) else 1e6

    def query_image(self, query_image_path, n=4):
        scores = {}
        for db_image_path in self.dataset.image_paths:
            scores[db_image_path] = self.score(db_image_path, query_image_path)

        sorted_scores = {k: v for k, v in sorted(
            scores.items(), key=lambda item: item[1])}
        return sorted_scores

    def save(self, path=None):
        if path is None:
            path = "data"

        with open(os.path.join(path, "index.pickle"), "wb") as f:
            pickle.dump(self._database, f)

        return True

    def load(self, path="data"):
        try:
            with open(os.path.join(path, "index.pickle"), "rb") as f:
                database = pickle.load(f)
                self._database = database
        except:
            print("Cannot load index file from %s/index.pickle" % path)
        return True

