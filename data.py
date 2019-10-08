import os
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors


class EdgeSampler(Sampler):

    def __init__(self, edges, edge_freq, smoothing_rate_for_edge, edge_table_size, node_freq, verbose=True):
        self.edge_num = len(edges)
        self.edge_freq = edge_freq
        self.smoothing_rate_for_edge = smoothing_rate_for_edge
        self.edge_table_size = int(edge_table_size)

        c = self.edge_freq ** self.smoothing_rate_for_edge

        self.sample_edge_table = np.zeros(self.edge_table_size, dtype=int)
        index = 0
        p = c / c.sum()
        d = p[index]
        for i in tqdm(range(self.edge_table_size)):
            self.sample_edge_table[i] = index
            if i / self.edge_table_size > d:
                index += 1
                d += p[index]
            if index >= self.edge_num:
                index = self.edge_num - 1;

    def __iter__(self):
        return (self.sample_edge_table[i] for i in torch.randperm(self.edge_table_size))

    def __len__(self):
        return self.edge_table_size


class GraphDataset(Dataset):
    ntries = 10
    smoothing_rate_for_edge = 1.0
    node_table_size = int(1e7)
    edge_table_size = int(5e7)

    def __init__(self, node2id, id2freq, edges2freq, nnegs, smoothing_rate_for_node, data_vectors=None, task="reconst", seed=0):
        assert task in ["reconst", "linkpred"]

        self.smoothing_rate_for_node = smoothing_rate_for_node
        self.nnegs = nnegs
        self.task = task

        if task == "linkpred":
            assert data_vectors is not None
            train_node, test_node = train_test_split(list(node2id.keys()), test_size=0.2, random_state=seed)
            train_node, valid_node = train_test_split(train_node, test_size=0.2, random_state=seed)
            train_node_set = set(train_node)
            node_freq = list()
            valid_node_set = set(valid_node)
            test_node_set = set(test_node)
            print(f"len(train_node) : {len(train_node)}, len(valid_node) : {len(valid_node)}, len(test_node) : {len(test_node)}")
            new_node2id = defaultdict(lambda: len(new_node2id))
            new_data_vectors = np.empty(data_vectors.shape)
            for i in train_node:
                new_data_vectors[new_node2id[i]] = data_vectors[node2id[i]]
                node_freq.append(id2freq[node2id[i]])
            for i in valid_node: new_data_vectors[new_node2id[i]] = data_vectors[node2id[i]]
            for i in test_node: new_data_vectors[new_node2id[i]] = data_vectors[node2id[i]]
            new_node2id = dict(new_node2id)

            id2node = dict((y,x) for x,y in node2id.items())
            train_edges = list()
            edge_freq = list()

            neighbor_train = defaultdict(lambda: set())
            neighbor_valid = defaultdict(lambda: set())
            neighbor_test = defaultdict(lambda: set())

            for edge, edgefreq in edges2freq.items():
                i, j = [id2node[k] for k in edge]
                if i in train_node_set and j in train_node_set:
                    train_edges.append((new_node2id[i],new_node2id[j]))
                    edge_freq.append(edgefreq)
                    neighbor_train[new_node2id[i]].add(new_node2id[j])
                    neighbor_train[new_node2id[j]].add(new_node2id[i])
                else:
                    if i in test_node_set or j in test_node_set:
                        if i in test_node_set:
                            neighbor_test[new_node2id[i]].add(new_node2id[j])
                        if j in test_node_set:
                            neighbor_test[new_node2id[j]].add(new_node2id[i])
                    else:
                        if i in valid_node_set:
                            neighbor_valid[new_node2id[i]].add(new_node2id[j])
                        if j in valid_node_set:
                            neighbor_valid[new_node2id[j]].add(new_node2id[i])

            train_edges = np.array(train_edges, dtype=np.int)
            neighbor_train = dict(neighbor_train)
            neighbor_valid = dict(neighbor_valid)
            neighbor_test = dict(neighbor_test)

            self.node2id = new_node2id
            self.data_vectors = new_data_vectors
            self.total_node_num = len(node2id)
            self.train_node_num = len(train_node)
            self.edges = train_edges
            self.total_edge_num = len(edges2freq)
            self.train_edge_num = len(self.edges)
            self.node_freq = np.array(node_freq, dtype=np.float)
            self.edge_freq = np.array(edge_freq, dtype=np.float)
            self.max_tries = self.nnegs * self.ntries
            self.neighbor_train = neighbor_train
            self.neighbor_valid = neighbor_valid
            self.neighbor_test = neighbor_test

        elif task == "reconst":
            self.node2id = node2id
            self.data_vectors = data_vectors
            self.total_node_num = len(node2id)
            self.train_node_num = self.total_node_num
            self.edges = list()
            self.edge_freq = list()
            for edge, freq in edges2freq.items():
                self.edges.append(edge)
                self.edge_freq.append(freq)
            self.edges = np.array(self.edges, dtype=np.int)
            self.edge_freq = np.array(self.edge_freq, dtype=np.float)
            self.total_edge_num = len(self.edges)
            self.train_edge_num = self.total_edge_num
            self.node_freq = np.zeros(self.train_node_num, dtype=np.float)
            for i,f in id2freq.items():
                self.node_freq[i] = f
            self.max_tries = self.nnegs * self.ntries
            self.neighbor_train = defaultdict(lambda: set())
            for i, j in self.edges:
                self.neighbor_train[i].add(j)
                self.neighbor_train[j].add(i)
            self.neighbor_train = dict(self.neighbor_train)
            self.neighbor_valid = None
            self.neighbor_test = None
            assert len(self.neighbor_train) == self.train_node_num

        c = self.node_freq ** self.smoothing_rate_for_node

        self.sample_node_table = np.zeros(self.node_table_size, dtype=int)
        index = 0
        p = c / c.sum()
        d = p[index]
        for i in tqdm(range(self.node_table_size)):
            self.sample_node_table[i] = index
            if i / self.node_table_size > d:
                index += 1
                d += p[index]
            if index >= self.train_node_num:
                index = self.train_node_num - 1;

        self.sampler = EdgeSampler(self.edges, self.edge_freq, self.smoothing_rate_for_edge, self.edge_table_size, self.node_freq)

    def __len__(self):
        return self.train_edge_num

    def __getitem__(self, i):
        i, j = [int(x) for x in self.edges[i]]
        if np.random.randint(2) == 1:
            i, j = j, i

        negs = set()
        ntries = 0
        nnegs = self.nnegs
        while ntries < self.max_tries and len(negs) < nnegs:
            n = np.random.randint(0, self.node_table_size)
            n = int(self.sample_node_table[n])
            if n != i and n != j:
                negs.add(n)
            ntries += 1
        ix = [i, j] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[np.random.randint(2, len(ix))])

        return torch.LongTensor(ix).view(1, len(ix)), torch.zeros(1).long()

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return Variable(torch.cat(inputs, 0)), Variable(torch.cat(targets, 0))


def preprocess_hirearchy(word2vec_path, use_rich_information=False, verbose=True):
    if word2vec_path is not None:
        assert use_rich_information == False
    def _clean(word):
        if use_rich_information :
            return word
        else:
            word = word.split(".n.")[0]
            word = word.lower()
            return word

    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    word2vec_vocab = set(word2vec.vocab.keys())
    word2data_vector = dict()
    for word in word2vec_vocab:
        lowerword = word.lower()
        if lowerword in word2vec_vocab:
            word2data_vector[lowerword] = word2vec[lowerword]
        else:
            word2data_vector[lowerword] = word2vec[word]

    word2id = defaultdict(lambda: len(word2id))
    id2freq = defaultdict(lambda: 0)
    edges2freq = defaultdict(lambda: 0)

    def _memo(word1, word2):
        if word1 in word2data_vector and word2 in word2data_vector:
            id_1 = word2id[word1]
            id_2 = word2id[word2]
            id2freq[id_1] += 1
            id2freq[id_2] += 1
            if id_1 > id_2:
                id_1, id_2 = id_2, id_1
            edges2freq[(id_1, id_2)] = 1

    if verbose: pbar = tqdm(total = len(list(wn.all_synsets(pos='n'))))
    for synset in wn.all_synsets(pos='n'):
        if verbose: pbar.update(1)
        for hyper in synset.closure(lambda s: s.hypernyms()):
            word1 = _clean(hyper.name());word2 = _clean(synset.name());_memo(word1, word2)
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                word1 = _clean(hyper.name());word2 = _clean(instance.name());_memo(word1, word2)
                for h in hyper.closure(lambda s: s.hypernyms()):
                    word1 = _clean(h.name());_memo(word1, word2)
    if verbose: pbar.close()

    word2id = dict(word2id)
    id2freq = dict(id2freq)
    edges2freq = dict(edges2freq)
    vectors = np.empty((len(word2id), 300))
    for word, index in word2id.items():
        vectors[index] = word2data_vector[word]

    print(f"""Node num : {len(word2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return word2id, id2freq, edges2freq, vectors


def iter_line(fname, sep='\t', type=tuple, comment='#', return_idx=False, convert=None):
    with open(fname, 'r') as fin:
        if return_idx: index = -1
        for line in fin:
            if line[0] == comment:
                continue
            if convert is not None:
                d = [convert(i) for i in line.strip().split(sep)]
            else:
                d = line.strip().split(sep)
            out = type(d)
            if out is not None:
                if return_idx:
                    index += 1
                    yield (index, out)
                else:
                    yield out


def preprocess_co_author_network(dir_path, undirect=True, seed=0):
    author2id = defaultdict(lambda: len(author2id))
    edges2freq = dict()
    for _i, _j in iter_line(dir_path + "/graph_dblp.txt", sep='\t', type=tuple, convert=int):
        i=author2id[_i];j=author2id[_j]
        if i > j: j, i = i, j
        edges2freq[(i,j)] = 1

    author2id = dict(author2id)

    vectors = np.empty((len(author2id), 33))
    selected_attributes = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 37])
    for i, vec in iter_line(dir_path + "/db_normalz_clus.txt", sep=',', type=np.array, convert=float, return_idx=True):
        assert vec.shape[0] == 38
        if i in author2id:
            vec = vec.astype(np.float32)[selected_attributes]
            vectors[author2id[i]] = vec

    id2freq = defaultdict(lambda: 0)
    for key, value in edges2freq.items():
        id2freq[key[0]] += value
        id2freq[key[1]] += value
    id2freq = dict(id2freq)

    print(f"""Node num : {len(author2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return author2id, id2freq, edges2freq, vectors


def preprocess_webkb_network(dir_path):
    node2id = defaultdict(lambda: len(node2id))
    edges2freq = dict()
    for _i, _j in iter_line(dir_path + "/WebKB.cites", sep='\t', type=tuple, convert=str):
        i = node2id[_i];j = node2id[_j]
        if i > j: j, i = i, j
        edges2freq[(i,j)] = 1
    node2id = dict(node2id)
    id2freq = defaultdict(lambda: 0)
    for key, value in edges2freq.items():
        id2freq[key[0]] += value
        id2freq[key[1]] += value
    id2freq = dict(id2freq)
    vectors = np.empty((len(node2id), 1703), dtype=np.float)
    lines = open(dir_path + "/WebKB.content").readlines()
    for line in lines:
        elements = line.strip().split()
        node = str(elements[0])
        vec = np.array([int(i) for i in elements[1:-1]], dtype=np.float)
        assert len(vec) == 1703
        vectors[node2id[node]] = vec

    print(f"""Node num : {len(node2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return node2id, id2freq, edges2freq, vectors
