import argparse
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split(".")))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"
import numpy as np
import os
import pickle as pkl
from sklearn.metrics import accuracy_score, f1_score
import random
import time
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from encoders import Encoder
from aggregators import MeanAggregator


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t(), embeds

    def loss(self, nodes, labels):
        scores, intermediate = self.forward(nodes)
        return self.xent(scores, labels.squeeze()), intermediate


def load_cora(feat_addition, rwrreg_without_feat_addition):
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            
    ###########################################################
    # Structural Information Injection: add adjacency matrix, or RWR coefficients to the node features
    additional_feat = None
    if feat_addition in ["AD", "RW"]:
        G = nx.from_dict_of_lists(adj_lists)
        adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
        A = adj.todense()
        num_nodes = A.shape[0]     
        if feat_addition == "AD": # Concatenate Adjacency Matrix rows
            A = [np.array(A[i].tolist()[0]) for i in range(num_nodes)]
            A = np.array(A, dtype=np.float64) 
            additional_feat = A
            if not rwrreg_without_feat_addition:
                feat_data = np.hstack((feat_data, A))
        if feat_addition == "RW": # Concatenate Random Walk Matrix Rows
            rw_matrix_file = "cora_rw"
            if os.path.exists(rw_matrix_file):
                with open(rw_matrix_file, "rb") as rw_file:
                    X_rw = pkl.load(rw_file)
            else:
                X_rw = []
                for i in range(num_nodes):
                    ppr_from_node_i = nx.pagerank(G, personalization={x:(1 if x==i else 0) for x in range(num_nodes)})
                    rw_weights = []
                    for j in range(num_nodes):
                        rw_weights.append(ppr_from_node_i[j])
                    X_rw.append(np.array(rw_weights)) 
                X_rw = np.array(X_rw)
                with open(rw_matrix_file, "wb") as rw_file:
                    pkl.dump(X_rw, rw_file)
                X_rw = np.random.randn(*A.shape)
            additional_feat = X_rw
            if not rwrreg_without_feat_addition:
                feat_data = np.hstack((feat_data, X_rw))
    ###########################################################
       
    return feat_data, additional_feat,  labels, adj_lists


def run_cora(feat_addition, rwrreg_without_feat_addition, rwr_reg, lr=0.7, rwr_reg_term=1e-7):
    num_nodes = 2708
    feat_data, additional_feat, labels, adj_lists = load_cora(feat_addition, rwrreg_without_feat_addition)
    num_features = 1433
    if feat_addition in ["AD", "RW"] and not rwrreg_without_feat_addition:
        num_features += num_nodes
    features = nn.Embedding(2708, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, num_features, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=lr)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss, intermediate = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))

    ###########################################################
    # Compute RWRReg additional loss term (for more information check the Appendix of the paper)
        if rwr_reg:
            S = additional_feat
            S = S[batch_nodes, :]
            S = S[:, batch_nodes]
            n = S.shape[1]

            lapl_loss = 0
            if feat_addition == "RW": #make rw matrix symmetric
                for i in range(0, n):
                    for j in range(i, n):
                        if i == j:
                            continue
                        S[i, j] = S[i, j] + S[j, i]
                        S[j, i] = S[i, j]

            rowsums = S.sum(axis=1)
            D = np.diag(rowsums)

            delta = D - S
            delta = torch.from_numpy(delta).float()

            lapl_loss = torch.trace( torch.matmul( torch.matmul(intermediate, delta), intermediate.t()) )  
            loss += rwr_reg_term*lapl_loss
    ###########################################################

        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data, flush=True)

    val_output, _ = graphsage.forward(val) 
    val_acc = accuracy_score(labels[val], val_output.data.numpy().argmax(axis=1))
    val_f1_score = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print("Validation Accuracy:", val_acc)
    print("Validation F1:", val_f1_score)
    print("Average batch time:", np.mean(times))
    return val_acc, val_f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSage_RWRReg")
    parser.add_argument("--num-experiments", type=int, default=1, required=False,
                    help="Number of times to repeat trainnig and validation with randomly sampled data.")
    parser.add_argument("--feat-addition", type=str, default="No", required=False,
                    help="Structural information injection: one of 'NO, 'AD', 'RW'.")
    parser.add_argument("--rwr-reg", action="store_true", default=False, required=False,
                    help="Wether or not to add the regularization term to the loss function.")
    parser.add_argument("--rwrreg-without-feat-addition", action="store_true", default=False, required=False,
                    help="Regularization without feat addition.")
    args = parser.parse_args()

    np.random.seed(1)
    random.seed(1)
    accuracies = []
    f1_scores = []
    for iteration in range(args.num_experiments):
        acc, f1 = run_cora(args.feat_addition, args.rwrreg_without_feat_addition, args.rwr_reg)
        accuracies.append(acc)
        f1_scores.append(f1)
        
    if args.num_experiments > 1:
        print("--- Final Random Splits results ---")
        print("- Accuracy")
        print("Mean:", np.mean(np.array(accuracies)))
        print("Std:", np.std(np.array(accuracies)))
        print("- F1 Score")
        print("Mean:", np.mean(np.array(f1_scores)))
        print("Std:", np.std(np.array(f1_scores)))
