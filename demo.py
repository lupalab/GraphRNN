from train import *
from utils import *
import pickle as pkl
import random
import networkx as nx
import glob
import numpy as np

args = Args()


# Directories of the graphs.
DIR_STAR_GEN = args.graph_save_path+'star_16/GraphRNN_RNN_generated_4_128__100_1.dat'
DIR_GRID_GEN = args.graph_save_path+'grid_16/GraphRNN_RNN_generated_4_128__700_1.dat'
DIR_CHAIN_GEN = args.graph_save_path+'chain_16/GraphRNN_RNN_generated_4_128__100_1.dat'
DIR_STAR = './generated_graphs/star_graphs_16.pkl'
DIR_GRID = './generated_graphs/grid_graphs_16.pkl'
DIR_CHAIN = './generated_graphs/chain_graphs_16.pkl'

# Load list of saved graphs.
GL_GN_star = load_graph_list(DIR_STAR_GEN)
GL_GN_grid = load_graph_list(DIR_GRID_GEN)
GL_GN_chain = load_graph_list(DIR_CHAIN_GEN)

# Load graphs in dataset.
A_DS_star = pkl.load(open(DIR_STAR,'rb'), encoding='latin-1')['data']
A_DS_grid = pkl.load(open(DIR_GRID, 'rb'), encoding='latin-1')['data']
A_DS_chain = pkl.load(open(DIR_CHAIN, 'rb'), encoding='latin-1')['data']

# Get adjacency matrix from generated graphs.
A_GN_star = np.stack([nx.adjacency_matrix(graph) for graph in GL_GN_star])
A_GN_grid = np.stack([nx.adjacency_matrix(graph) for graph in GL_GN_grid])
A_GN_chain = np.stack([nx.adjacency_matrix(graph) for graph in GL_GN_chain])

# Create list of graphs from the dataset.
GL_DS_star = [nx.from_numpy_matrix(graph) for graph in A_DS_star]
GL_DS_grid = [nx.from_numpy_matrix(graph) for graph in A_DS_grid]
GL_DS_chain = [nx.from_numpy_matrix(graph) for graph in A_DS_chain]

# Draw a random sample from the list.
fname = args.figure_prediction_save_path+'star_16'
draw_graph_list(random.sample(GL_DS_star, 25), 5, 5, fname)


