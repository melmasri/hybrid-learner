
import os
import sys
import pydot
sys.path.append('..')
sys.path.append('/Users/m.elmasri/src/parallelDG/parallelDG/')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
import parallelDG.parallelDG.auxiliary_functions as aux
from networkx.drawing.nx_pydot import graphviz_layout
sns.set_style("whitegrid")

import tarfile
import pandas as pd
from io import StringIO
import glob

## general setup
BENCHPRESS_LOC = 'results/'
SAVE_LOC = 'img/'
SAVE_PLOTS = True
BENNCHMARK_LOC = BENCHPRESS_LOC + 'output/benchmarks/'

def read_csv_from_tar_gz(tar_gz_path, csv_file_name=None):
    """
    Reads a CSV file from a tar.gz archive into a pandas DataFrame.
    
    Parameters:
    - tar_gz_path: str, the path to the tar.gz archive.
    - csv_file_name: str, optional, the name of the CSV file to extract. 
                     If not provided, the first CSV file found will be read.
    
    Returns:
    - A pandas DataFrame containing the data from the CSV file.
    """
    if 'csv' in tar_gz_path: 
        return pd.read_csv(tar_gz_path)
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        # If a specific CSV file name is provided, use it; otherwise, find the first CSV file
        if csv_file_name:
            try:
                csv_member = tar.getmember(csv_file_name)
            except KeyError:
                raise FileNotFoundError(f"The specified CSV file {csv_file_name} was not found in the archive.")
        else:
            # Assume there's only one CSV file and extract it
            csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
            if csv_member is None:
                raise FileNotFoundError("No CSV file found in the tar.gz archive.")
        
        # Extract the CSV file content
        f = tar.extractfile(csv_member)
        if f is not None:
            content = f.read()
            # Convert bytes to a string if necessary
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            # Use StringIO to convert the string to a file-like object and read it into a pandas DataFrame
            df = pd.read_csv(StringIO(content))
            return df
        else:
            raise Exception("Could not extract the CSV file from the tar.gz archive.")



def list_with_pattern(pat, ls):
    return [l for l in ls if pat in l]

def algo_files(location, algo_name = 'parallel', pattern=None): 
    filelist = []
    for root, dir, files in os.walk(BENCHPRESS_LOC): 
        for file in files: 
            filelist.append(os.path.join(root,file))
    
    alg = algo_name
    fl = list_with_pattern(alg, filelist)
    if pattern: 
        if type(pattern) is list: 
            final_ls = fl
            for p in pattern: 
                final_ls = list_with_pattern(p, final_ls)
            return sorted(final_ls)
        else:
            return sorted(list_with_pattern(pattern, fl))
    else: 
        return sorted(fl)
    
def edges_str_to_list(str, edgesymb="-"):
    """
    Converts a string representation of edges to a list of edges.

    Args:
      str: The string representation of the edges.
      edgesymb: The symbol used to separate the nodes in each edge.

    Returns:
      A list of edges, where each edge is a tuple of two nodes.
    """
    try:
        edges_str = str[1:-1].split(";")
        edges = []
        for edge in edges_str:
            if edgesymb in edge:
                e1, e2 = map(int, edge.split(edgesymb))
                edges.append((min(e1, e2), max(e1, e2)))
        return edges
    except (ValueError, IndexError) as e:
        print(f"Error parsing edges string: {e}")
        return []

def size_traj(filename):
    edgesymb = "-"
    g = nx.Graph()
    size = []
    df = read_csv_from_tar_gz(filename)
    if 'subindex' in df.columns: 
        df.rename(columns={'added_sub': 'added', 
                           'subindex': 'index', 
                          'removed_sub': 'removed'}
                  , inplace=True)
    
    for index, row in df.iterrows():
            added = edges_str_to_list(row["added"], edgesymb)
            removed = edges_str_to_list(row["removed"], edgesymb)
            g.add_edges_from(added)
            g.remove_edges_from(removed)
            size.append(g.size())
    df["size"] = size
    T = df["index"].iloc[-1]  # approximate length
    newindex = pd.Series(range(T))
    # removes the two first rows.
    df2 = df[["index", "size"]][2:].set_index("index")
    df2 = df2.reindex(newindex).reset_index().reindex(
        columns=df2.columns).ffill()
    return df2

def num_samples(filename):
    df = read_csv_from_tar_gz(filename)
    if 'subindex' in df.columns:
        a = df['subindex'].iloc[-1]
        print(f"num samples {a}")
    else:
        a = df['index'].iloc[-1]
        print(f"num samples {a}")
    return a


def size_traj_plot(filename): 
    df = size_traj(filename)
    df["size"].plot()
    plt.tight_layout()
    plt.xlabel('Sample number')
    plt.ylabel('Number of graph edges')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    return df
    #plt.savefig(snakemake.output["plot"])


def score_traj_plot(filename): 
    edgesymb = "-"
    g = nx.Graph()
    df = read_csv_from_tar_gz(filename)
    T = df["index"].iloc[-1]  # approximate length

    newindex = pd.Series(range(T))
      # removes the two first rows.
    df2 = df[["index", "score"]][2:].set_index("index")
    df2 = df2.reindex(newindex).reset_index().reindex(
        columns=df2.columns).ffill()
    df2["score"].plot()
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Log-likelihood')
    plt.xlabel('Sample number')
    plt.tight_layout()
    return df2
    #plt.savefig(snakemake.output["plot"])
    
    
def autocorrelation(traj, lag):
    return aux.autocorrelation_plot(traj,lag = lag,
                                 return_series=True)

def acceptance_ratio(traj):
    dx = (np.diff(traj)!=0) * 1.0
    ar = np.mean(dx, dtype=float)
    par = np.cumsum(dx, dtype=float)/(np.array(range(len(dx))) + 1.0)
    print('Acceptance ratio {:2f}'.format(ar))
    return ar, par

def heatmap_df(filename, burnin_frac=0.5):
    edgesymb = "-"
    g = nx.Graph()
    df = read_csv_from_tar_gz(filename)
    nodeorder = []
    tmpedges = edges_str_to_list(df["added"][0], edgesymb)
    nodeorder.append(tmpedges[0][0])
    for edge in tmpedges:
        nodeorder.append(edge[1])
            
    full_its = df["index"].iloc[-1]        
    totalits = int(full_its * (1.0 - burnin_frac))
    burnin_ind = int(full_its * burnin_frac)
    
    for index, row in df.iterrows():
        
        if row["index"] == 0:
            if burnin_ind == 0:
                heatmap = nx.to_numpy_array(g) #this is strange
            else:
                # Just to init the matrix with the right dimensions  
                heatmap = nx.to_numpy_array(g) * 0 

        if row["index"] > burnin_ind:
            cur_index = df["index"].iloc[index]
            prev_index = df["index"].iloc[index-1]
                
            weight = cur_index - prev_index            

            heatmap += nx.to_numpy_array(g, nodelist=nodeorder) * weight

        added = edges_str_to_list(row["added"], edgesymb)
        removed = edges_str_to_list(row["removed"], edgesymb)
        g.add_edges_from(added)
        g.remove_edges_from(removed)

   
    heatmap /= totalits                             
    heatmap_df = pd.DataFrame(heatmap)
    heatmap_df.columns = g.nodes()
    return heatmap_df

def plot_heatmap(filename, cbar=True, annot=False, xticklabels=10, yticklabels=10):
    heatmap = heatmap_df(filename, 0.5)
    plot_graph(heatmap, cbar, annot, xticklabels, yticklabels)
    

def plot_nx_graph(heatmap):
    sns.set_style("whitegrid")
    options = {
        "font_size": 8,
        "node_size": 10,
        "node_color": "red",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }
    heatmap.index = heatmap.columns
    ar_graph = nx.from_pandas_adjacency(heatmap)
    pos = graphviz_layout(ar_graph, prog="fdp")
    nx.draw_networkx(ar_graph,pos=pos, with_labels=False, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    

def most_samped_graph(filename):
    edgesymb = "-"
    g = nx.Graph()
    df = read_csv_from_tar_gz(filename)
    most_sampled = aux.most_sampled_graph(traj.trajectory[burnin:])
    most_sampled
    aux.plot_heatmap(nx.to_numpy_array(most_sampled['g']), xticklabels=10, yticklabels=10)
    
def plot_graph(heatmap, cbar=True, annot=False, xticklabels=10, yticklabels=10):
    heatmap.index = heatmap.columns
    mask = np.zeros_like(heatmap)
    mask[np.triu_indices_from(mask)] = True

    
    with sns.axes_style("white"):
        ax = sns.heatmap(heatmap, mask=mask, annot=annot,
                         cmap="Blues",
                         vmin=0.0, vmax=1.0, square=True,
                         cbar=cbar,linewidth=1,
                         xticklabels=xticklabels,
                         yticklabels=yticklabels)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        
    #sns.set_style("whitegrid")
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=12)
    plt.tight_layout()


def custom_roc_curve(true_labels, scores, thresholds):
    tpr_list = []
    fpr_list = []

    # Calculate TPR and FPR for each threshold
    for threshold in thresholds:
        TP = FP = TN = FN = 0
        for score, label in zip(scores, true_labels):
            if score >= threshold:
                if label == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if label == 1:
                    FN += 1
                else:
                    TN += 1

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return fpr_list, tpr_list
    

def fpr_tpr(filename_true_graph, filename_traj):
    thresholds = np.arange(0, 1.1, 0.1)  # From 0 to 1 with 0.1 increment
    adjacency_matrix = read_csv_from_tar_gz(filename_true_graph)
    heatmap = heatmap_df(filename_traj, 0.5).to_numpy().flatten()
    # Flatten the matrices to create a 1D array
    scores = heatmap.flatten() # These are your predicted scores/probabilities
    true_labels = adjacency_matrix.to_numpy().flatten() # These are your true binary labels

    # Calculate ROC metrics
    fpr, tpr = custom_roc_curve(true_labels, scores, thresholds)
    return np.array(fpr), np.array(tpr), thresholds


def fpr_tpr_files(list_traj, list_true_graphs):
    k =  len(list_traj)
    fpr = tpr = thresholds = 0.0
    for traj, true_graph in zip(list_traj, list_true_graphs):
        fpr1, tpr1, thresholds = fpr_tpr(true_graph, traj)
        fpr += fpr1 / k
        tpr += tpr1 /k
    
    return fpr, tpr, thresholds
    

    
def create_filename(filename, loc = SAVE_LOC):
    return loc + filename

    
def save_location(filename, loc = SAVE_LOC):
    if SAVE_PLOTS:
        f = create_filename(filename, loc) + '.jpg'
        print(f"save to: {f}")
        plt.savefig(f, dpi = 600, bbox_inches='tight')
        plt.clf()
    else: 
        plt.show()


def get_ROCfilenaem(benchmark_pattern):
    folder_name = glob.glob(benchmark_pattern, root_dir=BENNCHMARK_LOC, recursive=True)[0]
    roc_filename = f"{BENNCHMARK_LOC}{folder_name}/ROC_data.csv"
    return roc_filename

def filter_roc_data(roc_filename, patterns):
    roc_data = pd.read_csv(roc_filename)
    for key, value in patterns.items():
        if 'bandmat' in value:
            roc_data = roc_data[roc_data[key].str.contains('bandmat')]
        else:
            roc_data = roc_data[roc_data[key].str.contains(value)]
        if 'random' in value:
            roc_data = roc_data[roc_data[key].str.contains('prob=0.01')]
    return roc_data.sort_values(by='TPR_pattern_mean', ascending=True)

def file_roc_data(roc_filename, patterns):
    roc_data = filter_roc_data(roc_filename, patterns)
    return roc_data

def get_tpr_fpr_threshold(roc_file):
    tpr = np.array(roc_file['TPR_pattern_mean'].values)
    fpr = np.array(roc_file['FPR_pattern_mean'].values)
    threshold = np.array(roc_file['curve_vals'].values)
    tpr = np.hstack((np.array(0), tpr, np.array(1)))
    fpr = np.hstack((np.array(0), fpr, np.array(1)))
    threshold = np.hstack((np.array(0), threshold, np.max(threshold)+1))
    zipped = sorted(zip(fpr, tpr, threshold), key=lambda x: x[1])
    fpr, tpr, threshold = zip(*zipped)
    return np.array(fpr), np.array(tpr), np.array(threshold)

def get_model(roc_file, model):
    return roc_file[roc_file['id'].str.contains(model)]


def get_title(patterns):
    g = patterns['adjmat']
    d = patterns['data'][:-1]
    p = patterns['parameters']
    return d +', graph='+ g + ', param=' + p


def get_max_score_graph(df):
    edgesymb = "-"
    if edgesymb == "-":
        g = nx.Graph()
    elif edgesymb == "->":
        g = DiGraph()

    
    maxscore = df[3:]["score"].max()
    for index, row in df.iterrows():
        added = edges_str_to_list(row["added"], edgesymb=edgesymb)
        removed = edges_str_to_list(row["removed"], edgesymb=edgesymb)
        g.add_edges_from(added)
        g.remove_edges_from(removed)
        if row["score"] == maxscore:
            break

    df_adjmat = pd.DataFrame(nx.to_numpy_array(g), dtype=int)
    df_adjmat.columns = g.nodes()
    return df_adjmat

def plot_max_score_graph(df):
    df_adjmat = get_max_score_graph(df)
    plot_nx_graph(df_adjmat)
    return None




def get_decomposable_cover(df, true_graph, burnin_frac=0.5):
    edgesymb = "-"
    g = nx.Graph()
    true_edges = set(true_graph.edges())
    num_true_edges = len(true_edges)
    burnin_index = int(df["index"].iloc[-1] * burnin_frac)
    total_pop = df["index"].iloc[-1] - burnin_index
    previous_index = 0
    tp = list()
    for index, row in df.iterrows():
        added = edges_str_to_list(row["added"], edgesymb=edgesymb)
        removed = edges_str_to_list(row["removed"], edgesymb=edgesymb)
        g.add_edges_from(added)
        g.remove_edges_from(removed)
        if row["index"] > burnin_index:
            w = (row['index'] - previous_index)
            num_join_edges = [len(set(g.edges()) & true_edges) / num_true_edges] * w
            tp += num_join_edges
        previous_index = row["index"]
    return tp
