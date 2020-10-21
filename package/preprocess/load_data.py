# -*- coding:utf-8 -*-
from package.util.func import *
import scvelo as scv
import pandas as pd
import os

@timing
def gain_data(dataset, mode='stochastic', basis='umap', cluster_info=False):
    '''
    process data to get RNA velocity
    :param dataset: dataset name
    :return: adata
    '''
    if dataset == "dentategyrus":
        adata = scv.datasets.dentategyrus()
    if dataset == "pancreas":
        adata = scv.datasets.pancreas()
    if dataset == "forebrain":
        adata = scv.datasets.forebrain()

    scv.pl.proportions(adata)
    # filter genes, log = True or False
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
    # this is on pca space，compute moments
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30, mode='connectivities')

    # fitting a ratio between precursor and mature mRNA abundances
    if mode == 'dynamical':
        scv.tl.recover_dynamics(adata)
    # compute velo
    scv.tl.velocity(adata, mode=mode)

    # cosine correlation of potential cell transitions
    # approx=True： reduced PCA space with 50 components
    # embed lower D, compute transition matrix
    scv.tl.velocity_graph(adata)

    # scv.tl.umap(adata)
    scv.pl.velocity_graph(adata)
    # embedding
    scv.pl.velocity_embedding(adata, basis=basis, arrow_length=1.5, arrow_size=1.5, dpi=1000)

    # recover latent time
    # scv.tl.recover_latent_time(adata)
    # plot latent time
    # scv.pl.scatter(adata, color='latent_time', fontsize=24, size=100,
    # color_map='gnuplot', perc=[2, 98], colorbar=True, rescale_color=[0, 1])

    df = pd.DataFrame(adata.X.A, columns=adata.var_names)

    # whether use cluster information
    if cluster_info == True:
        clusters = pd.DataFrame(list(adata.obs["clusters"]), columns=["clusters"])
        df = pd.concat([df, clusters], axis=1)
        df = pd.get_dummies(df, columns=['clusters'])

    # create doc folder
    if not os.path.exists("./processed_data/" + dataset):
        os.makedirs("./processed_data/" + dataset)

    velo = pd.DataFrame(adata.layers["velocity"], columns=adata.var_names)
    # 存储target信息和数据矩阵信息

    df.to_csv("./processed_data/" + dataset + "/data.csv", index=False)
    np.savetxt("./processed_data/" + dataset + "/VData.txt", adata.obsm["velocity_umap"])
    velo.to_csv("./processed_data/" + dataset + "/target.csv", index=False)

    # cell state info: adata.obsm["X_umap"]
    # gene_name: adata.var_names

    # latent time: adata.obs["latent_time"]

    # cluster info
    # cluster = list(adata.obs["clusters_enlarged"])
    # type_list = set(cluster)
    # n = len(type_list)
    # values = [i for i in range(n)]
    # dictionary = dict(zip(type_list, values))
    # for i in range(len(cluster)):
    #     cluster[i] = dictionary[cluster[i]]
    # storage
    # np.savetxt("./processed_data/" + dataset + "/cluster_info.txt", cluster)
    # np.save("./processed_data/" + dataset + '/cluster_dic.npy', dictionary)
    return adata

# 目标基因预测
@timing
def read_data_gene(dataset, target_gene, if_denoise=False, use_gene_list=False, threshold=0):
    # cell_state
    path = "./processed_data/" + dataset + "/"
    if if_denoise:
        X = pd.read_csv(path + "denoised.csv", header=0)
    else:
        X = pd.read_csv(path + "data.csv", header=0)
    if use_gene_list:
        gene_list = np.load(path + "gene_list.npy", allow_pickle=True).tolist()
        X = pd.DataFrame(X, columns=gene_list)

    V = pd.read_csv( path + "target.csv", header=0 )
    target = get_target(V, target_gene, threshold=threshold)
    F = pd.concat([X, target], axis=1)
    F = F.reset_index().rename(columns={'index': 'ID'})
    return F

@timing
def read_data(dataset, if_denoise=False, num_class = 4, use_gene_list=False):
    # cell_state
    path = "./processed_data/" + dataset + "/"
    if if_denoise:
        X = pd.read_csv(path + "denoised.csv", header=0)
    else:
        X = pd.read_csv(path + "data.csv", header=0)
    if use_gene_list:
        gene_list = np.load(path + "gene_list.npy", allow_pickle=True).tolist()
        X = pd.DataFrame(X, columns=gene_list)

    V = pd.read_csv(path + "VData.txt", sep=' ', header=None, names=["V_xcom", "V_ycom"])
    direction = get_direction(V, num_class= num_class)

    F = pd.concat([X, direction], axis=1)
    F = F.reset_index().rename(columns={'index': 'ID'})

    return F

