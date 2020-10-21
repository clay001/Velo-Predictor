# -*- coding:utf-8 -*-

import numpy as np
import scvelo as scv
import pandas as pd
scv.logging.print_version()
scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.set_figure_params('scvelo')  # for beautified visualization


def feature_selection(adata, top_k):
    """
    get top genes
    :param adata: anndata for specific dataset
    :param top_k: top k genes for each cluster
    :return: a gene list
    """
    scv.tl.rank_velocity_genes(adata, groupby='clusters')
    a = pd.DataFrame(adata.uns['rank_velocity_genes']['names'][:top_k])
    gene_list = set()
    for i in a.columns:
        for j in range(len(a[i])):
            gene_list.add(a[i][j])
    gene_list = np.array(gene_list)

    return gene_list

