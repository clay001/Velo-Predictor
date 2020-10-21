# -*- coding:utf-8 -*-
from package.test.sampling_test import *
from package.model.RandomForest import model_RF
from package.preprocess.load_data import read_data, gain_data
from package.model.stacking import stacking_model
from package.util.func import *
from package.model import velo_xgb
import os
os.chdir(os.getcwd()[:-8])

# adata = gain_data(dataset="dentategyrus")
# get_gene_list(adata, dataset="dentategyrus", top_k=3)
X = read_data(dataset="dentategyrus", num_class=4, use_gene_list=True)
# origin(X)
# test_ensemble()
# test_oversample(X)
test_oversample(X)
#test_combine(X)
# test_real, test_predict = model_RF(X)
# test_real, test_predict, feature_importances, valid = velo_xgb.model_XGBC(X, num_class = 4)
# test_real, test_predict = stacking_model(X, num_class=4)
# plot_fi(feature_importances)
# eval_result(test_real, test_predict, 4)