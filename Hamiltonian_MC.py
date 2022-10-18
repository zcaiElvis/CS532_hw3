from pickle import TRUE
import torch as tc
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import math
import matplotlib
import matplotlib.pyplot as plt

# Project imports
from evaluation_based_sampling import evaluate_program
from graph_based_sampling import evaluate_graph
from general_sampling import get_sample, flatten_sample
from evaluation_based_sampling import evaluate
from primitives import primitives
from HMC import *
from utils import log_sample


from graphlib import TopologicalSorter # NOTE: This is useful



def add_functions(j):
    rho = {}
    for fname, f in j.items():
        fvars = f[1]
        fexpr = f[2]
        rho[fname] = [fexpr, fvars]
    return rho



def add_graphs_HMC(j, graph_dict, sigma_dict, num_samples):
    graph_vars = j["V"]
    graph_links = j["P"]
    graph_vertices = j["A"]
    graph_observed = j["Y"]

    X0 = {}

    ### If there is no link functino, then return empty graph dict
    if len(graph_links) == 0:
        return graph_dict, sigma_dict

    ### Sort by edges
    sorted = TopologicalSorter(graph_vertices)
    eval_order = tuple(reversed(tuple(sorted.static_order())))

    ### Split latent and observed
    vx_observed = list(graph_observed.keys())
    vx_latent = [x for x in eval_order if x not in vx_observed]

    ### Do a base run
    # for gs in eval_order:
    #     eval_link = evaluate(graph_links[gs], rho= graph_dict, sigma={'logW':0})
    #     graph_dict[gs] = eval_link[0]
    #     sigma_dict['logW'] = sigma_dict['logW'] + eval_link[1]['logW']

    #     if gs in vx_latent:
    #         X0[gs] = eval_link[0]



    for gs in eval_order:
        if gs in vx_latent:
            eval_link = evaluate(graph_links[gs], rho= {**graph_dict, **X0}, sigma={'logW':0})
            X0[gs] = eval_link[0]
            sigma_dict['logW'] = sigma_dict['logW'] + eval_link[1]['logW']
        else:
            X0[gs] = graph_observed[gs]


    


    def gam(X):
        X_dict = dict(zip(eval_order, X))

        sigma_total = tc.tensor(0.0)

        for x in vx_latent:
            d, _ = evaluate(graph_links[x][1], rho= {**graph_dict, **X_dict}, sigma={'logW':tc.tensor(0.0)})
            sigma_total += d.log_prob(X_dict[x])

        for y in vx_observed:
            d, _ = evaluate(graph_links[y][1], rho = {**graph_dict, **X_dict}, sigma={'logW':tc.tensor(0.0)})
            sigma_total += d.log_prob(tc.tensor(graph_observed[y]).float())
        

        return sigma_total


    X_start = tc.tensor(list(X0.values())) ### X_start is latent + observed

    chain = HMC_sampling(gam, start = X_start,  n_points=num_samples, T=5, M=1.)

    return chain, vx_latent


# observe, look up p, and plug in the observed




def evaluate_graph_HMC(ast_or_graph, verbose, num_samples):
    ### Initiate environment, initiate sigma
    env = add_functions(ast_or_graph.functions)
    env = {**env, **primitives}
    sig_dict = {'logW':0}
    samples, vx_latent = add_graphs_HMC(ast_or_graph.graph_spec, env, sig_dict, num_samples)

    results = []
    for X in samples:
        X_dict = dict(zip(vx_latent, X))
        result, sig = evaluate(ast_or_graph.program, rho = {**env, **X_dict}, sigma = {'logw':0})
        results.append(result)


    return results, None, None


def Hamiltonian_MCMC(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):

    samples, _ , _ = evaluate_graph_HMC(ast_or_graph, verbose, num_samples)


    if type(samples[1]) == bool:
        samples = [tc.tensor(int(x)) for x in samples]

    burnin = 0.18
    num_remove = math.floor(len(samples)*burnin)
    samples = samples[num_remove:]

    for i in range(num_samples-num_remove):
        log_sample(samples[i], i, wandb_name)


    return samples, None
