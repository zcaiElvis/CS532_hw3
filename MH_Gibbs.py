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
from general_sampling import flatten_sample
from evaluation_based_sampling import evaluate
from primitives import primitives
from utils import log_sample


from graphlib import TopologicalSorter # NOTE: This is useful



def add_functions(j):
    rho = {}
    for fname, f in j.items():
        fvars = f[1]
        fexpr = f[2]
        rho[fname] = [fexpr, fvars]
    return rho


'''
MH in Gibbs:

1. Run it once, record {'sample vertices': sample_value} and sigma

2. Then repeat:
    Pick a random sample vertex, sample a new value, everything else the same, record {'sample vertices': sample_value} and sigma
    reject or accept based on new sigma vs old sigma
'''

def add_graphs_MHG(j, graph_dict, sigma_dict, num_samples):
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
    for gs in eval_order:
        eval_link = evaluate(graph_links[gs], rho= graph_dict, sigma={'logW':0})
        graph_dict[gs] = eval_link[0]
        sigma_dict['logW'] = sigma_dict['logW'] + eval_link[1]['logW']

        if gs in vx_latent:
            X0[gs] = eval_link[0]


    def accept(x: str, Xprime: dict, X: dict) -> tc.tensor:

        d, _ = evaluate(graph_links[x][1], rho = {**graph_dict, **X}, sigma={'logW':0})

        dprime, _ = evaluate(graph_links[x][1], rho = {**graph_dict, **Xprime}, sigma={'logW':0})

        loga = dprime.log_prob(X[x]) - d.log_prob(Xprime[x])

        Vx = [var for var in eval_order if var in  graph_vertices[x]+[x]]

        for v in Vx:
            d1, _ = evaluate(graph_links[v][1], rho = {**graph_dict, **Xprime}, sigma={'logW':0})
            d2, _ = evaluate(graph_links[v][1], rho = {**graph_dict, **X}, sigma={'logW':0})
            
            if v in graph_observed.keys():
                s1 = d1.log_prob(tc.tensor(graph_observed[v]).float())
                s2 = d2.log_prob(tc.tensor(graph_observed[v]).float())
            else:
                s1 = d1.log_prob(Xprime[v])
                s2 = d2.log_prob(X[v])

            loga = loga + s1 - s2

        return tc.exp(loga)

    def gibb_steps(X: dict) -> dict:

        for x in list(X.keys()):

            d, sig = evaluate(graph_links[x][1], rho = {**graph_dict, **X}, sigma={'logW':0})

            Xprime = copy.deepcopy(X)

            Xprime[x] = d.sample()

            a = accept(x, Xprime, X)

            u = tc.distributions.Uniform(low=0, high=1).sample()

            if u < a:

                X = copy.deepcopy(Xprime)

        return X


    def gibbs(X0 : dict, S : int) -> list:
        samples = [X0]
        for s in tqdm(range(1,S)):
            Xs = gibb_steps(samples[s-1])

            samples.append(Xs)
        return samples

    samples = gibbs(X0, num_samples)

    return samples




def evaluate_graph_MHG(ast_or_graph, num_samples, verbose):
    ### Initiate environment, initiate sigma
    env = add_functions(ast_or_graph.functions)
    env = {**env, **primitives}
    sig_dict = {'logW':0}
    samples = add_graphs_MHG(ast_or_graph.graph_spec, env, sig_dict, num_samples)

    results = []
    for X in samples:
        result, sig = evaluate(ast_or_graph.program, rho = {**env, **X}, sigma = {'logw':0})
        results.append(result)


    return results, None, None


def Metropolis_in_Gibbs_sampler(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):

    samples, _ , _ = evaluate_graph_MHG(ast_or_graph, num_samples, verbose)

    if type(samples[1]) == bool:
        samples = [tc.tensor(int(x)) for x in samples]


    burnin = 0.18
    num_remove = math.floor(len(samples)*burnin)
    samples = samples[num_remove:]

    for i in range(num_samples-num_remove):
        log_sample(samples[i], i, wandb_name)

    # log_sample(flatten_sample(samples), program, "elvis_cai")



    # plt.plot(samples)
    # plt.savefig("output.png")

    # plot_samples = tc.stack(samples).type(tc.float)
    # print(plot_samples.size())

    # for i in range(0,plot_samples.size(dim=1)):
    #     plt.plot(plot_samples[:,i])

    # plt.savefig("output.png")

    



    return samples, None
