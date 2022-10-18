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


from graphlib import TopologicalSorter # NOTE: This is useful



def add_functions(j):
    rho = {}
    for fname, f in j.items():
        fvars = f[1]
        fexpr = f[2]
        rho[fname] = [fexpr, fvars]
    return rho



def add_graphs_HMC(j, graph_dict, sigma_dict):
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


    def HMC_sampling(lnf, start, n_points=int(1e3), M=1., dt=0.1, T=1., verbose=False):

        # Functions for leap-frog integration
        def get_gradient(x, lnf):
            # x = x.detach()
            x.requires_grad_()
            lnf(x).backward()
            dlnfx = x.grad
            x = x.detach() # TODO: Not sure if this is necessary
            return dlnfx
        def leap_frog_step(x, p, lnf, M, dt):
            dlnfx = get_gradient(x, lnf)
            p_half = p+0.5*dlnfx*dt
            x_full = x+p_half*dt/M
            dlnfx = get_gradient(x_full, lnf)
            p_full = p_half+0.5*dlnfx*dt
            return x_full, p_full
        def leap_frog_integration(x_init, p_init, lnf, M, dt, T):
            N_steps = int(T/dt)
            x, p = tc.clone(x_init), tc.clone(p_init)
            for _ in range(N_steps):
                x, p = leap_frog_step(x, p, lnf, M, dt)
            return x, p
        def Hamiltonian(x, p, lnf, M):
            T = 0.5*tc.dot(p, p)/M
            V = -lnf(x)
            return T+V
        # MCMC step
        n = len(start)
        x_old = tc.clone(start); xs = []; n_accepted = 0
        for i in range(n_points):
            p_old = tc.normal(0., 1., size=(n,))
            if i == 0: H_old = 0.
            x_new, p_new = leap_frog_integration(x_old, p_old, lnf, M, dt, T)
            H_new = Hamiltonian(x_new, p_new, lnf, M)
            acceptance = 1. if (i == 0) else min(tc.exp(H_old-H_new), 1.) # Acceptance probability
            accept = (tc.rand((1,)) < acceptance)
            if accept: x_old, H_old = x_new, H_new; n_accepted += 1
            xs.append(x_old)
        chain = tc.stack(xs)
        if verbose: print('Acceptance fraction: %1.2f'%(n_accepted/n_points))
        return chain

    print(chain)


    return 0





def evaluate_graph_HMC(ast_or_graph, verbose):
    ### Initiate environment, initiate sigma
    env = add_functions(ast_or_graph.functions)
    env = {**env, **primitives}
    sig_dict = {'logW':0}
    samples = add_graphs_HMC(ast_or_graph.graph_spec, env, sig_dict)

    results = []
    for X in samples:
        result, sig = evaluate(ast_or_graph.program, rho = {**env, **X}, sigma = {'logw':0})
        results.append(result)


    return results, None, None


def Hamiltonian_MCMC(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):

    samples, _ , _ = evaluate_graph_HMC(ast_or_graph, verbose)

    if type(samples[1]) == bool:
        samples = [tc.tensor(int(x)) for x in samples]





    return samples, None
