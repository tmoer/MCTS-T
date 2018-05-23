    # -*- coding: utf-8 -*-
"""
Default hyperparameter settings
@author: thomas
"""
from common.hps_setup import HParams

def override_hps_settings(hps):
    ''' some more global modifications to multiple settings based on 1 indicator.
    Note: when using this, the mode should always set a complete set of variables for all given modes '''
    if hps.mode == 'None':
        pass # do not modify anything
    elif hps.mode == 'off':
        # vanilla mcts 
        hps.sigma_tree = False
        hps.backup_Q = 'on_policy'
        hps.decision_type = 'count'
        hps.block_loop = False
    elif hps.mode == 'sigma':
        hps.sigma_tree = True
        hps.backup_Q = 'uct'
        hps.decision_type = 'mean'
        hps.block_loop = False
    elif hps.mode == 'sigma_loop':
        hps.sigma_tree = True
        hps.backup_Q = 'uct'
        hps.decision_type = 'mean'
        hps.block_loop = True
    return hps

def get_hps():
    ''' Hyperparameter settings '''
    return HParams(      
        # General
        game = 'CartPole-v0', # Environment name
        name = 'unnamed', # Name of experiment
        result_dir = '',
        n_t = 2000, # max timesteps
        n_eps = 100, # max episodes
        steps_per_ep = 400,
        
        # MCTS
        n_mcts = 10,
        c = 1.0,
        decision_type = 'count', # 'count' or 'mean'. For count always uses the backward counts

        sigma_tree = False, # whether to use tree uncertainty        
        backup_policy = 'on-policy', # 'on-policy', 'max' or 'thompson', 'uct-2.0': Type of policy used for value and sigma back-up
        block_loop = False, # Whether to block loops

        temp = 1.0,        
        mode = 'None', # overall indicator that switches on all sigma_tree uncertainty machinery

        # Other
        timeit = False,
        verbose = False
        )