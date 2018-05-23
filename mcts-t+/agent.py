# -*- coding: utf-8 -*-
"""
Chain experiments
@author: thomas
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

global mpl
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import os
import time
import tensorflow as tf
import argparse
from pprint import pformat

# common package import
from common.rl.make_game import make_game
from common.submit import make_unique_subfolder
from common.hps_setup import hps_to_dict
from common.visualize import plot_single_experiment
from common.rl.atari_copy import is_atari_game
from common.putils import store_safely

# local imports
from hps import get_hps,override_hps_settings
from lib.mcts import MCTS,display_info

def agent(hps):
    ''' Agent function '''
    tf.reset_default_graph()
    
    # storage
    result = {}
    steps = [] # will indicate the timestep for the learning curve
    returns = [] # will indicate the return for the timestep in env_steps  
    times = [] 
    best_R = -np.Inf    
               
    Env = make_game(hps.game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(hps.game) if is_atari else None

    with tf.Session() as sess, sess.as_default():
        sess.run(tf.global_variables_initializer())
        global_t_mcts = 0
        global_t = 0 
        
        for ep in range(hps.n_eps):
            root_index = Env.reset() 
            root = None
            seed = np.random.randint(1e7) 
            Env.seed(seed)
            subtimes = []
            
            if is_atari: 
                mcts_env.reset()
                mcts_env.seed(seed)                                

            a_store = []
            R = 0.0 # episode reward
            t = 0 # episode steps        
            
            while True:
                # run an episode
                if hps.timeit: now = time.time()
                root = MCTS(root_index=root_index,root=root,Env=Env,mcts_env=mcts_env,N=hps.n_mcts,c=hps.c,block_loop=hps.block_loop,
                            sigma_tree=hps.sigma_tree,backup_policy=hps.backup_policy,max_depth=hps.steps_per_ep-t,timeit=False)
                if hps.timeit:
                    time_spend = time.time()-now
                    subtimes.append(time_spend)
                    #print('One MCTS search takes {} seconds'.format(time_spend))  
                if hps.verbose: display_info(root,'{}'.format(t),hps.c)
                    
                probs,V,a = root.return_results(decision_type=hps.decision_type,backup_policy=hps.backup_policy,temperature=hps.temp,c=hps.c)
                
                # Make the step
                a_store.append(a)
                s1,r,terminal,_ = Env.step(a)
                R += r
                t += 1
                global_t += 1
                global_t_mcts += hps.n_mcts
    
                if hps.verbose:
                    if (t % 50) == 0: 
                        print('Overall step {}, root currently returns V {}, and considers probabilities {}'.format(global_t,V,probs))
                
                if terminal or (t > hps.steps_per_ep):
                    if hps.verbose:
                        print('Episode terminal, total reward {}, steps {}'.format(R,t))
                    returns.append(R)
                    steps.append(global_t_mcts)
                    break # break out, start new episode
                else:
                    root = root.forward(a=a,s1=s1,r=r,terminal=terminal,t=t)

            # saving
            result.update({'steps':steps,'return':returns})
            if hps.timeit:
                times.append(np.mean(subtimes))
                result.update({'time':times})
            #if R > best_R:
            #    result.update({'seed':seed,'actions':a_store,'R':best_R})
            #    best_R = R
            store_safely(hps.result_dir,'result',result)
  
            if (global_t_mcts > hps.n_t) or (ep > hps.n_eps):
                break # break out of episode loop
    
    return result
    
if __name__ == '__main__':
    '''Set-up training'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp', help='Hyperparameter configuration',default='')
    parser.add_argument('--no_plot', action='store_true',default=False)
    args = parser.parse_args()
    hps = get_hps().parse(args.hp)    
    hps = override_hps_settings(hps)

    # set-up result folder if not prespecified
    if hps.result_dir == '':
        result_folder = os.getcwd() + '/results/{}/{}/'.format(hps.name,hps.game)
        hps.result_dir = make_unique_subfolder(result_folder,hyperloop=False)
    else:
        with open(hps.result_dir + 'hps.txt','w') as file:
            file.write(pformat(hps_to_dict(hps)))
    #with open(subfolder + 'hps_raw.txt','w') as file:
    #    file.write(hps_to_list(hps)) 
    print(' ________________________________________ ')     
    print('Start learning on game {}'.format(hps.game))               
    result = agent(hps)
    
    if not args.no_plot:
        plot_single_experiment(result,hps.game,hps.result_dir,plot_type='lc')