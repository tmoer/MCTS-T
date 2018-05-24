# -*- coding: utf-8 -*-
"""
MCTS with tree uncertainty
@author: Thomas Moerland, Delft University of Technology
"""

import numpy as np
import random
import copy
import time

from common.rl.atari_copy import restore_atari_state, is_atari_game, copy_atari_state
from common.putils import my_argmax

def bring_Env_to_correct_state(Env,seed,a_his):
    ''' Forward simulates an environment based an a history of taken actions and a seed 
    Note: not used because seeding is just as slow as copy.deepcopy(env) '''
    if len(a_his) == 0:
        return Env
    Env.reset()
    #Env.seed(seed) # this takes just as long as copy.deepcopy(Env), and is therefore avoided. We choose to
    # only consider games with deterministic initial state and deterministic transitions, which avoids seeding
    for a in a_his:
        Env.step(a)
    return Env
    
def MCTS(root_index,root,Env,N,mcts_env=None,c=1.0,gamma=1.0,block_loop=False,sigma_tree=False,
         backup_policy='on-policy',max_depth=300,timeit=False):
    ''' Monte Carlo Tree Search function '''
    na = Env.action_space.n 
    if root is None:
        root = State(root_index,r=0.0,terminal=False,parent_action=None,na=na,sigma_tree=sigma_tree) # initialize the root node
    else:
        root.parent_action = None # continue from current root

    if root.terminal:
        raise(ValueError("Can't do tree search from a terminal state"))

    is_atari = is_atari_game(Env)
    if is_atari:
        snapshot = copy_atari_state(Env) # snapshot the root at the beginning     
    
    if timeit:
        copy_time = 0.0
        forward_time = 0.0
        backward_time = 0.0    
    
    for i in range(N):     
        state = root # reset to root for new trace
        depth = 0
        if timeit: now = time.time()
        if is_atari:
            restore_atari_state(mcts_env,snapshot)
        else:
            mcts_env = copy.deepcopy(Env) # copy original Env to rollout from
        if timeit:
            copy_time += time.time()-now 
            now = time.time()           
        
        while not state.terminal: 
            action = state.select(c=c)
            s1,r,t,_ = mcts_env.step(action.index)
            depth += 1
            if hasattr(action,'child_state'):
                state = action.child_state # select
                continue
            else:
                state = action.add_child_state(s1,r,t,sigma_tree) # expand
                state.evaluate(Env=mcts_env,gamma=gamma,block_loop=block_loop,max_roll=max_depth-depth) # evaluate/roll-out
                break
        if timeit:   
            forward_time += time.time()-now 
            now = time.time()           

        # backup the expansion    
        state.V = state.V_hat            
        # loop back up
        while True:
            action = state.parent_action
            action.backup(gamma=gamma)
            state = action.parent_state
            if state.parent_action is not None:
                state.backup(backup_policy=backup_policy,c=c)
            else:
                break # reached root node
                
        if timeit: backward_time += time.time()-now 
    if timeit:
        total_time = copy_time + forward_time + backward_time
        print('total time {}\n copy % {}, forward % {}, backward % {}'.format(total_time,100*copy_time/total_time,100*forward_time/total_time,100*backward_time/total_time))
    return root  

def check_for_loop_in_trace(state,threshold=0.01):
    ''' loops back through trace to check for a loop (= repetition of state) '''
    index = state.index # expanded state
    R = state.r
    n = 1
    while True:
        state = state.parent_action.parent_state
        if np.linalg.norm(state.index-index) < threshold:
            return True,R,n # found a loop        
        if state.parent_action is None:
            break # reached the root, no loop detected
        else:
            R += state.r # add reward and move up
            n += 1
    return False,R,n

class Action():
    ''' Action object '''
    def __init__(self,index,parent_state):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0 # sum
        self.n = 0 # counts
        self.backward_n = 0
        self.Q = 0.0 # mean
        self.sigma_t = 1.0
        #self.R = [] # individual returns estimates
                
    def add_child_state(self,s1,r,terminal,sigma_tree):
        self.child_state = State(index=s1,r=r,terminal=terminal,parent_action=self,na=self.parent_state.na,sigma_tree=sigma_tree)
        return self.child_state
        
    def backup(self,gamma):
        self.n += 1
        self.Q = self.child_state.r + gamma * self.child_state.V
        self.sigma_t = self.child_state.sigma_t

def stable_normalizer(x,temp):
    x = x / np.max(x)
    return (x ** temp)/np.sum(x ** temp)

def normalizer(x,temp):
    return np.abs((x ** temp)/np.sum(x ** temp))

class State():
    ''' State object '''

    def __init__(self,index,r,terminal,parent_action,na,sigma_tree=False):
        ''' Initialize a new state '''
        self.index = index # state
        self.r = r # reward upon arriving in this state
        self.terminal = terminal 
        self.parent_action = parent_action 
        self.na = na # number of actions
        self.n = 0 # visitation count
        self.sigma_tree = sigma_tree # boolean indicating use of sigma_tree
        self.sigma_t = 1.0 if not terminal else 0.0 
        if not terminal:
            self.add_child_actions()
    
    def add_child_actions(self):   
        ''' Adds child nodes for all actions '''
        self.child_actions = [Action(a,parent_state=self) for a in range(self.na)]
    
    def select(self,c):
        ''' Select one of the child actions based on UCT rule '''
        Q = np.array([child_action.Q for child_action in self.child_actions],dtype='float32')
        U = np.array([c * (np.sqrt(self.n)/(child_action.n)) if child_action.n > 0 else np.Inf for child_action in self.child_actions],dtype='float32')
        if self.sigma_tree:
            sigma_actions_t = np.array([child_action.sigma_t for child_action in self.child_actions])
            U *= sigma_actions_t
        scores = Q + U
        winner = my_argmax(scores)
        return self.child_actions[winner]

    def return_results(self,decision_type,backup_policy,temperature,c):
        counts = self.get_backward_counts(backward_policy=backup_policy,c=c)
        probs = stable_normalizer(counts,temperature)
        Q = np.array([child_action.Q for child_action in self.child_actions],dtype='float32')
        V = np.sum(counts*Q)/np.sum(counts)[None]
        
        if decision_type == 'count':
            a = my_argmax(counts)
        elif decision_type == 'mean':
            Q_ = np.array([child_action.Q if child_action.n > 0 else -np.Inf for child_action in self.child_actions])
            a = my_argmax(Q_) 
        return probs,V,a
    
    def evaluate(self,Env,gamma,block_loop,max_roll):
        ''' get an estimate V_hat of the state value through a (random) rollout '''        
        if self.terminal:
            self.V_hat,self.V = 0.0,0.0
            return
        if block_loop:
            looped,R,n = check_for_loop_in_trace(self,threshold=0.01)
            if looped:
                self.sigma_t = 0.0
                if R == 0:
                    self.V_hat,self.V= 0.0,0.0
                else:
                    est = (R/n)*max_roll # estimate the remaining worth of the loop
                    self.V_hat,self.V= est,est
                return
        self.V_hat = rollout(s=self.index,Env=Env,policy='random',gamma=gamma,max_roll=max_roll)
        self.V = self.V_hat
        #self.child_actions[a_init].backup(self.V) # already log which child action was first in the roll-out
        
    def backup(self,backup_policy,c):
        ''' update statistics (self.n,self.V,self.sigma_t) on back-ward pass'''
        self.n += 1
        counts = self.get_backward_counts(backward_policy=backup_policy,c=c)
        
        # update value
        Q = np.array([child_action.Q for child_action in self.child_actions])
        self.V = np.sum(counts*Q)/(np.sum(counts)+1) + self.V_hat/(np.sum(counts)+1) # always weight in the first rollout once
        # update tree sigma
        if self.sigma_tree:
            # modify the counts to have a 1 for untried actions:
            counts = np.array([count if child_action.n > 0 else 1 for count,child_action in zip(counts,self.child_actions)]) # replace the 0 counts with 1
            # collect the sigma_t
            sigma_t_actions = np.array([child_action.sigma_t for child_action in self.child_actions])
            # set the new sigma_t
            self.sigma_t = np.sum(counts*sigma_t_actions)/np.sum(counts)
                          
    def forward(self,a,s1,r,terminal,t):
        if not hasattr(self.child_actions[a],'child_state'):
            # still need to add the next state
            self.child_actions[a].add_child_state(s1,r,terminal,self.sigma_tree)    
        elif np.linalg.norm(self.child_actions[a].child_state.index-s1) > 0.01:
            print('Warning: this domain seems stochastic. Throwing away the tree')
            print(self.child_actions[a].child_state.index - s1)
            print('Timestep {}'.format(t))
            print(self.child_actions[a].n,self.child_actions[a].child_state.n,self.child_actions[a].child_state.terminal)
            print(a,self.child_actions[a].index)
            return None
        else:
            return self.child_actions[a].child_state

    def get_backward_counts(self,backward_policy,c):
        ''' returns a vector of counts to be used as policy in the backward pass '''
        if backward_policy == 'on-policy':
            counts = [child_action.n for child_action in self.child_actions]
        elif backward_policy == 'off-policy':
            Q = np.array([child_action.Q if child_action.n > 0 else -np.Inf for child_action in self.child_actions])
            counts = [0 for i in range(len(self.child_actions))]
            index = my_argmax(Q)
            counts[index] += 1
        elif 'ucb' in backward_policy:
            try:
                _,c = backward_policy.split('-')
            except:
                c = c
            backward_a = self.ucb_backward_sample(float(c))
            self.child_actions[backward_a].backward_n += 1
            counts = [child_action.backward_n for child_action in self.child_actions]
        elif backward_policy == 'thompson':
            backward_a = self.thompson_policy_sample()
            self.child_actions[backward_a].backward_n += 1
            counts = [child_action.backward_n for child_action in self.child_actions]
        return np.array(counts,dtype='float32')

    def ucb_backward_sample(self,c):
        ''' UCB sample for backward pass. Does not use sigma_tree. Note the -np.Inf in the U, which prevent selecting an untried action '''
        Q = np.array([child_action.Q for child_action in self.child_actions],dtype='float32')
        U = np.array([c * (np.sqrt(self.n)/(child_action.n)) if child_action.n > 0 else -np.Inf for child_action in self.child_actions],dtype='float32')
        scores = np.squeeze(Q + U)
        winner = my_argmax(scores)
        return winner

    def thompson_sample_n(self,n):
        ''' returns a vector of thompson sample frequencies of length len(self.child_actions) '''
        counts = [0 for i in range(len(self.child_actions))]
        for i in range(n):
            index = self.thompson_policy_sample()
            counts[index] += 1
        return counts

    def thompson_policy_sample(self):
        ''' Thompson sample for backward pass '''
        # not used right now
        samples = []
        for child_action in self.child_actions:
            if child_action.n > 0:
                samples.append(child_action.Q + np.random.normal(0,1)/np.sqrt(child_action.n))
            else:
                samples.append(-np.Inf) # cant select an untried action
        return my_argmax(np.array(samples))

def rollout(s,Env,policy,gamma,max_roll):
    ''' Small rollout function to estimate V(s)
    policy = random or targeted'''
    terminal = False
    R = 0.0
    for i in range(max_roll):
        a = Env.action_space.sample()
        s1,r,terminal,_ = Env.step(a)
        R += (gamma**i)*r
        s = s1
        #if i == 0:
            #a_init = a
#            s_i = s1
#            r_i = r
#            t_i = terminal
        if terminal:
            break
    return R#, a_init
        
def display_info(root,time,c):
    ''' Display MCTS node info for debugging '''
    if root is not None:
        print('MCTS status for timestep {}'.format(time))
        Q = [child_action.Q for child_action in root.child_actions]
        print('Q values: {}'.format(Q))
        print('counts: {}'.format([child_action.n for child_action in root.child_actions],[child_action.n for child_action in root.child_actions]))            
        U = [c * (np.sqrt(1 + root.n)/(1 + child_action.n)) for child_action in root.child_actions]
        print('U: {}'.format(U))
        scores = np.squeeze(np.array([Q]) + np.array([U]))
        print('scores: {}'.format(scores))
        print('winner: {}'.format(np.argwhere(scores == np.max(scores)).flatten()))             
        print('-----------------------------')
