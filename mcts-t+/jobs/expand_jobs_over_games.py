# -*- coding: utf-8 -*-
"""
Expand a submission over games
@author: thomas
"""
import os
import argparse

def expand_job(games,job,hp,hp_setup):
    # hacky way to bring in games
    #games = ['CartPole-vr','MountainCar-vr','Acrobot-vr','FrozenLake-v0','FrozenLakeNotSlippery-v0','FrozenLakeNotSlippery-v1']
    games = ['Breakout-ramDeterministic-v0','Pong-ramDeterministic-v0','AirRaid-ramDeterministic-v0','Amidar-ramDeterministic-v0',
             'Enduro-ramDeterministic-v0','MontezumaRevenge-ramDeterministic-v0','Venture-ramDeterministic-v0']
    # Regarding Atari:
    # Assault, Freeway, Seaquest have different initial states

    file = os.getcwd() + '/' + job
    with open(file,'w') as fp:
        fp.write('#!/bin/sh\n')   
        for i,game in enumerate(games):
            fp.write('python3 submit.py --hpsetup game={},{} --hp {}'.format(game,hp_setup,hp))
            fp.write('\n')

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+',type=str,default=[])
    parser.add_argument('--job', default='job.sh')    
    parser.add_argument('--slurm_mode', default='off')    
    parser.add_argument('--hp', help='Hyperparameter configuration',default='')
    parser.add_argument('--hpsetup', help='Hyperparameter configuration of slurm and hyperparameters and distribution',default='')
    args = parser.parse_args()
    
    if args.slurm_mode == 'short':
        args.hpsetup += ',slurm=True,slurm_qos=short,slurm_time=3:59:59'
    elif args.slurm_mode == 'long':
        args.hpsetup += ',slurm=True,slurm_qos=long,slurm_time=5-0:00:00'
        
    expand_job(games=args.games,job=args.job,hp=args.hp,hp_setup=args.hpsetup)