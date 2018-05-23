#!/bin/sh

python3 submit.py --hpsetup name=reruns,game=Chains,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma,item3=game,seq3=Chain-10+Chain-25+Chain-50+Chain-100,n_rep=1 --hp n_eps=50,n_t=1000000

python3 submit.py --hpsetup name=reruns,game=Chainloops,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma+sigma_loop,item3=game,seq3=ChainLoop-10+ChainLoop-25+ChainLoop-50+ChainLoop-100,n_rep=1 --hp n_eps=50,n_t=1000000

python3 submit.py --hpsetup name=reruns,game=CartPole-v0r,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma+sigma_loop,n_rep=50 --hp n_eps=1,n_t=1000000,c=1.0

python3 submit.py --hpsetup name=reruns,game=FrozenLakeNotSlippery-v1,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma+sigma_loop,n_rep=1 --hp n_eps=50,n_t=1000000,c=1.0

python3 submit.py --hpsetup game=Breakout-ramDeterministic-v0,name=atari_runs,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma+sigma_loop,n_rep=20 --hp n_eps=1,n_t=1000000,c=4.0

python3 submit.py --hpsetup game=Pong-ramDeterministic-v0,name=atari_runs,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma+sigma_loop,n_rep=20 --hp n_eps=1,n_t=1000000,c=4.0

python3 submit.py --hpsetup game=AirRaid-ramDeterministic-v0,name=atari_runs,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma+sigma_loop,n_rep=20 --hp n_eps=1,n_t=1000000,c=4.0

python3 submit.py --hpsetup game=Amidar-ramDeterministic-v0,name=atari_runs,item1=n_mcts,seq1=5+10+25+50+100+250,item2=mode,seq2=off+sigma+sigma_loop,n_rep=20 --hp n_eps=1,n_t=1000000,c=4.0

