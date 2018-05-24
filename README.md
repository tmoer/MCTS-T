# MCTS-T(+)
Code for the paper [Monte Carlo Tree Search for Asymmetric Trees](https://arxiv.org/pdf/1805.09218.pdf) by Thomas M. Moerland, Joost Broekens, Aske Plaat and Catholijn M. Jonker. 

## Prerequisites
1. Install recent versions of:
- Python 3
- Tensorflow   
- Numpy
- Matplotlib

2. Clone this repository:
```sh
git clone https://github.com/tmoer/mcts-t.git
```
## Syntax
You can run a new experiment from the agent.py function. Hyperparameters can be parsed through the --hp option. Default hyperparameters are listed in mcts-t+/hps.py. For example, to start a default experiment on CartPole-v0:
```sh
cd mcts-t+
python3 agent.py --hp game=CartPole-v0
```

## Reproducing Paper Results
The results of the paper can be reproduced by:
```sh
cd mcts-t+
bash jobs/paper_jobs.sh
``` 
This automatically loop over the necessary hyperparameters. Running it will take quite long on a regular laptop though. You can submitted the runs to a SLURM cluster via
```sh
bash jobs/paper_jobs_slurm.sh
``` 

## Visualization of Results
Subsequently, you can visualize the output with 
```sh
cd mcts-t+
python3 visualize.py --home --plot_type mean --game your_game 
``` 
for some your_game of your choice. 

## Citation
```
@proceedings{moerland2018monte,
	author = "Moerland, Thomas M and Broekens, Joost and Plaat, Aske and Jonker, Catholijn M",
	journal = "arXiv preprint arXiv:1805.09218",
	title = "{Monte Carlo Tree Search for Asymmetric Trees}",
	year = "2018"
}
```


