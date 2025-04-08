import sys
import torch
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.arguments import get_args
from agents.sequecing_brain_ppo import Sequencing_brain

def train(m, wc, length_list, tightness, add_job, total_episode, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = Sequencing_brain(m, wc, length_list, tightness, add_job, **hyperparameters)
	total_episode = total_episode
	model.train(total_steps = total_episode)
	print(model.tard) #observing tard value

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	# model.learn(total_timesteps=200_000_000)

#TO-DO nothing here, reserved in case future we need this
def test(actor_model):
     return

def main(args):
    """
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
    hyperparameters = {
                'timespan': 1000,
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'input_size': 25
			  }
    total_episode = 1

    if args.mode == 'train':
        m = [6,12,24]
        wc = [3, 4, 6]
    	# lst = [2 for _ in range(3)]
        length_list = [[2, 2, 2],[3, 3, 3, 3],[4, 4, 4, 4, 4, 4]]
        tightness = [0.6, 1.0, 1.6]
        add_job = [50,200]
        total_episode = 1

    for i in range(len(tightness)):
        for j in range(len(length_list)):
            for k in range(len(add_job)):
                if i == 1 and j == 1 and k == 1:
                    train(m = m[i], wc = wc[i], length_list = length_list[i], tightness = tightness[j], add_job = add_job[k], \
                           total_episode = total_episode, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(actor_model=args.actor_model)

    

    # downwards are loss info and the most important var we want to optim: tard
    # print(sequencing_brain.tard)
    #plot_loss(sequencing_brain.tard)
    #plot_loss(sequencing_brain.actor_losses)
    #plot_loss(sequencing_brain.critic_losses)


if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)
    