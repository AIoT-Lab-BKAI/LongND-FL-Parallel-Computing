
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


def plot(frame_idx, rewards):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    if not os.path.exists('./log/images/'):
        os.makedirs('./log/images/')
    plt.savefig('./log/images/'+date_time)
    plt.show()

def get_state(losses, epochs, num_samples, clients_id):
    losses = np.asarray(losses).reshape((len(epochs), 1))
    epochs = np.asarray(epochs).reshape((len(epochs), 1))
    num_samples = np.asarray(num_samples).reshape((len(num_samples), 1))/100
    clients_id = np.asarray(clients_id).reshape((len(epochs), 1))
    # print('break point here')
    retval = np.hstack((losses, epochs, num_samples, clients_id)).flatten()
    return retval

def get_reward(losses, beta):
    # beta = 0.45
    losses = np.asarray(losses)
    return - beta * np.mean(losses) - (1 - beta) * np.std(losses)

def get_info_from_dqn_weights(weights, num_clients, dqn_list_epochs):
    client_dicts = {}
    for cli in range(num_clients):
        cli_dict = {}
        cli_dict["mean"] = weights[0, 2*cli]
        cli_dict["std"] = weights[0, 2*cli+1]
        cli_dict["epoch"] = dqn_list_epochs[cli]
        client_dicts[cli] = cli_dict
    return client_dicts




