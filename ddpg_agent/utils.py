
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

def get_state(losses, epochs, num_samples, clients_id, M_matrix):
    losses = np.asarray(losses).reshape((len(epochs), 1))
    epochs = np.asarray(epochs).reshape((len(epochs), 1))
    num_samples = np.asarray(num_samples).reshape((len(num_samples), 1))/100
    clients_id = np.asarray(clients_id).reshape((len(epochs), 1))

    # print('break point here')
    retval = np.hstack((losses, epochs, num_samples, clients_id))
    retval = np.hstack((retval, M_matrix)).flatten()
    
    return retval

def get_state(start_loss, final_loss, std_local_losses, epochs, num_samples, clients_id, M_matrix):
    # losses = np.asarray(losses).reshape((len(epochs), 1))
    start_loss = np.asarray(start_loss).reshape((len(epochs), 1))
    final_loss = np.asarray(final_loss).reshape((len(epochs), 1))

    std_local_losses = np.asarray(std_local_losses).reshape((len(epochs), 1))

    normalized_start_loss = start_loss/(np.sum(start_loss))
    normalized_final_loss = final_loss/(np.sum(final_loss))

    num_samples = np.asarray(num_samples).reshape((len(num_samples), 1))/100
    normalized_samples = num_samples/(np.sum(num_samples))

    retval = np.hstack((normalized_start_loss, normalized_final_loss, normalized_samples)).flatten()
    retval = np.hstack((retval, M_matrix)).flatten()

    return retval

def get_reward(losses, M_matrix, beta=0.45):
    # beta = 0.45
    losses = np.asarray(losses)
    # return - beta * np.mean(losses) - (1 - beta) * np.std(losses)
    return - np.mean(losses) - (losses.max() - losses.min()) + 0.05 * np.sum(M_matrix)/2


def get_info_from_dqn_weights(weights, num_clients, dqn_list_epochs):
    client_dicts = {}
    for cli in range(num_clients):
        cli_dict = {}
        cli_dict["mean"] = weights[0, 2*cli]
        cli_dict["std"] = weights[0, 2*cli+1]
        cli_dict["epoch"] = dqn_list_epochs[cli]
        client_dicts[cli] = cli_dict
    return client_dicts




