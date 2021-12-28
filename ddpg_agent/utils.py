
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def get_state(losses, epochs, num_samples):
    losses = np.asarray(losses).reshape((len(epochs), 1))
    epochs = np.asarray(epochs).reshape((len(epochs), 1))
    num_samples = np.asarray(num_samples).reshape((len(num_samples), 1))/100
    # print('break point here')
    retval = np.hstack((losses, epochs, num_samples)).flatten()
    return retval

def get_reward(losses):
    losses = np.asarray(losses)
    return -np.mean(losses) - np.std(losses)


