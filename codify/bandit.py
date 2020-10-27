import numpy as np
import matplotlib.pyplot as plt

def exp3_distr(weights, gamma):
    normalizing = sum(weights)
    return [(1-gamma) * (w/normalizing) + (gamma/len(weights)) for w in weights]

# https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
def exp3(num_actions, reward, gamma, weights, t):
    '''
    :param num_actions:
    :param reward:
    :param gamma:  in [0, 1]. Higher gamma = more exploration
    :return:
    '''
    probs = exp3_distr(weights, gamma)
    choice = np.random.choice(range(num_actions), 1, p=probs)[0]
    #print('choice:', choice)
    reward = reward(choice, t)
    # https://www.youtube.com/watch?v=N5x48g2sp8M&t=2574s
    # inverse propensity score ==> correct for selection bias
    unbiased_reward = reward/probs[choice]
    #print('weight:', weights[choice], '->', end='')
    weights[choice] *= np.exp(unbiased_reward * gamma/num_actions)
    #print(weights[choice])
    return probs

# experiment. 10 slot machines
arm_probs = [1.0/i for i in range(2, 12)]
num_actions = len(arm_probs)
T = 10000
rewards = [[1 if np.random.random() < prob else 0 for prob in arm_probs] for _ in range(T)]
reward = lambda choice, t: rewards[t][choice]

bestAction = max(range(num_actions), key=lambda action: sum([rewards[t][action] for t in range(T)]))
print('best:', bestAction)

running_weigths = []
gamma = 0.07
weights = [1.0]* num_actions # initial weights

for t in range(T):
    probs = exp3(num_actions, reward, gamma, weights, t)
    running_weigths.append(probs)

running_weigths = np.array(running_weigths)
print(running_weigths.shape)

for action in range(num_actions):
    plt.plot(running_weigths[:, action], label=action)

print(running_weigths)
plt.legend()
plt.show()