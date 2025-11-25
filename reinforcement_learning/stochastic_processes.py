import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt


params = np.array([0, 0], dtype=float)
params_hist = []
rewards_hist = []
rewards_running_avg = [0.0]
lr = .01

for i in range(1, 1001):
    r1 = np.random.choice([0, 1], p=[0.25, 0.75])
    r2 = np.random.choice([0, 1], p=[0.75, 0.25])

    action = np.random.choice([1, 2], p=softmax(params))

    reward = r1 if action == 1 else r2


    grad = np.array([0,0], dtype=float)

    grad[action - 1] = 1

    rewards_running_avg.append(rewards_running_avg[i-1] + (1/i)*(reward - rewards_running_avg[i-1])) 
    params += lr*(grad-softmax(params))*(reward-rewards_running_avg[i])
    params_hist.append(params.copy())
    rewards_hist.append(reward)



print("Final parameters:", params)
print("Average reward:", np.mean(rewards_hist))


plt.figure()
# plt.plot([p[0] for p in params_hist], label='Param 1')
# plt.plot([p[1] for p in params_hist], label='Param 2')
plt.plot(params_hist, label=['Param 1', 'Param 2'])
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title('Parameter values over time')
plt.legend()
plt.show()