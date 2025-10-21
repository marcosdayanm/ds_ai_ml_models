import random
import matplotlib.pyplot as plt

slot1 = {
    "p": 40,
    "a": 10
}

slot2 = {
    "p": 3,
    "a": 100
}

slot1_running_average = [0.0]
slot2_running_average = [0.0]

for i in range(1, 10001):
  slot1_result = slot1["a"] if random.random() * 100 < slot1["p"] else 0
  slot1_running_average.append(slot1_running_average[i-1] + (1/i)*(slot1_result - slot1_running_average[i-1]))

  slot2_result = slot2["a"] if random.random() * 100 < slot2["p"] else 0
  slot2_running_average.append(slot2_running_average[i-1] + (1/i)*(slot2_result - slot2_running_average[i-1]))


print(f"Final running average for Slot 1: {slot1_running_average[-1]}")
print(f"Final running average for Slot 2: {slot2_running_average[-1]}")



plt.figure(figsize=(10, 6))
plt.plot(slot1_running_average, label='Slot 1 Running Average')
plt.plot(slot2_running_average, label='Slot 2 Running Average')
plt.xlabel('Number of Iterations')
plt.ylabel('Running Average')
plt.title('Running Average of Slot Results Over Time')
plt.legend()
plt.grid(True)
plt.show()