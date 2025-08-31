import matplotlib.pyplot as plt

# bias, x1, x2, y]
or_table = [
    [1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
]



alpha = 0.1      # learning rate
epochs = 100      # max epochs

# Weights: [w_bias, w1, w2]
w = [1.5, 0.5, 1.5]

def step(z):
    return 1 if z > 0 else 0

def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

for epoch in range(epochs):
    total_errors = 0

    for row in or_table:
        x = row[:3]           # [1, x1, x2] includes bias input
        target = row[3]       # desired output

        u = dot(w, x)         # weighted sum
        y_pred = step(u)      # perceptron output

        error = target - y_pred
        if error != 0:
            for j in range(len(w)):
                w[j] += alpha * error * x[j]
            total_errors += 1

    if total_errors == 0:
        # early stop when perfectly classified
        break

print("Final weights:", w, "at epoch:", epoch)