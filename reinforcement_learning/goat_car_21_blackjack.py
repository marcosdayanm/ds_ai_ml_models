import random

i = 0

stay_score = 0
switch_score = 0

while i < 1000:
    options = [0, 0, 0]
    correct = random.randint(0, 2)
    options[correct] = 1

    guess = random.randint(0, 2)
    op = random.randint(0, 2)

    if op == guess or op == correct:
        continue

    options[op] = 1
