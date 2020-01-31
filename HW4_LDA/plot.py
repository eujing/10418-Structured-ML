import matplotlib.pyplot as plt
import json

with open("results.json", "r") as f:
    results = json.load(f)

log_likes = results["log_likes"]
iters = range(1, 10 * len(log_likes) + 1, 11)

plt.plot(iters, log_likes)
plt.xlabel("Iteration")
plt.ylabel("Joint Log Likelihood")
plt.title("Data Log Likelihood over Iterations")
plt.show()
