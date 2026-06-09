import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the data using the updated regex separator
file_path = "data.txt"  # Make sure to replace this with your actual file name if it's different!
data = pd.read_csv(file_path, sep=r"\s+", header=None, names=["X", "Y"])
print(data["X"])
print(data["Y"])
# 2. Create the plot
plt.figure(figsize=(8, 6))
plt.plot(data["Y"],data["X"], marker="o", linestyle="-", color="b", label="Data Path")

# 3. Customize and beautify
plt.title("Plot of X vs Y", fontsize=14, fontweight="bold")
plt.xlabel("X Axis Label", fontsize=12)
plt.ylabel("Y Axis Label", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.axis('equal')
plt.legend()

# 4. Show the plot
plt.show()