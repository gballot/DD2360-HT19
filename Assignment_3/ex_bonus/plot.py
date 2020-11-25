import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./result.csv", index_col="size")

# Format for latex table
data["latex"] = data.index.to_series().apply(str) + " * " + data.index.to_series().apply(str) + " & " + (data.index.to_series() * data.index.to_series()).apply(str) + " & " +  data["cpu_time"].apply(str) + " & " + data["gpu_naive"].apply(str) + " & " + data["gpu_shared"].apply(str) + " & " + data["cublas"].apply(str) + "\\\\"

pd.set_option('display.max_colwidth', 50000)
print(data["latex"].to_string(index=False))

# diagrams

data.index = data.index * data.index
data[["cpu_time", "gpu_shared", "cublas"]].plot()
plt.xlabel("Number entries")
plt.ylabel("Execution time (ms)")
plt.show()
