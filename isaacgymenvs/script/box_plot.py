import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Given data, from csv file
data = pd.read_csv("./box_plot_data.csv")

# Convert to DataFrame
df = pd.DataFrame(data)

# Melting the DataFrame for better handling of boxplot with seaborn
df_melted = df.melt(
    id_vars=["env_name", "pt_name"], var_name="statistic", value_name="value"
)


def custom_scale(y):
    return np.where(y > 1.5, (y - 1.5) / 15 + 1.5, y)


def inverse_custom_scale(y):
    return np.where(y > 1.5, (y - 1.5) * 15 + 1.5, y)


# Apply the custom scale to the y-values
df_melted["custom_value"] = custom_scale(df_melted["value"])

# Creating the boxplot
plt.figure(figsize=(12, 8))
# sns.boxplot(
#     x="env_name", y="custom_value", hue="pt_name", whis=[0, 100], data=df_melted
# )

sns.boxplot(x="env_name", y="value", hue="pt_name", whis=[0, 100], data=df_melted)

yticks = plt.yticks()[0]
# make yticks starts at -0.5
# plt.yticks(ticks=yticks, labels=inverse_custom_scale(yticks).round(2))
plt.yticks(ticks=yticks, labels=yticks.round(2))
plt.title("Boxplot of Statistics by Environment and PT Name")
plt.xlabel("Environment Name")
plt.ylabel("Value")
# make axis only x and y
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
# add horizontal grid lines, dotted
plt.gca().yaxis.grid(True, linestyle="--")
plt.legend(title="PT Name")
plt.show()
