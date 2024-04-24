import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_logits_and_multiple_probabilities(logits, t_list, probabilities_list):
  """Plot logits and multiple softmax probability distributions on the same graph."""
  # Set up the figure and axis
  fig, ax1 = plt.subplots(figsize=(10, 5))

  # Create bar chart for logits
  indices = np.arange(len(logits))  # the x locations for the groups
  ax1.bar(indices, logits, width=0.35, label='Logit values', color='#89CFF0', alpha=0.7)
  ax1.set_xlabel('Class index')
  ax1.set_ylabel('Logit values', color='#89CFF0')
  ax1.tick_params('y', colors='#89CFF0')

  # Create a second y-axis for the softmax probabilities
  ax2 = ax1.twinx()
  
  colors = ['g', 'r', 'b', 'y', 'm', 'c', 'k']  # Define a list of colors for the lines
  for i, probabilities in enumerate(probabilities_list):
    if t_list[i] == 1.0:
      ax2.plot(indices, probabilities, linewidth=4.0, label=f'Scaled Post-Softmax Probability T={t_list[i]}', color=colors[i % len(colors)])
    else:    
      ax2.plot(indices, probabilities, label=f'Scaled Post-Softmax Probability T={t_list[i]}', color=colors[i % len(colors)])
  
  ax2.set_ylabel('Class Probability', color='r')
  ax2.tick_params('y', colors='r')

  # Title and legend
  plt.title('Model Logits vs Temperature-Scaled Probabilities')
  fig.tight_layout()
  ax1.legend(loc='upper left')
  ax2.legend(loc='upper right')

  # Show the plot
  plt.show()

# Example logits
ts = [0.5, 1.0, 2.0, 4.0, 8.0]
logits = torch.tensor([3.1230, 5.0000, 3.2340, 2.6420, 2.4660, 3.3532, 3.8000, 2.9110])
probs = [torch.softmax(logits / t, dim=-1) for t in ts]

# Plotting the graph
plot_logits_and_multiple_probabilities(logits, ts, probs)
