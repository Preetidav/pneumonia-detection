import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("training_result.csv")

# Separate Train and Val data
train_df = df[df['Phase'] == 'Train']
val_df = df[df['Phase'] == 'Val']

# Set the plot style
plt.style.use("ggplot")

# Define a function to plot metrics
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(train_df['Epoch'], train_df[metric_name], label='Train', marker='o')
    plt.plot(val_df['Epoch'], val_df[metric_name], label='Validation', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric_name.lower()}_plot.png") 
    plt.show()

# Plot all metrics
plot_metric('Loss', 'Loss')
plot_metric('Accuracy', 'Accuracy')
plot_metric('Precision', 'Precision')
plot_metric('Recall', 'Recall')
plot_metric('AUC', 'AUC')
