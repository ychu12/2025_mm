import matplotlib.pyplot as plt
import numpy as np

# Matchup data for visualization
matchups = [
    "UNC vs SDSU", 
    "Texas vs Xavier",
    "Alabama St vs St Francis", 
    "American vs Mt St Mary's"
]

# Confidence values for each model and prediction (all values show confidence in eventual winner)
current_only = [81.6, 53.3, 100.0, 77.3]  # 2025 only
recent = [87.3, 67.1, 60.7, 58.2]         # 2023-2025
extended = [82.1, 49.7, 66.9, 78.5]       # 2018-2025 (Note: adjusted Texas-Xavier to show confidence in Texas)
comprehensive = [87.4, 63.8, 70.8, 68.0]  # 2003-2025
ensemble = [84.8, 59.8, 72.8, 68.0]       # Ensemble prediction

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Width of the bars
width = 0.15

# Positions of the bars on the x-axis
r1 = np.arange(len(matchups))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]
r4 = [x + width for x in r3]
r5 = [x + width for x in r4]

# Create the bars
bars1 = ax.bar(r1, current_only, width, label='2025 Only', color='#3498db', alpha=0.8)
bars2 = ax.bar(r2, recent, width, label='Recent (2023-2025)', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(r3, extended, width, label='Extended (2018-2025)', color='#e74c3c', alpha=0.8)
bars4 = ax.bar(r4, comprehensive, width, label='Comprehensive (2003-2025)', color='#9b59b6', alpha=0.8)
bars5 = ax.bar(r5, ensemble, width, label='Ensemble', color='#f39c12', alpha=1.0, edgecolor='black', linewidth=1.5)

# Add some text for labels, title and axes
ax.set_ylabel('Confidence in Predicted Winner (%)', fontsize=12)
ax.set_title('March Madness First Four: Model Confidence Comparison', fontsize=14)
ax.set_xticks([r + 2 * width for r in range(len(matchups))])
ax.set_xticklabels(matchups, fontsize=10)
ax.set_ylim(0, 105)

# Add a thin horizontal line at y=50 to show toss-up threshold
ax.axhline(y=50, color='black', linestyle='--', alpha=0.3)
ax.text(3.7, 51, 'Toss-up threshold', fontsize=8)

# Add confidence thresholds with light background colors
ax.axhspan(80, 105, alpha=0.1, color='green', label='High Confidence')
ax.axhspan(65, 80, alpha=0.1, color='lightgreen', label='Medium-High Confidence')
ax.axhspan(55, 65, alpha=0.1, color='yellow', label='Medium Confidence')
ax.axhspan(50, 55, alpha=0.1, color='orange', label='Low Confidence')

# Add confidence level labels
ax.text(3.7, 95, 'High', fontsize=8)
ax.text(3.7, 75, 'Medium-High', fontsize=8)
ax.text(3.7, 60, 'Medium', fontsize=8)
ax.text(3.7, 52.5, 'Low', fontsize=8)

# Add value labels on the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)
add_labels(bars5)

# Add bold outline to ensemble model bars
for bar in bars5:
    bar.set_linewidth(1.5)
    bar.set_edgecolor('black')

# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

# Add a grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Add model consensus information
consensus_values = [100, 75, 100, 100]  # Percentage of models agreeing with ensemble prediction
for i, cons in enumerate(consensus_values):
    x_pos = r5[i]
    ax.annotate(f"{cons}% consensus",
                xy=(x_pos, 5),
                xytext=(0, -15),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Adjust layout and save
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the legend at bottom
plt.savefig('ensemble_comparison.png', dpi=300)
plt.close()

# Create a pie chart showing the Ensemble model weights
fig, ax = plt.subplots(figsize=(8, 8))
weights = [25, 45, 20, 10]  # Current, Recent, Extended, Comprehensive
labels = ['2025 Only (25%)', 'Recent 2023-2025 (45%)', 'Extended 2018-2025 (20%)', 'Comprehensive 2003-2025 (10%)']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
explode = (0, 0.1, 0, 0)  # Explode the 2nd slice (Recent model)

# Create pie chart
wedges, texts, autotexts = ax.pie(
    weights, 
    explode=explode, 
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    shadow=True, 
    startangle=90,
    textprops={'fontsize': 12}
)

# Make the percentage text white for better visibility
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  
plt.title('Ensemble Model Component Weights', fontsize=16)
plt.tight_layout()
plt.savefig('ensemble_weights.png', dpi=300)

print("Created visualizations: ensemble_comparison.png and ensemble_weights.png")