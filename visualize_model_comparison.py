import matplotlib.pyplot as plt
import numpy as np

# Data for the visualizations
matchups = [
    "UNC vs SDSU",
    "Texas vs Xavier",
    "Alabama St vs St Francis",
    "American vs Mt St Mary's"
]

# Confidence values for each model (in %)
recent_model = [72.6, 51.9, 63.7, 69.1]  # 2023-2025
current_year = [88.6, 79.0, 98.1, 76.2]  # 2025 only
extended = [100.0, 54.6, 100.0, 80.7]    # 2018-2025
comprehensive = [80.6, 67.8, 99.8, 67.2] # 2003-2025

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Width of the bars
width = 0.2

# Positions of the bars on the x-axis
r1 = np.arange(len(matchups))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]
r4 = [x + width for x in r3]

# Create the bars
bars1 = ax.bar(r1, current_year, width, label='2025 Only', color='#3498db')
bars2 = ax.bar(r2, recent_model, width, label='Recent (2023-2025)', color='#2ecc71')
bars3 = ax.bar(r3, extended, width, label='Extended (2018-2025)', color='#e74c3c')
bars4 = ax.bar(r4, comprehensive, width, label='Comprehensive (2003-2025)', color='#9b59b6')

# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel('Confidence (%)', fontsize=12)
ax.set_title('March Madness First Four Prediction Confidence by Model', fontsize=14)
ax.set_xticks([r + 1.5 * width for r in range(len(matchups))])
ax.set_xticklabels(matchups, fontsize=10)
ax.set_ylim(0, 110)  # To give some room above 100%

# Add a thin horizontal line at y=50 to show toss-up threshold
ax.axhline(y=50, color='black', linestyle='--', alpha=0.3)
ax.text(3.8, 51, 'Toss-up threshold', fontsize=8)

# Add confidence thresholds with light background colors
ax.axhspan(80, 110, alpha=0.1, color='green')
ax.axhspan(65, 80, alpha=0.1, color='lightgreen')
ax.axhspan(55, 65, alpha=0.1, color='yellow')
ax.axhspan(50, 55, alpha=0.1, color='orange')

# Add confidence level labels
ax.text(3.8, 95, 'Very High Confidence', fontsize=8)
ax.text(3.8, 75, 'High Confidence', fontsize=8)
ax.text(3.8, 60, 'Medium Confidence', fontsize=8)
ax.text(3.8, 52.5, 'Low Confidence', fontsize=8)

# Add value labels above the bars
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

# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

# Add a grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_confidence_comparison.png', dpi=300)
plt.close()

# Create a second visualization showing consensus
fig, ax = plt.subplots(figsize=(10, 6))

# Winners by model
winners = {
    "UNC vs SDSU": ["UNC", "UNC", "UNC", "UNC"],
    "Texas vs Xavier": ["Texas", "Xavier", "Texas", "Xavier"],
    "Alabama St vs St Francis": ["Alabama St", "Alabama St", "Alabama St", "Alabama St"],
    "American vs Mt St Mary's": ["American", "American", "American", "American"]
}

# Calculate consensus for each matchup
consensus = []
for matchup in matchups:
    win_count = {}
    for winner in winners[matchup]:
        if winner not in win_count:
            win_count[winner] = 0
        win_count[winner] += 1
    
    # Get max agreement percentage (25% per model, up to 100% for full consensus)
    max_consensus = max(win_count.values()) * 25
    consensus.append(max_consensus)

# Create consensus bars
bars = ax.bar(matchups, consensus, color=['green' if c == 100 else 'orange' for c in consensus])

# Add some text for labels, title and axes
ax.set_ylabel('Model Consensus (%)', fontsize=12)
ax.set_title('First Four Prediction Consensus Across All Models', fontsize=14)
ax.set_ylim(0, 110)

# Add value labels above the bars
for bar in bars:
    height = bar.get_height()
    winner = ""
    for matchup in winners:
        if matchup == bar.get_x():
            most_common = max(winners[matchup], key=winners[matchup].count)
            winner = most_common
            break
    
    ax.annotate(f'{height:.0f}% ({winner})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Add a grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_consensus.png', dpi=300)
print("Visualizations created: model_confidence_comparison.png and model_consensus.png")