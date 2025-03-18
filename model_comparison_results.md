# March Madness Prediction Model Comparison

This document compares the results of different prediction approaches for the March Madness First Four games, using various historical time frames.

## Model Approaches

1. **2025-Only Model**
   - Uses only current season (2025) data
   - Focus on current team performance without historical context
   - Advantages: Most relevant to current team composition
   - Disadvantages: Limited data points, no trend information

2. **Recent Model (2023-2025)**
   - Uses 3 most recent seasons (2023-2025)
   - Includes trend analysis component
   - Weighs recent seasons more heavily (2025 > 2024 > 2023)
   - Advantages: Balance of recency and sufficient data, trend detection
   - Disadvantages: Might miss longer historical patterns

3. **Extended Model (2018-2025)**
   - Uses 8 years of data (2018-2025)
   - Larger training dataset
   - Advantages: More robust statistical foundation
   - Disadvantages: May include less relevant historical data

4. **Comprehensive Model (2003-2025)**
   - Uses 20+ years of data (2003-2025)
   - Largest historical context
   - Advantages: Captures long-term patterns and tournament dynamics
   - Disadvantages: Includes potentially outdated data

## First Four Matchup Predictions

| Matchup | 2025-Only Model | Recent Model (2023-2025) | Extended Model (2018-2025) | Comprehensive Model (2003-2025) |
|---------|-----------------|--------------------------|----------------------------|--------------------------------|
| **North Carolina vs. San Diego State (#11)** | **North Carolina** (88.6%) | **North Carolina** (72.6%) | **North Carolina** (100%) | **North Carolina** (80.6%) |
| **Texas vs. Xavier (#11)** | **Texas** (79.0%) | **Xavier** (51.9%) | **Texas** (54.6%) | **Xavier** (67.8%) |
| **Alabama State vs. Saint Francis (#16)** | **Alabama State** (98.1%) | **Alabama State** (63.7%) | **Alabama State** (100%) | **Alabama State** (99.8%) |
| **American vs. Mount St. Mary's (#16)** | **American** (76.2%) | **American** (69.1%) | **American** (80.7%) | **American** (67.2%) |

## Key Observations

1. **Consensus Winners**
   - North Carolina over San Diego State (all models)
   - Alabama State over Saint Francis (all models)
   - American over Mount St. Mary's (all models)
   - Texas vs. Xavier shows disagreement between models

2. **Confidence Level Variations**
   - Recent model (2023-2025) generally has **lower confidence levels** than other models
   - Extended model (2018-2025) shows highest confidence levels overall
   - The 2025-only model has high confidence despite limited data points
   - Comprehensive model shows variable confidence levels

3. **Model-Specific Insights**
   - The recent model uniquely incorporates team performance trends
   - Texas vs. Xavier is the most uncertain matchup across all models
   - The 2025-only model is the only one highly confident in Texas over Xavier
   - The comprehensive model shows moderate confidence in Xavier over Texas

## Feature Importance Analysis

All models identified similar top predictive features:
1. Adjusted Efficiency Margin (differential)
2. Team Rankings (differential)
3. Offensive/Defensive Efficiency (differential)

However, the recent model also highlighted the importance of:
- Experience differential
- Average height differential
- Tempo differential
- Bench contribution differential

## Conclusion

The most reliable predictions appear to be those with consensus across multiple models:
- North Carolina over San Diego State
- Alabama State over Saint Francis
- American over Mount St. Mary's

For the Texas vs. Xavier matchup, the recent model with trend analysis suggests it's nearly a toss-up (51.9% confidence in Xavier), while other models are split on the winner with varying confidence levels. This indicates this particular matchup requires more careful analysis and is likely the most unpredictable of the First Four games.

The recent model (2023-2025) with trend analysis provides a balanced approach that avoids over-confidence while still incorporating sufficient historical context. Its unique consideration of team performance trends adds valuable insight not present in the other models.