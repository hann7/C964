import pandas as pd
import matplotlib.pyplot as plt

subset_file_path = '../data/sample_200_labels.csv'
full_file_path = '../data/trainLabels.csv'

df_subset = pd.read_csv(subset_file_path)
df_full = pd.read_csv(full_file_path)

# gather basic statistics about the data
subset_stats = df_subset['level'].describe(include='all')
full_stats = df_full['level'].describe(include='all')

stats_df = pd.DataFrame({
    'Subset Data Stats': subset_stats,
    'Full Data Stats': full_stats
})

print(stats_df)

# number of images for each severity level
subset_sum_severity_levels = df_subset.groupby(by='level').size()
full_sum_severity_levels = df_full.groupby(by='level').size()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

subset_sum_severity_levels.plot.bar(
    ax=axes[0, 0],
    title="Number of Images Within Each Severity Level - SUBSET"
)
axes[0, 0].set_xlabel("Severity Level")
axes[0, 0].set_ylabel("Number of Images")

full_sum_severity_levels.plot.bar(
    ax=axes[0, 1],
    title="Number of Images Within Each Severity Level - FULL"
)
axes[0, 1].set_xlabel("Severity Level")
axes[0, 1].set_ylabel("Number of Images")

subset_sum_severity_levels.plot.pie(
    ax=axes[1, 0],
    autopct='%1.1f%%',
    startangle=90,
    title="Percentage Distribution of Severity Levels - Subset",
    legend=True,
    labels=None,
    pctdistance=1.2
)

full_sum_severity_levels.plot.pie(
    ax=axes[1, 1],
    autopct='%1.1f%%',
    startangle=90,
    title="Percentage Distribution of Severity Levels - Full",
    legend=True,
    labels=None,
    pctdistance=1.2
)

plt.tight_layout()
plt.show()

# add a column called side to track whether the eye is the left or right
df_subset['side'] = df_subset['image'].apply(lambda x: 'left' if 'left' in x else 'right')
df_full['side'] = df_full['image'].apply(lambda x: 'left' if 'left' in x else 'right')

# get the count of left vs right eye pics
severity_counts_subset = df_subset.groupby(['level', 'side']).size().unstack(fill_value=0)
severity_counts_full = df_full.groupby(['level', 'side']).size().unstack(fill_value=0)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

severity_counts_subset.plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_xlabel('Severity Level')
axes[0].set_ylabel('Count of Images')
axes[0].set_title('Severity Levels for Left vs. Right Eye Images - Subset')
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 5), textcoords='offset points')

severity_counts_full.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_xlabel('Severity Level')
axes[1].set_ylabel('Count of Images')
axes[1].set_title('Severity Levels for Left vs. Right Eye Images - Full')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()
