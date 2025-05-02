# Step 1: Import Libraries
import pandas as pd
import sys
from dateutil import parser
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Step 2: Configure Console Encoding
sys.stdout.reconfigure(encoding='utf-8')

# Step 3: Set Up Chart Styling
sns.set_style('whitegrid')
sns.set_palette("tab20", 20)  # Use a distinct color palette with 20 unique colors
plt.rcParams['font.family'] = 'Arial'  # Set font to Arial for Hebrew support
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper minus sign rendering

# Function to reverse Hebrew text for correct display
def reverse_hebrew(text):
    if any(0x0590 <= ord(char) <= 0x05FF for char in text):
        words = text.split()
        reversed_words = []
        for word in words:
            if all(0x0590 <= ord(char) <= 0x05FF or char in '־-/' for char in word):
                reversed_words.append(''.join(reversed(word)))
            else:
                reversed_words.append(word)
        return ' '.join(reversed_words)
    return text

# Step 4: Load the Dataset
try:
    df = pd.read_csv('ED_full_data.csv', encoding='utf-8')
    print("Columns in df after loading:", df.columns.tolist())  # Debug: Print column names
except FileNotFoundError:
    print("Error: 'ED_full_data.csv' not found. Please ensure the file is in the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Step 4.5: Remove Data Before July 1, 2024
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
initial_na_count = df['timestamp'].isna().sum()
if initial_na_count > 0:
    print(f"\nDropping {initial_na_count} rows with unparseable timestamps before filtering:")
    df = df.dropna(subset=['timestamp'])
cutoff_date = pd.Timestamp('2024-07-01', tz='UTC')
initial_row_count = len(df)
df = df[df['timestamp'] >= cutoff_date]
filtered_row_count = len(df)
print(f"\nRemoved data before July 1, 2024:")
print(f"Rows before filtering: {initial_row_count}")
print(f"Rows after filtering: {filtered_row_count}")
print(f"Rows removed: {initial_row_count - filtered_row_count}")
print(f"New earliest timestamp: {df['timestamp'].min()}")

# Step 5: Analyze Timestamps Before Cleaning
timestamp_column = 'timestamp'
print("\nTimestamp Format Analysis:")
print("Sample of unique timestamps:\n", df[timestamp_column].unique()[:10])
null_before = df[timestamp_column].isnull().sum()
print(f"Null timestamps before cleaning: {null_before}")

# Step 6: Make a Copy of Timestamps
df['timestamp_original'] = df[timestamp_column].copy()

# Step 7: Fix Timestamp Formats
def standardize_timestamp_string(ts):
    if pd.isna(ts):
        return ts
    ts = str(ts).strip()
    if '+' not in ts and '-' not in ts[10:] and 'Z' not in ts:
        ts = ts + '+00:00'
    elif ts.endswith('Z'):
        ts = ts.replace('Z', '+00:00')
    return ts

df[timestamp_column] = df[timestamp_column].apply(standardize_timestamp_string)

# Step 8: Convert Timestamps to Date Format
df['timestamp_parsed'] = pd.to_datetime(df[timestamp_column], errors='coerce', utc=True)

# Step 9: Handle Unparseable Timestamps
mask = df['timestamp_parsed'].isna()
if mask.any():
    print(f"\nAttempting to parse {mask.sum()} timestamps with dateutil parser")
    def try_dateutil_parse(ts):
        if pd.isna(ts):
            return pd.NaT
        try:
            return parser.parse(ts, fuzzy=True)
        except:
            return pd.NaT
    df.loc[mask, 'timestamp_parsed'] = df.loc[mask, 'timestamp_original'].apply(try_dateutil_parse)

# Step 10: Check Timestamp Parsing Results and Drop NaT Rows
nat_count = df['timestamp_parsed'].isna().sum()
print(f"\nRemaining unparseable timestamps: {nat_count}")
if nat_count > 0:
    print("Sample of problematic original timestamps:")
    problem_samples = df[df['timestamp_parsed'].isna()]['timestamp_original'].sample(min(5, nat_count)).tolist()
    for sample in problem_samples:
        print(f"  - '{sample}'")
    df['timestamp_parse_failed'] = df['timestamp_parsed'].isna()
else:
    print("All timestamps parsed successfully!")
df = df.dropna(subset=['timestamp_parsed'])

# Step 11: Ensure Timestamps Are in UTC
df['timestamp_parsed'] = pd.to_datetime(df['timestamp_parsed'], utc=True)

# Step 12: Update the Timestamp Column
df[timestamp_column] = df['timestamp_parsed']

# Step 13: Remove Temporary Columns
df = df.drop(columns=['timestamp_original', 'timestamp_parsed'])

# Step 14: Verify Cleaned Timestamps
print("\nCleaned timestamp data sample:")
print(df[timestamp_column].head())
null_after = df[timestamp_column].isnull().sum()
print(f"Null timestamps after cleaning: {null_after}")

if not df[timestamp_column].isna().all():
    print("\nTimestamp statistics:")
    print(f"Min date: {df[timestamp_column].min()}")
    print(f"Max date: {df[timestamp_column].max()}")
    print(f"Date range: {df[timestamp_column].max() - df[timestamp_column].min()}")

# Step 14.5: Add 'day_of_the_week' Column in English (Israel Context)
df['timestamp_local'] = df['timestamp'].dt.tz_convert('Asia/Jerusalem')
english_days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['day_of_the_week'] = df['timestamp_local'].dt.dayofweek.map(english_days)
df = df.drop(columns=['timestamp_local'])
print("\nSample of 'day_of_the_week' column:")
print(df[['timestamp', 'day_of_the_week']].head())

# Step 15: Analyze Letters in Room Names
unique_rooms = df['room'].unique()
unique_letters = set()
for room in unique_rooms:
    for char in str(room):
        if char.isalpha():
            unique_letters.add(char)
print("\nUnique letters in 'room' column:", sorted(unique_letters))

# Step 16: Categorize Rooms
def is_one_letter_before_number(room):
    room = str(room).strip()
    pattern = r'^[A-Za-zא-ת]\d+$'
    return bool(re.match(pattern, room))

one_letter_rooms = set(room for room in unique_rooms if is_one_letter_before_number(room))
other_rooms = set(room for room in unique_rooms if not is_one_letter_before_number(room))
print("\nRooms with 1 letter before a number (Pattern Rooms):")
print(sorted(one_letter_rooms))
print(f"Number of pattern rooms: {len(one_letter_rooms)}\n")
print("Other rooms:")
print(sorted(other_rooms))
print(f"Number of other rooms: {len(other_rooms)}\n")

# Step 16.5: Count Rows for "Other Rooms" and Remove Them
other_rooms_list = list(other_rooms)
other_rooms_count = df['room'].isin(other_rooms_list).sum()
print(f"\nNumber of rows with 'other rooms': {other_rooms_count}")
print(f"Percentage of total rows: {(other_rooms_count / len(df) * 100):.2f}%")
df_pattern = df[~df['room'].isin(other_rooms_list)].copy()
print(f"\nNumber of rows after removing 'other rooms': {len(df_pattern)}")
print(f"Percentage of original rows retained: {(len(df_pattern) / len(df) * 100):.2f}%")

# Debug: Print columns of df_pattern
print("Columns in df_pattern after Step 16.5:", df_pattern.columns.tolist())

# Step 17: Add 'Shifts' and 'Distance' Columns to Pattern Rooms
def add_shift_and_distance_columns(df):
    df['timestamp_local'] = df['timestamp'].dt.tz_convert('Asia/Jerusalem')
    hour_local = df['timestamp_local'].dt.hour
    conditions = [
        (hour_local >= 6) & (hour_local <= 13),
        (hour_local >= 14) & (hour_local <= 21),
        (hour_local >= 22) | (hour_local <= 5)
    ]
    shift_values = [1, 2, 3]
    df['shifts'] = np.select(conditions, shift_values, default=np.nan)
    df['shifts'] = df['shifts'].astype('Int64')
    distance_conditions = [
        df['room'].str.contains('T', case=True, na=False),
        df['room'].str.contains('A', case=True, na=False),
        df['room'].str.contains('B', case=True, na=False),
        df['room'].str.contains('C', case=True, na=False)
    ]
    distance_values = [1, 2, 3, 4]
    df['distance'] = np.select(distance_conditions, distance_values, default=np.nan)
    df['distance'] = df['distance'].astype('Int64')
    df = df.drop(columns=['timestamp_local'])
    return df

df_pattern = add_shift_and_distance_columns(df_pattern)

# Step 18: Visualize Room Usage for Pattern Rooms
plt.figure(figsize=(10, 6))
df_pattern['room'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Most Used Pattern Rooms')
plt.xlabel('Room')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_10_pattern_rooms.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df_pattern, x='shifts')
plt.title('Pattern Room Usage by Shift')
plt.xlabel('Shift (1: 06:00-13:59, 2: 14:00-21:59, 3: 22:00-05:59)')
plt.ylabel('Number of Entries')
plt.tight_layout()
plt.savefig('pattern_room_usage_by_shift.png')
plt.show()

# Step 19: Analyze Time Sinks in Pattern Rooms (Delays in Status Changes)
required_df = df_pattern[df_pattern['status'] == 'required'].copy()
satisfied_df = df_pattern[df_pattern['status'] == 'satisfied'].copy()

# Debug: Print columns to confirm 'timestamp' exists
print("Columns in required_df:", required_df.columns.tolist())
print("Columns in satisfied_df:", satisfied_df.columns.tolist())

# Sort both dataframes by timestamp
required_df = required_df.sort_values('timestamp')
satisfied_df = satisfied_df.sort_values('timestamp')

# Explicitly rename 'timestamp' in satisfied_df to 'timestamp_sat'
satisfied_df = satisfied_df.rename(columns={'timestamp': 'timestamp_sat'})

# Use merge_asof to find the next satisfied timestamp for each required
status_changes_full = pd.merge_asof(
    required_df,
    satisfied_df[['room', 'requirement', 'timestamp_sat']],
    left_on='timestamp',
    right_on='timestamp_sat',
    by=['room', 'requirement'],
    direction='forward'
)

# Debug: Print columns of status_changes_full to confirm 'timestamp_sat' exists
print("Columns in status_changes_full:", status_changes_full.columns.tolist())

# Extract completed assignments and calculate time to completion
status_changes = status_changes_full.dropna(subset=['timestamp_sat']).copy()
status_changes['time_to_completion'] = (status_changes['timestamp_sat'] - status_changes['timestamp']).dt.total_seconds() / 60

# Analyze average completion times by room and requirement
status_change_times = status_changes.groupby(['room', 'requirement'])['time_to_completion'].mean().reset_index()
status_change_times = status_change_times.sort_values('time_to_completion', ascending=False)
print("\nAverage Time for Status Changes in Pattern Rooms (Required → Satisfied) in Minutes:")
print(status_change_times.head(10))

# Plot top 10 slowest status changes
plt.figure(figsize=(12, 6))
top_status_changes = status_change_times.head(10)
top_status_changes['label'] = top_status_changes['room'] + ' (' + top_status_changes['requirement'] + ')'
top_status_changes['label'] = top_status_changes['label'].apply(reverse_hebrew)
sns.barplot(data=top_status_changes, x='time_to_completion', y='label')
plt.title('Top 10 Slowest Status Changes in Pattern Rooms (Required → Satisfied)')
plt.xlabel('Average Time (Minutes)')
plt.ylabel('Room (Requirement)')
plt.tight_layout()
plt.savefig('slowest_status_changes_pattern.png')
plt.show()

# Step 20: Analyze Requirements in Pattern Rooms
plt.figure(figsize=(10, 6))
df_pattern['requirement'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Most Frequent Requirements in Pattern Rooms')
plt.xlabel('Requirement')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.xticks(ticks=range(10), labels=[reverse_hebrew(label.get_text()) for label in plt.gca().get_xticklabels()])
plt.tight_layout()
plt.savefig('top_requirements_pattern.png')
plt.show()

req_times = status_changes.groupby('requirement')['time_to_completion'].mean().reset_index()
req_times = req_times.sort_values('time_to_completion', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=req_times.head(10), x='time_to_completion', y='requirement')
plt.title('Top 10 Slowest Requirements in Pattern Rooms (Required → Satisfied)')
plt.xlabel('Average Time (Minutes)')
plt.ylabel('Requirement')
plt.yticks(ticks=range(10), labels=[reverse_hebrew(label.get_text()) for label in plt.gca().get_yticklabels()])
plt.tight_layout()
plt.savefig('slowest_requirements_pattern.png')
plt.show()

# Step 21: Analyze Shifts and Distance in Pattern Rooms
shift_times = status_changes.groupby('shifts')['time_to_completion'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=shift_times, x='shifts', y='time_to_completion')
plt.title('Average Status Change Time by Shift in Pattern Rooms')
plt.xlabel('Shift (1: 06:00-13:59, 2: 14:00-21:59, 3: 22:00-05:59)')
plt.ylabel('Average Time (Minutes)')
plt.tight_layout()
plt.savefig('status_change_by_shift_pattern.png')
plt.show()

distance_times = status_changes.groupby('distance')['time_to_completion'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=distance_times, x='distance', y='time_to_completion')
plt.title('Average Status Change Time by Distance in Pattern Rooms')
plt.xlabel('Distance (T=1, A=2, B=3, C=4)')
plt.ylabel('Average Time (Minutes)')
plt.tight_layout()
plt.savefig('status_change_by_distance_pattern.png')
plt.show()

# Step 22: Dynamically Generate Departments and Plot Requirements by Room for Each Department
departments = {}
for room in one_letter_rooms:
    dept = room[0]
    if dept not in departments:
        departments[dept] = []
    departments[dept].append(room)
for dept in departments:
    departments[dept] = sorted(departments[dept])
print("\nDynamically Generated Departments:")
for dept, rooms in departments.items():
    print(f"Department {dept}: {rooms}")

for dept, rooms in departments.items():
    dept_df = df_pattern[(df_pattern['room'].isin(rooms)) & (df_pattern['status'] == 'required')]
    requirement_counts = dept_df.groupby(['room', 'requirement']).size().reset_index(name='count')
    pivot_table = requirement_counts.pivot(index='room', columns='requirement', values='count').fillna(0)
    plt.figure(figsize=(12, 8))
    pivot_table.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title(f'Requirements by Room in Department {dept} (Required Status Only)')
    plt.xlabel('Room')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45)
    handles, labels = plt.gca().get_legend_handles_labels()
    reversed_labels = [reverse_hebrew(label) for label in labels]
    plt.legend(handles, reversed_labels, title='Requirement', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'requirements_by_room_{dept}.png', bbox_inches='tight')
    plt.show()

# Step 23: Save the Updated Dataset (Pattern Rooms Only)
df_pattern.to_csv("ED_pattern_rooms_only.csv", index=False, encoding='utf-8')
print("\nDataset (pattern rooms only) saved as 'ED_pattern_rooms_only.csv'")

# Step 24: Monthly Analysis of Completed Assignments
status_changes['month_year'] = status_changes['timestamp'].dt.to_period('M')
monthly_completions = status_changes.groupby('month_year').size().reset_index(name='completions')
monthly_completions['month_start'] = monthly_completions['month_year'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
plt.plot(monthly_completions['month_start'], monthly_completions['completions'], marker='o')
plt.title('Number of Completed Assignments per Month')
plt.xlabel('Month')
plt.ylabel('Number of Completions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_completed_assignments.png')
plt.show()

print("\nMonthly Completions:")
print(monthly_completions)

# Step 25: Time to Completion by Requirement
req_stats = status_changes.groupby('requirement')['time_to_completion'].agg(['mean', 'median', 'count']).reset_index()
req_stats = req_stats.sort_values('median', ascending=False)

valid_requirements = req_stats[req_stats['count'] >= 5]['requirement']
filtered_status_changes = status_changes[status_changes['requirement'].isin(valid_requirements)]

filtered_status_changes['requirement'] = pd.Categorical(
    filtered_status_changes['requirement'],
    categories=req_stats[req_stats['requirement'].isin(valid_requirements)]['requirement'],
    ordered=True
)

plt.figure(figsize=(16, 8))
sns.violinplot(data=filtered_status_changes, x='requirement', y='time_to_completion', inner=None, cut=0)
medians = filtered_status_changes.groupby('requirement')['time_to_completion'].median()
for i, median in enumerate(medians):
    plt.plot(i, median, 'ro', markersize=8, label='Median' if i == 0 else "")
plt.title('Distribution of Time to Completion by Requirement (Min 5 Completions)')
plt.xlabel('Requirement')
plt.ylabel('Time to Completion (Minutes)')
y_max = filtered_status_changes['time_to_completion'].quantile(0.95)
plt.ylim(0, y_max)
unique_requirements = filtered_status_changes['requirement'].cat.categories
plt.xticks(ticks=range(len(unique_requirements)), labels=[reverse_hebrew(req) for req in unique_requirements], rotation=45, ha='right')
for i, req in enumerate(unique_requirements):
    count = len(filtered_status_changes[filtered_status_changes['requirement'] == req])
    plt.text(i, y_max * 0.9, f'n={count}', ha='center', va='bottom', fontsize=8, color='red')
plt.legend()
plt.tight_layout()
plt.savefig('time_to_completion_by_requirement.png')
plt.show()

print("\nAverage and Median Time to Completion by Requirement (Minutes):")
print(req_stats.head(10))

# Step 26: Completion Rates by Shift
shift_completions = status_changes.groupby('shifts').size().reset_index(name='completions')
plt.figure(figsize=(8, 6))
sns.barplot(data=shift_completions, x='shifts', y='completions')
plt.title('Number of Completed Assignments by Shift')
plt.xlabel('Shift (1: 06:00-13:59, 2: 14:00-21:59, 3: 22:00-05:59)')
plt.ylabel('Number of Completions')
plt.tight_layout()
plt.savefig('completed_assignments_by_shift.png')
plt.show()

shift_avg_time = status_changes.groupby('shifts')['time_to_completion'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=shift_avg_time, x='shifts', y='time_to_completion')
plt.title('Average Time to Completion by Shift')
plt.xlabel('Shift')
plt.ylabel('Average Time (Minutes)')
plt.tight_layout()
plt.savefig('avg_time_to_completion_by_shift.png')
plt.show()
print("\nCompletions by Shift:")
print(shift_completions)
print("\nAverage Time by Shift (Minutes):")
print(shift_avg_time)

# Step 27: Identify Uncompleted Assignments
uncompleted = status_changes_full[status_changes_full['timestamp_sat'].isna()]
uncompleted_counts = uncompleted['requirement'].value_counts().reset_index()
uncompleted_counts.columns = ['requirement', 'uncompleted_count']
plt.figure(figsize=(10, 6))
top_n = min(len(uncompleted_counts), 10)
top_uncompleted_data = uncompleted_counts.head(top_n)
sns.barplot(data=top_uncompleted_data, x='uncompleted_count', y='requirement')
plt.title('Top Requirements with Uncompleted Assignments')
plt.xlabel('Number of Uncompleted Assignments')
plt.ylabel('Requirement')
top_uncompleted = top_uncompleted_data['requirement']
plt.yticks(ticks=range(len(top_uncompleted)), labels=[reverse_hebrew(req) for req in top_uncompleted])
plt.tight_layout()
plt.savefig('uncompleted_assignments_by_requirement.png')
plt.show()
print("\nUncompleted Assignments by Requirement:")
print(uncompleted_counts.head(10))