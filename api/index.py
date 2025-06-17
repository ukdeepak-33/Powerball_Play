import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
import random
from itertools import combinations
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
import requests
import json
import numpy as np

# --- Supabase Configuration ---
SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NzMwNjUsImV4cCI6MjA2NTM0OTA2NX0.AzUD7wjR7VbvtUH27NDqJ3AlvFW0nCWpiN9ADG8T_t4")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "YOUR_SUPABASE_SERVICE_ROLE_KEY")

SUPABASE_TABLE_NAME = 'powerball_draws'
GENERATED_NUMBERS_TABLE_NAME = 'generated_powerball_numbers'

# --- Flask App Initialization with Template Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = 'supersecretkey'

# --- Global Data and Cache ---
df = pd.DataFrame()
last_draw = pd.Series(dtype='object') 

# Cache for precomputed analysis data
analysis_cache = {}
last_analysis_cache_update = datetime.min # Initialize with the earliest possible datetime

# Cache expiration time (e.g., 1 hour for analysis data)
CACHE_EXPIRATION_SECONDS = 3600

# --- Utility Functions ---

def _get_supabase_headers(is_service_key=False):
    key = SUPABASE_SERVICE_KEY if is_service_key else SUPABASE_ANON_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

def load_historical_data_from_supabase():
    all_data = []
    offset = 0
    limit = 1000

    try:
        url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        headers = _get_supabase_headers(is_service_key=False)
        
        while True:
            params = {
                'select': '*',
                'order': 'Draw Date.asc',
                'offset': offset,
                'limit': limit
            }
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            chunk = response.json()
            if not chunk:
                break
            all_data.extend(chunk)
            offset += limit

        if not all_data:
            print("No data fetched from Supabase after pagination attempts.")
            return pd.DataFrame()

        df_loaded = pd.DataFrame(all_data)
        df_loaded['Draw Date_dt'] = pd.to_datetime(df_loaded['Draw Date'], errors='coerce')
        df_loaded = df_loaded.dropna(subset=['Draw Date_dt'])

        numeric_cols = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']
        for col in numeric_cols:
            if col in df_loaded.columns:
                df_loaded[col] = pd.to_numeric(df_loaded[col], errors='coerce')
                df_loaded[col] = df_loaded[col].fillna(0).astype(int)
            else:
                print(f"Warning: Column '{col}' not found in fetched data. Skipping conversion for this column.")

        df_loaded['Draw Date'] = df_loaded['Draw Date_dt'].dt.strftime('%Y-%m-%d')
        
        print(f"Successfully loaded and processed {len(df_loaded)} records from Supabase.")
        return df_loaded

    except requests.exceptions.RequestException as e:
        print(f"Error during Supabase data fetch request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Supabase response content: {e.response.text}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Supabase: {e}")
        if 'response' in locals() and response is not None:
            print(f"Response content that failed JSON decode: {response.text}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred in load_historical_data_from_supabase: {e}")
        return pd.DataFrame()

def get_last_draw(df):
    if df.empty:
        return pd.Series({
            'Draw Date': 'N/A', 'Number 1': 'N/A', 'Number 2': 'N/A',
            'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A'
        }, dtype='object')
    return df.iloc[-1]

def check_exact_match(df, white_balls):
    if df.empty: return False
    for _, row in df.iterrows():
        historical_white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        if set(white_balls) == set(historical_white_balls):
            return True
    return False

def generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None):
    if df.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 1000
    attempts = 0
    while attempts < max_attempts:
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        
        if len(available_numbers) < 5:
            raise ValueError("Not enough available numbers for white balls after exclusions and range constraints.")
            
        white_balls = sorted(random.sample(available_numbers, 5))

        group_a_numbers = [num for num in white_balls if num in group_a]
        if len(group_a_numbers) < 2:
            attempts += 1
            continue

        powerball = random.randint(powerball_range[0], powerball_range[1])

        last_draw_data = get_last_draw(df)
        if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
            last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
            if set(white_balls) == set(last_white_balls) and powerball == int(last_draw_data['Powerball']):
                attempts += 1
                continue

        if check_exact_match(df, white_balls):
            attempts += 1
            continue

        even_count = sum(1 for num in white_balls if num % 2 == 0)
        odd_count = 5 - even_count

        if odd_even_choice == "All Even" and even_count != 5:
            attempts += 1
            continue
        elif odd_even_choice == "All Odd" and odd_count != 5:
            continue
        elif odd_even_choice == "3 Even / 2 Odd" and (even_count != 3 or odd_count != 2):
            attempts += 1
            continue
        elif odd_even_choice == "3 Odd / 2 Even" and (odd_count != 3 or even_count != 2):
            attempts += 1
            continue
        elif odd_even_choice == "1 Even / 4 Odd" and (even_count != 1 or odd_count != 4):
            attempts += 1
            continue
        elif odd_even_choice == "1 Odd / 4 Even" and (odd_count != 1 or even_count != 4):
            attempts += 1
            continue

        if high_low_balance is not None:
            low_numbers_count = sum(1 for num in white_balls if num <= 34)
            high_numbers_count = sum(1 for num in white_balls if num >= 35)
            if low_numbers_count != high_low_balance[0] or high_numbers_count != high_low_balance[1]:
                attempts += 1
                continue
        
        break
    else:
        raise ValueError("Could not generate a unique combination meeting all criteria after many attempts.")

    return white_balls, powerball

def check_historical_match(df, white_balls, powerball):
    if df.empty: return None
    for _, row in df.iterrows():
        historical_white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        historical_powerball = int(row['Powerball'])
        if set(white_balls) == set(historical_white_balls) and powerball == historical_powerball:
            return row['Draw Date']
    return None

def frequency_analysis(df_source): # Renamed df to df_source to avoid conflict with global df
    if df_source.empty: return pd.Series([], dtype=int), pd.Series([], dtype=int)
    white_balls = df_source[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)
    powerball_freq = df_source['Powerball'].astype(int).value_counts().reindex(range(1, 27), fill_value=0)
    return white_ball_freq, powerball_freq

def hot_cold_numbers(df_source, last_draw_date_str): # Renamed df to df_source
    if df_source.empty or last_draw_date_str == 'N/A': return pd.Series([], dtype=int), pd.Series([], dtype=int)
    last_draw_date = pd.to_datetime(last_draw_date_str)
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    
    recent_data = df_source[df_source['Draw Date_dt'] >= one_year_ago].copy()
    if recent_data.empty: return pd.Series([], dtype=int), pd.Series([], dtype=int)

    white_balls = recent_data[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts()

    hot_numbers = white_ball_freq.nlargest(14).sort_values(ascending=False)
    cold_numbers = white_ball_freq.nsmallest(14).sort_values(ascending=True)

    if hot_numbers.empty:
        hot_numbers = pd.Series([], dtype=int)
    if cold_numbers.empty:
        cold_numbers = pd.Series([], dtype=int)

    return hot_numbers, cold_numbers

def monthly_white_ball_analysis(df_source, last_draw_date_str): # Renamed df to df_source
    print("[DEBUG-Monthly] Inside monthly_white_ball_analysis function.")
    if df_source.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-Monthly] df_source is empty or last_draw_date_str is N/A. Returning empty dict.")
        return {}

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-Monthly] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-Monthly] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty dict.")
        return {}

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-Monthly] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-Monthly] 'Draw Date_dt' column missing or not datetime type in df_source. Attempting to re-create it.")
        try:
            df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'], errors='coerce')
            df_source = df_source.dropna(subset=['Draw Date_dt'])
            if df_source.empty:
                print("[ERROR-Monthly] Re-creating 'Draw Date_dt' resulted in empty DataFrame. Returning empty dict.")
                return {}
            print("[DEBUG-Monthly] Successfully re-created 'Draw Date_dt' column.")
        except Exception as e_recreate:
            print(f"[ERROR-Monthly] Failed to re-create 'Draw Date_dt' column: {e_recreate}. Returning empty dict.")
            return {}


    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    print(f"[DEBUG-Monthly] recent_data shape after filtering: {recent_data.shape}")
    if recent_data.empty:
        print("[DEBUG-Monthly] recent_data is empty after filtering. Returning empty dict.")
        return {}

    monthly_balls = {}
    try:
        # Check if 'Month' column exists before creating
        if 'Month' not in recent_data.columns:
            recent_data['Month'] = recent_data['Draw Date_dt'].dt.to_period('M')
            print(f"[DEBUG-Monthly] 'Month' column added to recent_data. First 2 months: {recent_data['Month'].head(2).tolist()}")
        
        # Ensure all necessary numeric columns exist and are numeric
        required_cols = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
        for col in required_cols:
            if col not in recent_data.columns:
                print(f"[[ERROR-Monthly] Missing required column '{col}' in recent_data. Cannot perform analysis.")
                return {}
            # Ensure they are numeric, coerce errors to NaN
            recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')

        # Drop rows where any of the numeric ball columns are NaN, before flattening
        recent_data = recent_data.dropna(subset=required_cols)
        if recent_data.empty:
            print("[DEBUG-Monthly] recent_data is empty after dropping NaN in ball columns. Returning empty dict.")
            return {}

        # The core change: Ensure all numbers are converted to Python native int
        # after flattening and before being added to the list for the dictionary value.
        monthly_balls_raw = recent_data.groupby('Month')[required_cols].apply(
            lambda x: sorted([int(num) for num in x.values.flatten() if not pd.isna(num)])
        ).to_dict()

        # Convert Period keys to string keys, and ensure values are lists of native ints
        monthly_balls_str_keys = {}
        for period_key, ball_list in monthly_balls_raw.items():
            monthly_balls_str_keys[str(period_key)] = [int(ball) for ball in ball_list] # Explicitly convert to Python int again
        
        print(f"[DEBUG-Monthly] Groupby and apply successful. First item in monthly_balls_str_keys: {next(iter(monthly_balls_str_keys.items())) if monthly_balls_str_keys else 'N/A'}")

    except Exception as e:
        print(f"[ERROR-Monthly] Error during groupby/apply operation or conversion: {e}. Returning empty dict.")
        import traceback
        traceback.print_exc() # Print full traceback to logs for detailed error
        return {}
    
    print("[DEBUG-Monthly] Successfully computed monthly_balls_str_keys.")
    return monthly_balls_str_keys


def sum_of_main_balls(df_source): # Renamed df to df_source
    """Calculates the sum of the five main white balls for each draw."""
    if df_source.empty:
        return pd.DataFrame(), [], 0, 0, 0.0
    
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)
    
    temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    sum_freq = temp_df['Sum'].value_counts().sort_index()
    sum_freq_list = [{'sum': int(s), 'count': int(c)} for s, c in sum_freq.items()]

    min_sum = int(temp_df['Sum'].min()) if not temp_df['Sum'].empty else 0
    max_sum = int(temp_df['Sum'].max()) if not temp_df['Sum'].empty else 0
    avg_sum = round(temp_df['Sum'].mean(), 2) if not temp_df['Sum'].empty else 0.0

    return temp_df[['Draw Date', 'Sum']], sum_freq_list, min_sum, max_sum, avg_sum

def find_results_by_sum(df_source, target_sum): # Renamed df to df_source
    if df_source.empty: return pd.DataFrame()
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)

    if 'Sum' not in temp_df.columns:
        temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    results = temp_df[temp_df['Sum'] == target_sum]
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Sum']]

def simulate_multiple_draws(df_source, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100): # Renamed df to df_source
    if df_source.empty: return pd.Series([], dtype=int)
    results = []
    for _ in range(num_draws):
        try:
            white_balls, powerball = generate_powerball_numbers(df_source, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
            results.append(white_balls + [powerball])
        except ValueError:
            pass
    
    if not results: return pd.Series([], dtype=int)
    all_numbers = [num for draw in results for num in draw]
    freq = pd.Series(all_numbers).value_counts().sort_index()
    return freq

def calculate_combinations(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

def winning_probability(white_ball_range, powerball_range):
    total_white_balls_in_range = white_ball_range[1] - white_ball_range[0] + 1
    white_ball_combinations = calculate_combinations(total_white_balls_in_range, 5)

    total_powerballs_in_range = powerball_range[1] - powerball_range[0] + 1

    total_combinations = white_ball_combinations * total_powerballs_in_range

    probability_1_in_x = f"1 in {total_combinations:,}" if total_combinations > 0 else "N/A"
    probability_percentage = f"{1 / total_combinations * 100:.10f}%" if total_combinations > 0 else "N/A"

    return probability_1_in_x, probability_percentage

def partial_match_probabilities(white_ball_range, powerball_range):
    total_white_balls_in_range = white_ball_range[1] - white_ball_range[0] + 1
    total_powerballs_in_range = powerball_range[1] - powerball_range[0] + 1

    # Total possible combinations for 5 white balls from the range
    total_white_ball_combinations_possible = calculate_combinations(total_white_balls_in_range, 5)
    
    probabilities = {}

    prizes = {
        "Match 5 White Balls + Powerball": {"matched_w": 5, "unmatched_w": 0, "matched_p": 1},
        "Match 5 White Balls Only": {"matched_w": 5, "unmatched_w": 0, "matched_p": 0},
        "Match 4 White Balls + Powerball": {"matched_w": 4, "unmatched_w": 1, "matched_p": 1},
        "Match 4 White Balls Only": {"matched_w": 4, "unmatched_w": 1, "matched_p": 0},
        "Match 3 White Balls + Powerball": {"matched_w": 3, "unmatched_w": 2, "matched_p": 1},
        "Match 3 White Balls Only": {"matched_w": 3, "unmatched_w": 2, "matched_p": 0},
        "Match 2 White Balls + Powerball": {"matched_w": 2, "unmatched_w": 3, "matched_p": 1},
        "Match 1 White Ball + Powerball": {"matched_w": 1, "unmatched_w": 4, "matched_p": 1},
        "Match Powerball Only": {"matched_w": 0, "unmatched_w": 5, "matched_p": 1},
    }

    for scenario, data in prizes.items():
        # Combinations of matching white balls
        comb_matched_w = calculate_combinations(5, data["matched_w"])
        # Combinations of UNMATCHING white balls from the remaining pool
        comb_unmatched_w = calculate_combinations(total_white_balls_in_range - 5, data["unmatched_w"])

        # Combinations for Powerball
        if data["matched_p"] == 1:
            comb_p = 1 # Must match the one Powerball
        else:
            comb_p = total_powerballs_in_range - 1 # Must NOT match the Powerball (pick from remaining)
            if comb_p < 0: # Handle cases where total_powerballs_in_range is 1 and we need to "not match"
                comb_p = 0
        
        numerator = comb_matched_w * comb_unmatched_w * comb_p
        
        if numerator == 0:
            probabilities[scenario] = "N/A"
        else:
            # Total possible unique combinations (5 white balls * 1 powerball)
            total_possible_combinations_for_draw = calculate_combinations(total_white_balls_in_range, 5) * total_powerballs_in_range
            
            # Probability is the inverse: total ways to draw / ways to win this specific scenario
            probability = total_possible_combinations_for_draw / numerator
            probabilities[scenario] = f"{probability:,.0f} to 1"

    return probabilities


def export_analysis_results(df_source, file_path="analysis_results.csv"): # Renamed df to df_source
    df_source.to_csv(file_path, index=False)
    print(f"Analysis results saved to {file_path}")

def find_last_draw_dates_for_numbers(df_source, white_balls, powerball): # Renamed df to df_source
    if df_source.empty: return {}
    last_draw_dates = {}
    
    sorted_df = df_source.sort_values(by='Draw Date_dt', ascending=False)

    for number in white_balls:
        found = False
        for _, row in sorted_df.iterrows():
            historical_white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
            if number in historical_white_balls:
                last_draw_dates[f"White Ball {number}"] = row['Draw Date']
                found = True
                break
        if not found:
            last_draw_dates[f"White Ball {number}"] = "N/A (Never Drawn)"

    found_pb = False
    for _, row in sorted_df.iterrows():
        if powerball == int(row['Powerball']):
            last_draw_dates[f"Powerball {powerball}"] = row['Draw Date']
            found_pb = True
            break
    if not found_pb:
        last_draw_dates[f"Powerball {powerball}"] = "N/A (Never Drawn)"

    return last_draw_dates

def modify_combination(df_source, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers): # Renamed df to df_source
    if df_source.empty:
        raise ValueError("Cannot modify combination: Historical data is empty.")

    white_balls = list(white_balls)
    
    if len(white_balls) < 5:
        raise ValueError("Initial white balls list is too short for modification.")

    indices_to_modify = random.sample(range(5), 3)
    
    for i in indices_to_modify:
        attempts = 0
        max_attempts_single_num = 100
        while attempts < max_attempts_single_num:
            new_number = random.randint(white_ball_range[0], white_ball_range[1])
            if new_number not in excluded_numbers and new_number not in white_balls:
                white_balls[i] = new_number
                break
            attempts += 1
        else:
            print(f"Warning: Could not find unique replacement for white ball at index {i}. Proceeding without replacement for this slot.")

    attempts_pb = 0
    max_attempts_pb = 100
    while attempts_pb < max_attempts_pb:
        new_powerball = random.randint(powerball_range[0], powerball_range[1])
        if new_powerball not in excluded_numbers and new_powerball != powerball:
            powerball = new_powerball
            break
        attempts_pb += 1
    else:
        print("Warning: Could not find a unique replacement for powerball. Keeping original.")

    white_balls = sorted([int(num) for num in white_balls])
    powerball = int(powerball)
    
    return white_balls, powerball

def find_common_pairs(df_source, top_n=10): # Renamed df to df_source
    if df_source.empty: return []
    pair_count = defaultdict(int)
    for _, row in df_source.iterrows():
        nums = sorted([int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])])
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair = tuple(sorted((nums[i], nums[j])))
                pair_count[pair] += 1
    
    sorted_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in sorted_pairs[:top_n]]

def filter_common_pairs_by_range(common_pairs, num_range):
    filtered_pairs = []
    if not num_range or len(num_range) != 2:
        return common_pairs
        
    min_val, max_val = num_range
    for pair in common_pairs:
        if min_val <= pair[0] <= max_val and min_val <= pair[1] <= max_val:
            filtered_pairs.append(pair)
    return filtered_pairs

def generate_with_common_pairs(df_source, common_pairs, white_ball_range, excluded_numbers): # Renamed df to df_source
    if df_source.empty:
        raise ValueError("Cannot generate numbers with common pairs: Historical data is empty.")

    if not common_pairs:
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        if len(available_numbers) < 5:
             raise ValueError("Not enough numbers to generate 5 white balls after exclusions.")
        return sorted(random.sample(available_numbers, 5))

    num1, num2 = random.choice(common_pairs)
    
    available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) 
                         if num not in excluded_numbers and num not in [num1, num2]]
    
    if len(available_numbers) < 3:
        available_numbers_fallback = [n for n in range(white_ball_range[0], white_ball_range[1] + 1) if n not in excluded_numbers]
        if len(available_numbers_fallback) < 5:
            raise ValueError("Not enough numbers to generate 5 white balls even with fallback after exclusions.")
        return sorted(random.sample(available_numbers_fallback, 5))

    remaining_numbers = random.sample(available_numbers, 3)
    white_balls = sorted([num1, num2] + remaining_numbers)
    return white_balls

def get_number_age_distribution(df_source): # Renamed df to df_source
    if df_source.empty: return [], [] # Return both lists
    df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'])
    all_draw_dates = sorted(df_source['Draw Date_dt'].drop_duplicates().tolist())
    
    detailed_ages = []
    
    # Process White Balls
    for i in range(1, 70):
        last_appearance_date = None
        temp_df_filtered = df_source[(df_source['Number 1'].astype(int) == i) | (df_source['Number 2'].astype(int) == i) |
                              (df_source['Number 3'].astype(int) == i) | (df_source['Number 4'].astype(int) == i) |
                              (df_source['Number 5'].astype(int) == i)]
        
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()

        miss_streak_count = 0
        if last_appearance_date is not None:
            # Count draws AFTER the last appearance date
            for d_date in reversed(all_draw_dates):
                if d_date > last_appearance_date:
                    miss_streak_count += 1
                else:
                    break # Stop when we reach or pass the last appearance date
            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': miss_streak_count})
        else:
            # If never drawn, age is total number of draws
            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': len(all_draw_dates)})

    # Process Powerballs
    for i in range(1, 27):
        last_appearance_date = None
        temp_df_filtered = df_source[df_source['Powerball'].astype(int) == i]
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()

        miss_streak_count = 0
        if last_appearance_date is not None:
            # Count draws AFTER the last appearance date
            for d_date in reversed(all_draw_dates):
                if d_date > last_appearance_date:
                    miss_streak_count += 1
                else:
                    break # Stop when we reach or pass the last appearance date
            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': miss_streak_count})
        else:
            # If never drawn, age is total number of draws
            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': len(all_draw_dates)})

    # Aggregate for the bar chart (same as before)
    all_miss_streaks_only = [item['age'] for item in detailed_ages]
    age_counts = pd.Series(all_miss_streaks_only).value_counts().sort_index()
    age_counts_list = [{'age': int(age), 'count': int(count)} for age, count in age_counts.items()]
    
    return age_counts_list, detailed_ages # Return both

def get_co_occurrence_matrix(df_source): # Renamed df to df_source
    if df_source.empty: return [], 0
    co_occurrence = defaultdict(int)
    
    for index, row in df_source.iterrows():
        white_balls = sorted([int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])])
        for i in range(len(white_balls)):
            for j in range(i + 1, len(white_balls)):
                pair = tuple(sorted((white_balls[i], white_balls[j])))
                co_occurrence[pair] += 1
    
    co_occurrence_data = []
    for pair, count in co_occurrence.items():
        co_occurrence_data.append({'x': int(pair[0]), 'y': int(pair[1]), 'count': int(count)})
    
    max_co_occurrence = max(item['count'] for item in co_occurrence_data) if co_occurrence_data else 0
    
    return co_occurrence_data, max_co_occurrence

def get_powerball_position_frequency(df_source): # Renamed df to df_source
    if df_source.empty: return []
    position_frequency_data = []
    
    for index, row in df_source.iterrows():
        powerball = int(row['Powerball'])
        for i in range(1, 6):
            col_name = f'Number {i}'
            if col_name in row and pd.notna(row[col_name]):
                position_frequency_data.append({
                    'powerball_number': powerball,
                    'white_ball_value_at_position': int(row[col_name]),
                    'white_ball_position': i
                })
    return position_frequency_data

# Function to check for consecutive numbers in a draw
def _has_consecutive(numbers_list):
    """Checks if a list of numbers contains any consecutive sequence."""
    if len(numbers_list) < 2:
        return 0 # No consecutive numbers possible
    
    sorted_nums = sorted(numbers_list)
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i] + 1 == sorted_nums[i+1]:
            return 1 # Found at least one consecutive pair
    return 0

# Get consecutive numbers trends
def get_consecutive_numbers_trends(df_source, last_draw_date_str): # Renamed df to df_source
    print("[DEBUG-ConsecutiveTrends] Inside get_consecutive_numbers_trends function.")
    if df_source.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-ConsecutiveTrends] df_source is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-ConsecutiveTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-ConsecutiveTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return []

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-ConsecutiveTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-ConsecutiveTrends] 'Draw Date_dt' column missing or not datetime type in df_source. Returning empty list.")
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt') # Ensure chronological order
    if recent_data.empty:
        print("[DEBUG-ConsecutiveTrends] recent_data is empty after filtering. Returning empty list.")
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        has_consecutive = _has_consecutive(white_balls)
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            'consecutive_present': has_consecutive # 1 if consecutive, 0 otherwise
        })
    
    print(f"[DEBUG-ConsecutiveTrends] Generated {len(trend_data)} trend data points.")
    return trend_data

# Get most frequent triplets
def get_most_frequent_triplets(df_source, top_n=10): # Renamed df to df_source
    print("[DEBUG-Triplets] Inside get_most_frequent_triplets function.")
    if df_source.empty:
        print("[DEBUG-Triplets] df_source is empty. Returning empty list.")
        return []

    triplet_counts = defaultdict(int)

    # Iterate through each draw
    for idx, row in df_source.iterrows():
        # Get the 5 white balls for the current draw
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        # Generate all unique combinations of 3 numbers (triplets) from the 5 white balls
        for triplet_combo in combinations(sorted(white_balls), 3):
            triplet_counts[triplet_combo] += 1
    
    # Sort triplets by their frequency in descending order
    sorted_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Format for JSON serialization and return top_n
    formatted_triplets = []
    for triplet, count in sorted_triplets[:top_n]:
        formatted_triplets.append({
            'triplet': list(triplet), # Convert tuple to list for JSON
            'count': int(count)
        })
    
    print(f"[DEBUG-Triplets] Found {len(triplet_counts)} unique triplets. Returning top {len(formatted_triplets)}.")
    return formatted_triplets


def get_odd_even_split_trends(df_source, last_draw_date_str): # Renamed df to df_source
    print("[DEBUG-OddEvenTrends] Inside get_odd_even_split_trends function.")
    if df_source.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-OddEvenTrends] df_source is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-OddEvenTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-OddEvenTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return []

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-OddEvenTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-OddEvenTrends] 'Draw Date_dt' column missing or not datetime type in df_source. Returning empty list.")
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt') # Ensure chronological order
    if recent_data.empty:
        print("[DEBUG-OddEvenTrends] recent_data is empty after filtering. Returning empty list.")
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        even_count = sum(1 for num in white_balls if num % 2 == 0)
        odd_count = 5 - even_count

        split_category = "Other" # Default for categories not explicitly listed

        if odd_count == 5:
            split_category = "All Odd"
        elif even_count == 5:
            split_category = "All Even"
        elif odd_count == 4 and even_count == 1:
            split_category = "4 Odd / 1 Even"
        elif odd_count == 1 and even_count == 4:
            split_category = "1 Odd / 4 Even"
        elif odd_count == 3 and even_count == 2:
            split_category = "3 Odd / 2 Even"
        elif odd_count == 2 and even_count == 3:
            split_category = "2 Odd / 3 Even"
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            'split_category': split_category
        })
    
    # Create a DataFrame from the raw trend_data
    trend_df = pd.DataFrame(trend_data)
    
    if trend_df.empty:
        print("[DEBUG-OddEvenTrends] trend_df is empty after categorizing. Returning empty list.")
        return []

    # Get all unique draw dates to ensure a continuous x-axis for plotting
    all_dates = pd.to_datetime(trend_df['draw_date']).dt.date.unique()
    all_dates.sort()

    # Define all expected categories for consistent columns
    categories = ["All Odd", "All Even", "4 Odd / 1 Even", "1 Odd / 4 Even", "3 Odd / 2 Even", "2 Odd / 3 Even", "Other"]

    # Create a list of dictionaries for the chart
    chart_data = []
    for date_obj in all_dates:
        date_str = date_obj.strftime('%Y-%m-%d')
        daily_counts = trend_df[trend_df['draw_date'] == date_str]['split_category'].value_counts().to_dict()
        
        row_data = {'draw_date': date_str}
        for cat in categories:
            row_data[cat] = int(daily_counts.get(cat, 0)) # Ensure it's int
        chart_data.append(row_data)

    print(f"[DEBUG-OddEvenTrends] Generated {len(chart_data)} trend data points.")
    return chart_data


# --- NEW UTILITY FUNCTIONS for Generated Numbers and Manual Draws ---

def save_manual_draw_to_db(draw_date, n1, n2, n3, n4, n5, pb):
    """
    Saves a manually entered official Powerball draw to the 'powerball_draws' table.
    Checks for existence of the draw date to prevent duplicates.
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=True) # Use service key for writes

    # First, check if the draw date already exists
    check_params = {'select': 'Draw Date', 'Draw Date': f'eq.{draw_date}'}
    check_response = requests.get(url, headers=headers, params=check_params)
    check_response.raise_for_status()
    existing_draws = check_response.json()

    if existing_draws:
        print(f"Draw for date {draw_date} already exists in {SUPABASE_TABLE_NAME}.")
        return False, f"Draw for {draw_date} already exists."

    new_draw_data = {
        'Draw Date': draw_date,
        'Number 1': n1,
        'Number 2': n2,
        'Number 3': n3,
        'Number 4': n4,
        'Number 5': n5,
        'Powerball': pb
    }

    insert_response = requests.post(url, headers=headers, data=json.dumps(new_draw_data))
    insert_response.raise_for_status()

    if insert_response.status_code == 201:
        print(f"Successfully inserted manual draw: {new_draw_data}")
        return True, "Official draw saved successfully!"
    else:
        print(f"Failed to insert manual draw. Status: {insert_response.status_code}, Response: {insert_response.text}")
        return False, f"Error saving official draw: {insert_response.status_code} - {insert_response.text}"

def save_generated_numbers_to_db(numbers, powerball):
    """
    Saves a generated Powerball combination to the 'generated_powerball_numbers' table.
    Ensures the combination is unique before saving.
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{GENERATED_NUMBERS_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=True) # Use service key for writes

    # Ensure numbers are sorted for consistent comparison
    sorted_numbers = sorted(numbers)

    # Check for existing combination
    check_params = {
        'select': 'id', # Only need the ID to confirm existence
        'number_1': f'eq.{sorted_numbers[0]}',
        'number_2': f'eq.{sorted_numbers[1]}',
        'number_3': f'eq.{sorted_numbers[2]}',
        'number_4': f'eq.{sorted_numbers[3]}',
        'number_5': f'eq.{sorted_numbers[4]}',
        'powerball': f'eq.{powerball}'
    }
    check_response = requests.get(url, headers=headers, params=check_params)
    check_response.raise_for_status()
    existing_combinations = check_response.json()

    if existing_combinations:
        print(f"Combination {sorted_numbers} + {powerball} already exists in {GENERATED_NUMBERS_TABLE_NAME}.")
        return False, f"This exact combination ({', '.join(map(str, sorted_numbers))} + {powerball}) has already been saved."

    # If not exists, proceed with insert
    new_generated_data = {
        'number_1': sorted_numbers[0],
        'number_2': sorted_numbers[1],
        'number_3': sorted_numbers[2],
        'number_4': sorted_numbers[3],
        'number_5': sorted_numbers[4],
        'powerball': powerball,
        'generated_date': datetime.now().isoformat() # Use ISO format for timestamp
    }

    insert_response = requests.post(url, headers=headers, data=json.dumps(new_generated_data))
    insert_response.raise_for_status()

    if insert_response.status_code == 201:
        print(f"Successfully inserted generated numbers: {new_generated_data}")
        return True, "Generated numbers saved successfully!"
    else:
        print(f"Failed to insert generated numbers. Status: {insert_response.status_code}, Response: {insert_response.text}")
        return False, f"Error saving generated numbers: {insert_response.status_code} - {insert_response.text}"

def get_generated_numbers_history():
    """
    Fetches all generated Powerball numbers and groups them by date, sorted by date descending.
    """
    all_data = []
    offset = 0
    limit = 1000

    try:
        url = f"{SUPABASE_PROJECT_URL}/rest/v1/{GENERATED_NUMBERS_TABLE_NAME}"
        headers = _get_supabase_headers(is_service_key=False) # Read-only, so anon key is fine
        
        while True:
            params = {
                'select': 'generated_date,number_1,number_2,number_3,number_4,number_5,powerball',
                'order': 'generated_date.desc', # Order by date descending for latest first
                'offset': offset,
                'limit': limit
            }
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            chunk = response.json()
            if not chunk:
                break
            all_data.extend(chunk)
            offset += limit

        if not all_data:
            print("No generated numbers fetched from Supabase.")
            return {}

        # Group by date for expand/collapse
        grouped_data = defaultdict(list)
        for record in all_data:
            # Parse the full timestamp, then format just the date part for grouping
            gen_dt = datetime.fromisoformat(record['generated_date'].replace('Z', '+00:00')) # Handle 'Z' for UTC
            date_key = gen_dt.strftime('%Y-%m-%d')
            
            # Format time for display within the group
            formatted_time = gen_dt.strftime('%I:%M %p') # e.g., "03:30 PM"

            generated_balls = sorted([
                int(record['number_1']), int(record['number_2']), int(record['number_3']),
                int(record['number_4']), int(record['number_5'])
            ])
            
            grouped_data[date_key].append({
                'time': formatted_time,
                'white_balls': generated_balls,
                'powerball': int(record['powerball'])
            })
        
        # Convert defaultdict to a regular dict, sorting dates for consistent display (already sorted by keys if iterating)
        # However, to explicitly ensure order for template, we can sort items by date_key
        sorted_grouped_data = dict(sorted(grouped_data.items(), key=lambda item: item[0], reverse=True))

        return sorted_grouped_data

    except requests.exceptions.RequestException as e:
        print(f"Error during Supabase data fetch request for generated numbers: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Supabase response content: {e.response.text}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for generated numbers from Supabase: {e}")
        if 'response' in locals() and response is not None:
            print(f"Response content that failed JSON decode: {response.text}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred in get_generated_numbers_history: {e}")
        import traceback
        traceback.print_exc()
        return {}


def check_generated_against_history(generated_white_balls, generated_powerball, df_historical):
    """
    Checks a generated Powerball number against historical official draws from the last two years
    and returns match counts and details.
    """
    results = {
        "generated_balls": generated_white_balls,
        "generated_powerball": generated_powerball,
        "summary": {
            "Match 5 White Balls + Powerball": {"count": 0, "draws": []},
            "Match 5 White Balls Only": {"count": 0, "draws": []},
            "Match 4 White Balls + Powerball": {"count": 0, "draws": []},
            "Match 4 White Balls Only": {"count": 0, "draws": []},
            "Match 3 White Balls + Powerball": {"count": 0, "draws": []},
            "Match 3 White Balls Only": {"count": 0, "draws": []},
            "Match 2 White Balls + Powerball": {"count": 0, "draws": []},
            "Match 1 White Ball + Powerball": {"count": 0, "draws": []},
            "Match Powerball Only": {"count": 0, "draws": []},
            "No Match": {"count": 0, "draws": []}
        }
    }

    if df_historical.empty:
        return results

    # Filter for last two years
    two_years_ago = datetime.now() - timedelta(days=2 * 365) # Approximate two years
    recent_historical_data = df_historical[df_historical['Draw Date_dt'] >= two_years_ago].copy()

    if recent_historical_data.empty:
        return results

    # Ensure generated_white_balls is a set for efficient matching
    gen_white_set = set(generated_white_balls)

    for index, row in recent_historical_data.iterrows():
        historical_white_balls = sorted([
            int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
            int(row['Number 4']), int(row['Number 5'])
        ])
        historical_powerball = int(row['Powerball'])
        historical_draw_date = row['Draw Date']

        hist_white_set = set(historical_white_balls)

        # Count matching white balls
        white_matches = len(gen_white_set.intersection(hist_white_set))

        # Check Powerball match
        powerball_match = 1 if generated_powerball == historical_powerball else 0

        # Determine the match category
        category = "No Match"
        if white_matches == 5 and powerball_match == 1:
            category = "Match 5 White Balls + Powerball"
        elif white_matches == 5 and powerball_match == 0:
            category = "Match 5 White Balls Only"
        elif white_matches == 4 and powerball_match == 1:
            category = "Match 4 White Balls + Powerball"
        elif white_matches == 4 and powerball_match == 0:
            category = "Match 4 White Balls Only"
        elif white_matches == 3 and powerball_match == 1:
            category = "Match 3 White Balls + Powerball"
        elif white_matches == 3 and powerball_match == 0:
            category = "Match 3 White Balls Only"
        elif white_matches == 2 and powerball_match == 1:
            category = "Match 2 White Balls + Powerball"
        elif white_matches == 1 and powerball_match == 1:
            category = "Match 1 White Ball + Powerball"
        elif white_matches == 0 and powerball_match == 1:
            category = "Match Powerball Only"
        # If no other match, it remains "No Match"

        results["summary"][category]["count"] += 1
        results["summary"][category]["draws"].append({
            "date": historical_draw_date,
            "white_balls": historical_white_balls,
            "powerball": historical_powerball
        })
    
    # Sort the draws within each category by date (most recent first)
    for category in results["summary"]:
        results["summary"][category]["draws"].sort(key=lambda x: x['date'], reverse=True)

    return results


# --- Data Initialization (Only df and last_draw on startup) ---
def initialize_core_data():
    global df, last_draw
    print("Attempting to load core historical data...")
    try:
        df_temp = load_historical_data_from_supabase()
        if not df_temp.empty:
            df = df_temp
            last_draw = get_last_draw(df)
            print("Core historical data loaded successfully.")
        else:
            print("Core historical data is empty after loading. df remains empty.")
    except Exception as e:
        print(f"An error occurred during initial core data loading: {e}")
        import traceback
        traceback.print_exc()

initialize_core_data() # Call this once on module load


# --- Analysis Cache Management ---
def get_cached_analysis(key, compute_function, *args, **kwargs):
    global analysis_cache, last_analysis_cache_update
    
    # Check if cache is still fresh and data exists
    if key in analysis_cache and (datetime.now() - last_analysis_cache_update).total_seconds() < CACHE_EXPIRATION_SECONDS:
        print(f"Serving '{key}' from cache.")
        return analysis_cache[key]
    
    print(f"Computing and caching '{key}'.")
    # Compute the data
    computed_data = compute_function(*args, **kwargs)
    
    # Store in cache
    analysis_cache[key] = computed_data
    last_analysis_cache_update = datetime.now() # Update timestamp whenever anything is computed/cached
    return computed_data

def invalidate_analysis_cache():
    global analysis_cache, last_analysis_cache_update
    analysis_cache = {} # Clear the cache
    last_analysis_cache_update = datetime.min # Reset timestamp
    print("Analysis cache invalidated.")

# Group A numbers (constants)
group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]
white_ball_range = (1, 69)
powerball_range = (1, 26)
excluded_numbers = [] # Global excluded numbers, can be extended by user input

# --- Flask Routes ---

@app.route('/')
def index():
    last_draw_dict = last_draw.to_dict()
    if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
        try:
            last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
        except ValueError:
            pass
    return render_template('index.html', last_draw=last_draw_dict)

@app.route('/generate', methods=['POST'])
def generate():
    if df.empty:
        flash("Cannot generate numbers: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        last_draw_dict = last_draw.to_dict() # Ensure last_draw is passed even on error
        if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
            try:
                last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
            except ValueError:
                pass
        return render_template('index.html', last_draw=last_draw_dict)

    odd_even_choice = request.form.get('odd_even_choice', 'Any')
    combo_choice = request.form.get('combo_choice', 'No Combo')
    white_ball_min = int(request.form.get('white_ball_min', 1))
    white_ball_max = int(request.form.get('white_ball_max', 69))
    white_ball_range_local = (white_ball_min, white_ball_max)
    powerball_min = int(request.form.get('powerball_min', 1))
    powerball_max = int(request.form.get('powerball_max', 26))
    powerball_range_local = (powerball_min, powerball_max)
    excluded_numbers_local = [int(num.strip()) for num in request.form.get('excluded_numbers', '').split(",") if num.strip().isdigit()] if request.form.get('excluded_numbers') else []
    
    high_low_balance_str = request.form.get('high_low_balance', '')
    high_low_balance = None
    if high_low_balance_str:
        try:
            parts = [int(num.strip()) for num in high_low_balance_str.split() if num.strip().isdigit()]
            if len(parts) == 2:
                high_low_balance = tuple(parts)
            else:
                flash("High/Low Balance input must be two numbers separated by space (e.g., '2 3').", 'error')
        except ValueError:
            flash("Invalid High/Low Balance format. Please enter numbers separated by space.", 'error')

    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        white_balls, powerball = generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range_local, powerball_range_local, excluded_numbers_local, high_low_balance)
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)
    except ValueError as e:
        flash(str(e), 'error')
        last_draw_dict = last_draw.to_dict()
        if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
            try:
                last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
            except ValueError:
                pass
        return render_template('index.html', last_draw=last_draw_dict)

    last_draw_dict = last_draw.to_dict()
    if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
        try:
            last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
        except ValueError:
            pass

    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw_dict, 
                           last_draw_dates=last_draw_dates)


@app.route('/generate_modified', methods=['POST'])
def generate_modified():
    if df.empty:
        flash("Cannot generate modified combination: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        last_draw_dict = last_draw.to_dict() # Ensure last_draw is passed even on error
        if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
            try:
                last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
            except ValueError:
                pass
        return render_template('index.html', last_draw=last_draw_dict)

    use_common_pairs = request.form.get('use_common_pairs') == 'on'
    num_range_str = request.form.get('num_range', '')
    num_range = None
    if num_range_str:
        try:
            parts = [int(num.strip()) for num in num_range_str.split() if num.strip().isdigit()]
            if len(parts) == 2:
                num_range = tuple(parts)
            else:
                flash("Filter Common Pairs by Range input must be two numbers separated by space (e.g., '1 20').", 'error')
        except ValueError:
            flash("Invalid Filter Common Pairs by Range format. Please enter numbers separated by space.", 'error')

    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        if not df.empty:
            random_row = df.sample(1).iloc[0]
            white_balls_base = [int(random_row['Number 1']), int(random_row['Number 2']), int(random_row['Number 3']), int(random_row['Number 4']), int(random_row['Number 5'])]
            powerball_base = int(random_row['Powerball'])
        else:
            flash("Historical data is empty, cannot generate or modify numbers.", 'error')
            last_draw_dict = last_draw.to_dict() # Ensure last_draw is passed even on error
            if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
                try:
                    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
                except ValueError:
                    pass
            return render_template('index.html', last_draw=last_draw_dict)


        if use_common_pairs:
            common_pairs = find_common_pairs(df, top_n=20)
            if num_range:
                common_pairs = filter_common_pairs_by_range(common_pairs, num_range)
            
            if not common_pairs:
                flash("No common pairs found with the specified filter. Generating a random combination instead.", 'info')
                white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
            else:
                white_balls = generate_with_common_pairs(df, common_pairs, white_ball_range, excluded_numbers)
                powerball = random.randint(powerball_range[0], powerball_range[1])
        else:
            white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, white_ball_range, powerball_range, excluded_numbers)
            
        max_attempts_unique = 100
        attempts_unique = 0
        while check_exact_match(df, white_balls) and attempts_unique < max_attempts_unique:
            if use_common_pairs:
                common_pairs_recheck = find_common_pairs(df, top_n=20)
                if num_range:
                    common_pairs_recheck = filter_common_pairs_by_range(common_pairs_recheck, num_range)
                if common_pairs_recheck:
                    white_balls = generate_with_common_pairs(df, common_pairs_recheck, white_ball_range, excluded_numbers)
                else:
                    white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
            else:
                random_row = df.sample(1).iloc[0]
                white_balls_base = [int(random_row['Number 1']), int(random_row['Number 2']), int(random_row['Number 3']), int(random_row['Number 4']), int(random_row['Number 5'])]
                powerball_base = int(random_row['Powerball'])
                white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, white_ball_range, powerball_range, excluded_numbers)
            attempts_unique += 1
        
        if attempts_unique == max_attempts_unique:
            flash("Could not find a unique modified combination after many attempts. Please try again.", 'error')
            last_draw_dict = last_draw.to_dict() # Ensure last_draw is passed even on error
            if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
                try:
                    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
                except ValueError:
                    pass
            return render_template('index.html', last_draw=last_draw_dict)

        white_balls = [int(num) for num in white_balls]
        powerball = int(powerball)

        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        last_draw_dict = last_draw.to_dict()
        if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
            try:
                last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
            except ValueError:
                pass

        return render_template('index.html', 
                            white_balls=white_balls, 
                            powerball=powerball, 
                            last_draw=last_draw_dict, 
                            last_draw_dates=last_draw_dates)
    except ValueError as e:
        flash(str(e), 'error')
        last_draw_dict = last_draw.to_dict()
        if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
            try:
                last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
            except ValueError:
                pass
        return render_template('index.html', last_draw=last_draw_dict)


@app.route('/frequency_analysis')
def frequency_analysis_route():
    white_ball_freq_list, powerball_freq_list = get_cached_analysis('freq_analysis', frequency_analysis, df)
    return render_template('frequency_analysis.html', 
                           white_ball_freq=white_ball_freq_list, 
                           powerball_freq=powerball_freq_list)

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    hot_numbers_list, cold_numbers_list = get_cached_analysis('hot_cold_numbers', hot_cold_numbers, df, last_draw['Draw Date'])
    return render_template('hot_cold_numbers.html', 
                           hot_numbers=hot_numbers_list, 
                           cold_numbers=cold_numbers_list)

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    monthly_balls = get_cached_analysis('monthly_balls', monthly_white_ball_analysis, df, last_draw['Draw Date'])
    monthly_balls_json = json.dumps(monthly_balls)
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_balls=monthly_balls,
                           monthly_balls_json=monthly_balls_json)


@app.route('/sum_of_main_balls')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot display Sum of Main Balls: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    sums_data_df, sum_freq_list, min_sum, max_sum, avg_sum = get_cached_analysis('sum_of_main_balls', sum_of_main_balls, df)
    
    sums_data = sums_data_df.to_dict('records')
    sum_freq_json = json.dumps(sum_freq_list)

    return render_template('sum_of_main_balls.html', 
                           sums_data=sums_data,
                           sum_freq=sum_freq_list,
                           sum_freq_json=sum_freq_json,
                           min_sum=min_sum,
                           max_sum=max_sum,
                           avg_sum=avg_sum)

@app.route('/find_results_by_sum', methods=['GET', 'POST'])
def find_results_by_sum_route():
    if df.empty:
        flash("Cannot find results: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    results = []
    target_sum_display = None

    if request.method == 'POST':
        target_sum_str = request.form.get('target_sum')
        if target_sum_str and target_sum_str.isdigit():
            target_sum = int(target_sum_str)
            target_sum_display = target_sum
            results_df = find_results_by_sum(df, target_sum)
            results = results_df.to_dict('records')
        else:
            flash("Please enter a valid number for Target Sum.", 'error')
    return render_template('find_results_by_sum.html', 
                           results=results,
                           target_sum=target_sum_display)

@app.route('/simulate_multiple_draws', methods=['GET', 'POST'])
def simulate_multiple_draws_route():
    if df.empty:
        flash("Cannot run simulation: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    simulated_freq_list = []
    num_draws_display = None

    if request.method == 'POST':
        num_draws_str = request.form.get('num_draws')
        if num_draws_str and num_draws_str.isdigit():
            num_draws = int(num_draws_str)
            num_draws_display = num_draws
            simulated_freq = simulate_multiple_draws(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers, num_draws)
            simulated_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in simulated_freq.items()]
        else:
            flash("Please enter a valid number for Number of Simulations.", 'error')

    return render_template('simulate_multiple_draws.html', 
                           simulated_freq=simulated_freq_list, 
                           num_simulations=num_draws_display)


@app.route('/winning_probability')
def winning_probability_route():
    probability_1_in_x, probability_percentage = get_cached_analysis('winning_probability', winning_probability, white_ball_range, powerball_range)
    return render_template('winning_probability.html', 
                           probability_1_in_x=probability_1_in_x, 
                           probability_percentage=probability_percentage)

@app.route('/partial_match_probabilities')
def partial_match_probabilities_route():
    probabilities = get_cached_analysis('partial_match_probabilities', partial_match_probabilities, white_ball_range, powerball_range)
    return render_template('partial_match_probabilities.html', 
                           probabilities=probabilities)

@app.route('/export_analysis_results')
def export_analysis_results_route():
    if df.empty:
        flash("Cannot export results: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    export_analysis_results(df) 
    flash("Analysis results exported to analysis_results.csv (this file is temporary on Vercel's serverless environment).", 'info')
    return redirect(url_for('index'))

@app.route('/number_age_distribution')
def number_age_distribution_route():
    number_age_counts, detailed_number_ages = get_cached_analysis('number_age_distribution', get_number_age_distribution, df)
    return render_template('number_age_distribution.html',
                           number_age_data=number_age_counts, # For the chart (renamed for clarity)
                           detailed_number_ages=detailed_number_ages) # For the new table

@app.route('/co_occurrence_analysis')
def co_occurrence_analysis_route():
    co_occurrence_data, max_co_occurrence = get_cached_analysis('co_occurrence_analysis', get_co_occurrence_matrix, df)
    return render_template('co_occurrence_analysis.html',
                           co_occurrence_data=co_occurrence_data,
                           max_co_occurrence=max_co_occurrence)

@app.route('/powerball_position_frequency')
def powerball_position_frequency_route():
    powerball_position_data = get_cached_analysis('powerball_position_frequency', get_powerball_position_frequency, df)
    return render_template('powerball_position_frequency.html',
                           powerball_position_data=powerball_position_data)

@app.route('/odd_even_trends')
def odd_even_trends_route():
    odd_even_trends = get_cached_analysis('odd_even_trends', get_odd_even_split_trends, df, last_draw['Draw Date'])
    return render_template('odd_even_trends.html',
                           odd_even_trends=odd_even_trends)

@app.route('/consecutive_trends')
def consecutive_trends_route():
    consecutive_trends = get_cached_analysis('consecutive_trends', get_consecutive_numbers_trends, df, last_draw['Draw Date'])
    return render_template('consecutive_trends.html',
                           consecutive_trends=consecutive_trends)

@app.route('/triplets_analysis')
def triplets_analysis_route():
    triplets_data = get_cached_analysis('triplets_analysis', get_most_frequent_triplets, df)
    return render_template('triplets_analysis.html',
                           triplets_data=triplets_data)

@app.route('/find_results_by_first_white_ball', methods=['GET', 'POST'])
def find_results_by_first_white_ball():
    if df.empty:
        flash("Cannot find results: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    results_dict = []
    white_ball_number_display = None
    sort_by_year_flag = False

    if request.method == 'POST':
        white_ball_number_str = request.form.get('white_ball_number')
        if white_ball_number_str and white_ball_number_str.isdigit():
            white_ball_number = int(white_ball_number_str)
            white_ball_number_display = white_ball_number
            
            results = df[df['Number 1'].astype(int) == white_ball_number].copy()

            if sort_by_year_flag:
                results['Year'] = pd.to_datetime(results['Draw Date'], errors='coerce').dt.year
                results = results.sort_values(by='Year')
            
            results_dict = results.to_dict('records')
        else:
            flash("Please enter a valid number for First White Ball Number.", 'error')

    return render_template('find_results_by_first_white_ball.html', 
                           results_by_first_white_ball=results_dict, 
                           white_ball_number=white_ball_number_display,
                           sort_by_year=sort_by_year_flag)

@app.route('/update_powerball_data', methods=['GET'])
def update_powerball_data():
    service_headers = _get_supabase_headers(is_service_key=True)
    anon_headers = _get_supabase_headers(is_service_key=False)

    try:
        url_check_latest = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        params_check_latest = {
            'select': 'Draw Date',
            'order': 'Draw Date.desc',
            'limit': 1
        }
        response_check_latest = requests.get(url_check_latest, headers=anon_headers, params=params_check_latest)
        response_check_latest.raise_for_status()
        
        latest_db_draw_data = response_check_latest.json()
        last_db_draw_date = None
        if latest_db_draw_data:
            last_db_draw_date = latest_db_draw_data[0]['Draw Date']
        
        print(f"Last draw date in Supabase: {last_db_draw_date}")

        simulated_draw_date_dt = datetime.now()
        simulated_draw_date = simulated_draw_date_dt.strftime('%Y-%m-%d')
        simulated_numbers = sorted(random.sample(range(1, 70), 5))
        simulated_powerball = random.randint(1, 26)

        new_draw_data = {
            'Draw Date': simulated_draw_date,
            'Number 1': simulated_numbers[0],
            'Number 2': simulated_numbers[1],
            'Number 3': simulated_numbers[2],
            'Number 4': simulated_numbers[3],
            'Number 5': simulated_numbers[4],
            'Powerball': simulated_powerball
        }
        
        print(f"Simulated new draw data: {new_draw_data}")

        if new_draw_data['Draw Date'] == last_db_draw_date:
            print(f"Draw for {new_draw_data['Draw Date']} already exists. No update needed.")
            return "No new draw data. Database is up-to-date.", 200
        
        url_insert = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        insert_response = requests.post(url_insert, headers=service_headers, data=json.dumps(new_draw_data))
        insert_response.raise_for_status()

        if insert_response.status_code == 201:
            print(f"Successfully inserted new draw: {new_draw_data}")
            
            # Re-load core data and invalidate analysis cache
            initialize_core_data() 
            invalidate_analysis_cache()

            return f"Data updated successfully with draw for {simulated_draw_date}.", 200
        else:
            print(f"Failed to insert data. Status: {insert_response.status_code}, Response: {insert_response.text}")
            return f"Error updating data: {insert_response.status_code} - {insert_response.text}", 500

    except requests.exceptions.RequestException as e:
        print(f"Network or HTTP error during update_powerball_data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Supabase response content: {e.response.text}")
        return f"Network or HTTP error: {e}", 500
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in update_powerball_data: {e}")
        if 'insert_response' in locals() and insert_response is not None:
            print(f"Response content that failed JSON decode: {insert_response.text}")
        return f"JSON parsing error: {e}", 500
    except Exception as e:
        print(f"An unexpected error occurred during data update: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for update errors
        return f"An internal error occurred: {e}", 500

# NEW ROUTE for saving manual official draws
@app.route('/save_official_draw', methods=['POST'])
def save_official_draw_route():
    try:
        draw_date = request.form.get('draw_date')
        n1 = int(request.form.get('n1'))
        n2 = int(request.form.get('n2'))
        n3 = int(request.form.get('n3'))
        n4 = int(request.form.get('n4'))
        n5 = int(request.form.get('n5'))
        pb = int(request.form.get('pb'))

        # Basic validation for numbers (1-69 for white, 1-26 for powerball)
        if not (1 <= n1 <= 69 and 1 <= n2 <= 69 and 1 <= n3 <= 69 and 1 <= n4 <= 69 and 1 <= n5 <= 69 and 1 <= pb <= 26):
            flash("White balls must be between 1-69 and Powerball between 1-26.", 'error')
            return redirect(url_for('index'))
        
        # Check for duplicate white balls within the submitted draw
        submitted_white_balls = sorted([n1, n2, n3, n4, n5])
        if len(set(submitted_white_balls)) != 5:
            flash("White ball numbers must be unique within a single draw.", 'error')
            return redirect(url_for('index'))

        success, message = save_manual_draw_to_db(draw_date, n1, n2, n3, n4, n5, pb)
        if success:
            flash(message, 'info')
            # Re-load core data and invalidate analysis cache
            initialize_core_data()
            invalidate_analysis_cache()
        else:
            flash(message, 'error')
    except ValueError:
        flash("Invalid input. Please ensure all numbers and date are correctly entered.", 'error')
    except Exception as e:
        flash(f"An error occurred: {e}", 'error')
    return redirect(url_for('index'))

# NEW ROUTE for saving generated numbers
@app.route('/save_generated_pick', methods=['POST'])
def save_generated_pick_route():
    try:
        white_balls_str = request.form.get('generated_white_balls')
        powerball_str = request.form.get('generated_powerball')

        if not white_balls_str or not powerball_str:
            flash("No numbers generated to save.", 'error')
            return redirect(url_for('index'))

        white_balls = [int(x.strip()) for x in white_balls_str.split(',') if x.strip().isdigit()]
        powerball = int(powerball_str)

        if len(white_balls) != 5:
            flash("Invalid white balls format. Expected 5 numbers.", 'error')
            return redirect(url_for('index'))
        
        # Basic validation for numbers (1-69 for white, 1-26 for powerball)
        if not (all(1 <= n <= 69 for n in white_balls) and 1 <= powerball <= 26):
            flash("White balls must be between 1-69 and Powerball between 1-26 for saving.", 'error')
            return redirect(url_for('index'))

        success, message = save_generated_numbers_to_db(white_balls, powerball)
        if success:
            flash(message, 'info')
            # Only update the generated history cache, not analysis cache for this
            analysis_cache['generated_history'] = get_generated_numbers_history()
        else:
            flash(message, 'error')

    except ValueError:
        flash("Invalid number format for saving generated numbers.", 'error')
    except Exception as e:
        flash(f"An error occurred while saving generated numbers: {e}", 'error')
    return redirect(url_for('index'))

# NEW ROUTE for viewing generated numbers history
@app.route('/generated_numbers_history')
def generated_numbers_history_route():
    generated_history = get_cached_analysis('generated_history', get_generated_numbers_history)
    return render_template('generated_numbers_history.html', 
                           generated_history=generated_history) # Pass grouped data

# NEW ROUTE for analyzing generated numbers against historical data
@app.route('/analyze_generated_historical_matches', methods=['POST'])
def analyze_generated_historical_matches_route():
    if df.empty:
        flash("Cannot perform historical analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    try:
        generated_white_balls_str = request.form.get('generated_white_balls')
        generated_powerball_str = request.form.get('generated_powerball')

        if not generated_white_balls_str or not generated_powerball_str:
            flash("No generated numbers provided for analysis.", 'error')
            return redirect(url_for('index'))

        generated_white_balls = sorted([int(x.strip()) for x in generated_white_balls_str.split(',') if x.strip().isdigit()])
        generated_powerball = int(generated_powerball_str)

        if len(generated_white_balls) != 5:
            flash("Invalid generated white balls format. Expected 5 numbers for analysis.", 'error')
            return redirect(url_for('index'))

        # This analysis is specific to the generated numbers, so it won't be cached in the general analysis_cache
        # But if you wanted to cache specific generated number analyses, you'd extend get_cached_analysis with a key unique to the combination
        historical_match_results = check_generated_against_history(generated_white_balls, generated_powerball, df)
        
        return render_template('historical_match_analysis_results.html', 
                               generated_numbers_for_analysis=generated_white_balls,
                               generated_powerball_for_analysis=generated_powerball,
                               match_summary=historical_match_results['summary'])

    except ValueError:
        flash("Invalid number format for historical analysis.", 'error')
    except Exception as e:
        flash(f"An error occurred during historical analysis: {e}", 'error')
        import traceback
        traceback.print_exc()
    return redirect(url_for('index'))
