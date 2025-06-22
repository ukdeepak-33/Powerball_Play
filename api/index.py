import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NzMwNjUsImexFQI6MjA2NTM0OTA2NX0.AzUD7wjR7VbvtUH27NDqJ3AlvFW0nCWpiN9ADG8T_t4")
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

# Set to store all historical white ball combinations for fast lookup
historical_white_ball_sets = set() 

# Cache for precomputed analysis data
analysis_cache = {}
last_analysis_cache_update = datetime.min # Initialize with the earliest possible datetime

# Cache expiration time (e.g., 1 hour for analysis data)
CACHE_EXPIRATION_SECONDS = 3600

# Group A numbers (constants)
group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]
# Global default ranges - used if no specific range is provided by the user
GLOBAL_WHITE_BALL_RANGE = (1, 69)
GLOBAL_POWERBALL_RANGE = (1, 26)
excluded_numbers = []

# Define number ranges for grouped patterns analysis
NUMBER_RANGES = {
    "1-9": (1, 9),
    "10s": (10, 19),
    "20s": (20, 29),
    "30s": (30, 39),
    "40s": (40, 49),
    "50s": (50, 59),
    "60s": (60, 69)
}


# --- Core Utility Functions (All helpers defined here) ---

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
            'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A',
            'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A'] # Ensure 'Numbers' key is present
        }, dtype='object')
    
    last_row = df.iloc[-1].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Manually format 'Numbers' list if not already present
    if 'Numbers' not in last_row or not isinstance(last_row['Numbers'], list):
        last_row['Numbers'] = [
            int(last_row['Number 1']), int(last_row['Number 2']), int(last_row['Number 3']), 
            int(last_row['Number 4']), int(last_row['Number 5'])
        ]
    return last_row

def check_exact_match(white_balls):
    """
    Checks if the given white_balls combination exactly matches any historical draw.
    Uses the precomputed global historical_white_ball_sets for efficient lookup.
    """
    global historical_white_ball_sets
    return frozenset(white_balls) in historical_white_ball_sets

def generate_powerball_numbers(df_source, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None):
    if df_source.empty:
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

        last_draw_data = get_last_draw(df_source)
        if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
            last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
            if set(white_balls) == set(last_white_balls) and powerball == int(last_draw_data['Powerball']):
                attempts += 1
                continue

        if check_exact_match(white_balls): 
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

def generate_from_group_a(df_source, num_from_group_a, white_ball_range, powerball_range, excluded_numbers):
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 1000
    attempts = 0
    
    valid_group_a = [num for num in group_a if white_ball_range[0] <= num <= white_ball_range[1] and num not in excluded_numbers]
    
    remaining_pool = [num for num in range(white_ball_range[0], white_ball_range[1] + 1)
                      if num not in valid_group_a and num not in excluded_numbers]

    if len(valid_group_a) < num_from_group_a:
        raise ValueError(f"Not enough unique numbers in Group A ({len(valid_group_a)}) to pick {num_from_group_a}.")
    
    num_from_remaining = 5 - num_from_group_a
    if len(remaining_pool) < num_from_remaining:
        raise ValueError(f"Not enough unique numbers in the remaining pool ({len(remaining_pool)}) to pick {num_from_remaining}.")

    while attempts < max_attempts:
        try:
            selected_from_group_a = random.sample(valid_group_a, num_from_group_a)
            
            # Ensure numbers picked from remaining pool are not already in selected_from_group_a
            available_for_remaining = [num for num in remaining_pool if num not in selected_from_group_a]
            if len(available_for_remaining) < num_from_remaining:
                attempts += 1
                continue # Retry if not enough unique numbers for remaining

            selected_from_remaining = random.sample(available_for_remaining, num_from_remaining) 
            
            white_balls = sorted(selected_from_group_a + selected_from_remaining)
            
            powerball = random.randint(powerball_range[0], powerball_range[1])

            if check_exact_match(white_balls): 
                attempts += 1
                continue

            break
        except ValueError as e:
            print(f"Attempt failed during group_a strategy: {e}. Retrying...")
            attempts += 1
            continue
    else:
        raise ValueError("Could not generate a unique combination with Group A strategy after many attempts.")

    return white_balls, powerball


def check_historical_match(df_source, white_balls, powerball):
    if df_source.empty: return None
    for _, row in df_source.iterrows():
        historical_white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        historical_powerball = int(row['Powerball'])
        if set(white_balls) == set(historical_white_balls) and powerball == historical_powerball:
            return row['Draw Date']
    return None

def frequency_analysis(df_source):
    if df_source.empty: 
        return [], []
    white_balls = df_source[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)
    powerball_freq = df_source['Powerball'].astype(int).value_counts().reindex(range(1, 27), fill_value=0)
    
    white_ball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.items()]
    powerball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in powerball_freq.items()]

    return white_ball_freq_list, powerball_freq_list

def hot_cold_numbers(df_source, last_draw_date_str):
    if df_source.empty or last_draw_date_str == 'N/A': 
        return [], []
    
    last_draw_date = pd.to_datetime(last_draw_date_str)
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    
    recent_data = df_source[df_source['Draw Date_dt'] >= one_year_ago].copy()
    if recent_data.empty: 
        return [], []

    white_balls = recent_data[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts()

    hot_numbers = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.nlargest(14).sort_values(ascending=False).items()]
    cold_numbers = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.nsmallest(14).sort_values(ascending=True).items()]

    return hot_numbers, cold_numbers

def monthly_white_ball_analysis(df_source, last_draw_date_str):
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
        if 'Month' not in recent_data.columns:
            recent_data['Month'] = recent_data['Draw Date_dt'].dt.to_period('M')
            print(f"[DEBUG-Monthly] 'Month' column added to recent_data. First 2 months: {recent_data['Month'].head(2).tolist()}")
        
        required_cols = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
        for col in required_cols:
            if col in recent_data.columns:
                recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
            else:
                print(f"[ERROR-Monthly] Missing required column '{col}' in recent_data. Cannot perform analysis.")
                return {}

        recent_data = recent_data.dropna(subset=required_cols)
        if recent_data.empty:
            print("[DEBUG-Monthly] recent_data is empty after dropping NaN in ball columns. Returning empty dict.")
            return {}

        monthly_balls_raw = recent_data.groupby('Month')[required_cols].apply(
            lambda x: sorted([int(num) for num in x.values.flatten() if not pd.isna(num)])
        ).to_dict()

        monthly_balls_str_keys = {}
        for period_key, ball_list in monthly_balls_raw.items():
            monthly_balls_str_keys[str(period_key)] = [int(ball) for ball in ball_list]
        
        print(f"[DEBUG-Monthly] Groupby and apply successful. First item in monthly_balls_str_keys: {next(iter(monthly_balls_str_keys.items())) if monthly_balls_str_keys else 'N/A'}")

    except Exception as e:
        print(f"[ERROR-Monthly] Error during groupby/apply operation or conversion: {e}. Returning empty dict.")
        import traceback
        traceback.print_exc()
        return {}
    
    print("[DEBUG-Monthly] Successfully computed monthly_balls_str_keys.")
    return monthly_balls_str_keys


def sum_of_main_balls(df_source):
    """Calculates the sum of the five main white balls for each draw."""
    if df_source.empty:
        return pd.DataFrame(), [], 0, 0, 0.0
    
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)
    
    temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    sum_freq = temp_df['Sum'].value_counts().sort_index()
    sum_freq_list = [{'sum': int(s), 'count': int(c)} for s, c in sum_freq.items()]

    min_sum = int(temp_df['Sum'].min()) if not temp_df['Sum'].empty else 0
    max_sum = int(temp_df['Sum'].max()) if not temp_df['Sum'].empty else 0
    avg_sum = round(temp_df['Sum'].mean(), 2) if not temp_df['Sum'].empty else 0.0

    return temp_df[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum']], sum_freq_list, min_sum, max_sum, avg_sum

def find_results_by_sum(df_source, target_sum):
    if df_source.empty: return pd.DataFrame()
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']: # Include Powerball here
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)

    if 'Sum' not in temp_df.columns:
        temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    results = temp_df[temp_df['Sum'] == target_sum]
    # Return all necessary columns for rendering, including Powerball
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum']]

def simulate_multiple_draws(df_source, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
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

def winning_probability(white_ball_range_tuple, powerball_range_tuple):
    total_white_balls_in_range = white_ball_range_tuple[1] - white_ball_range_tuple[0] + 1
    white_ball_combinations = calculate_combinations(total_white_balls_in_range, 5)

    total_powerballs_in_range = powerball_range_tuple[1] - powerball_range_tuple[0] + 1

    total_combinations = white_ball_combinations * total_powerballs_in_range

    probability_1_in_x = f"1 in {total_combinations:,}" if total_combinations > 0 else "N/A"
    probability_percentage = f"{1 / total_combinations * 100:.10f}%" if total_combinations > 0 else "N/A"

    return probability_1_in_x, probability_percentage

def partial_match_probabilities(white_ball_range_tuple, powerball_range_tuple):
    total_white_balls_in_range = white_ball_range_tuple[1] - white_ball_range_tuple[0] + 1
    total_powerballs_in_range = powerball_range_tuple[1] - powerball_range_tuple[0] + 1

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
        comb_matched_w = calculate_combinations(5, data["matched_w"])
        comb_unmatched_w = calculate_combinations(total_white_balls_in_range - 5, data["unmatched_w"])

        if data["matched_p"] == 1:
            comb_p = 1
        else:
            comb_p = total_powerballs_in_range - 1
            if comb_p < 0:
                comb_p = 0
        
        numerator = comb_matched_w * comb_unmatched_w * comb_p
        
        if numerator == 0:
            probabilities[scenario] = "N/A"
        else:
            total_possible_combinations_for_draw = calculate_combinations(total_white_balls_in_range, 5) * total_powerballs_in_range
            
            probability = total_possible_combinations_for_draw / numerator
            probabilities[scenario] = f"{probability:,.0f} to 1"

    return probabilities


def export_analysis_results(df_source, file_path="analysis_results.csv"):
    df_source.to_csv(file_path, index=False)
    print(f"Analysis results saved to {file_path}")

def find_last_draw_dates_for_numbers(df_source, white_balls, powerball):
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

def modify_combination(df_source, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers):
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

def find_common_pairs(df_source, top_n=10):
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

def generate_with_common_pairs(df_source, common_pairs, white_ball_range, excluded_numbers):
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

def get_number_age_distribution(df_source):
    if df_source.empty: return [], []
    df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'])
    all_draw_dates = sorted(df_source['Draw Date_dt'].drop_duplicates().tolist())
    
    detailed_ages = []
    
    for i in range(1, 70):
        last_appearance_date = None
        last_appearance_date_str = "N/A" # Default to N/A
        temp_df_filtered = df_source[(df_source['Number 1'].astype(int) == i) | (df_source['Number 2'].astype(int) == i) |
                              (df_source['Number 3'].astype(int) == i) | (df_source['Number 4'].astype(int) == i) |
                              (df_source['Number 5'].astype(int) == i)]
        
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()
            last_appearance_date_str = last_appearance_date.strftime('%Y-%m-%d')

        miss_streak_count = 0
        if last_appearance_date is not None:
            for d_date in reversed(all_draw_dates):
                if d_date > last_appearance_date:
                    miss_streak_count += 1
                else:
                    break
            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': miss_streak_count, 'last_drawn_date': last_appearance_date_str})
        else:
            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': len(all_draw_dates), 'last_drawn_date': last_appearance_date_str})

    for i in range(1, 27):
        last_appearance_date = None
        last_appearance_date_str = "N/A" # Default to N/A
        temp_df_filtered = df_source[df_source['Powerball'].astype(int) == i]
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()
            last_appearance_date_str = last_appearance_date.strftime('%Y-%m-%d')

        miss_streak_count = 0
        if last_appearance_date is not None:
            for d_date in reversed(all_draw_dates):
                if d_date > last_appearance_date:
                    miss_streak_count += 1
                else:
                    break
            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': miss_streak_count, 'last_drawn_date': last_appearance_date_str})
        else:
            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': len(all_draw_dates), 'last_drawn_date': last_appearance_date_str})

    all_miss_streaks_only = [item['age'] for item in detailed_ages]
    age_counts = pd.Series(all_miss_streaks_only).value_counts().sort_index()
    age_counts_list = [{'age': int(age), 'count': int(count)} for age, count in age_counts.items()]
    
    return age_counts_list, detailed_ages

def get_co_occurrence_matrix(df_source):
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

def get_powerball_position_frequency(df_source):
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

def _find_consecutive_pairs(numbers_list):
    """Identifies and returns all consecutive pairs in a sorted list of numbers."""
    pairs = []
    sorted_nums = sorted(numbers_list)
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i] + 1 == sorted_nums[i+1]:
            pairs.append([sorted_nums[i], sorted_nums[i+1]])
    return pairs

def get_consecutive_numbers_trends(df_source, last_draw_date_str):
    print("[DEBUG-ConsecutiveTrends] Inside get_consecutive_numbers_trends function.")
    if df_source.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-ConsecutiveTrends] df_source is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-ConsecutiveTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-ConsecutiveTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return {}

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-ConsecutiveTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-ConsecutiveTrends] 'Draw Date_dt' column missing or not datetime type in df_source. Returning empty list.")
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt', ascending=False)
    if recent_data.empty:
        print("[DEBUG-ConsecutiveTrends] recent_data is empty after filtering. Returning empty list.")
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        consecutive_pairs = _find_consecutive_pairs(white_balls)
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            'consecutive_present': "Yes" if consecutive_pairs else "No",
            'consecutive_pairs': consecutive_pairs
        })
    
    print(f"[DEBUG-ConsecutiveTrends] Generated {len(trend_data)} trend data points.")
    return trend_data

def get_most_frequent_triplets(df_source, top_n=10):
    print("[DEBUG-Triplets] Inside get_most_frequent_triplets function.")
    if df_source.empty:
        print("[DEBUG-Triplets] df_source is empty. Returning empty list.")
        return []

    triplet_counts = defaultdict(int)

    for idx, row in df_source.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        for triplet_combo in combinations(sorted(white_balls), 3):
            triplet_counts[triplet_combo] += 1
    
    sorted_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)
    
    formatted_triplets = []
    for triplet, count in sorted_triplets[:top_n]:
        formatted_triplets.append({
            'triplet': list(triplet),
            'count': int(count)
        })
    
    print(f"[DEBUG-Triplets] Found {len(triplet_counts)} unique triplets. Returning top {len(formatted_triplets)}.")
    return formatted_triplets


def get_odd_even_split_trends(df_source, last_draw_date_str):
    print("[DEBUG-OddEvenTrends] Inside get_odd_even_split_trends function.")
    if df_source.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-OddEvenTrends] df_source is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-OddEvenTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-OddEvenTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return {}

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-OddEvenTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-OddEvenTrends] 'Draw Date_dt' column missing or not datetime type in df_source. Returning empty list.")
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt', ascending=False)
    if recent_data.empty:
        print("[DEBUG-OddEvenTrends] recent_data is empty after filtering. Returning empty list.")
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        # Calculate WB_Sum
        wb_sum = sum(white_balls)

        # Identify group_a numbers present in the draw
        group_a_numbers_present = sorted([num for num in white_balls if num in group_a])

        even_count = sum(1 for num in white_balls if num % 2 == 0)
        odd_count = 5 - even_count

        split_category = "Other"

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
            'split_category': split_category,
            'wb_sum': wb_sum,
            'group_a_numbers': group_a_numbers_present
        })
    
    print(f"[DEBUG-OddEvenTrends] Generated {len(trend_data)} trend data points with WB_Sum and Group A numbers.")
    return trend_data

def get_powerball_frequency_by_year(df_source, num_years=5):
    """
    Calculates the frequency of each Powerball number per year for the last `num_years`.
    Returns a list of dictionaries, where each dict represents a Powerball number
    and its count for each of the last `num_years`.
    """
    print(f"[DEBUG-YearlyPB] Inside get_powerball_frequency_by_year for last {num_years} years.")
    if df_source.empty:
        print("[DEBUG-YearlyPB] df_source is empty. Returning empty data.")
        return [], []

    current_year = datetime.now().year
    
    years = [y for y in range(current_year - num_years + 1, current_year + 1)]
    print(f"[DEBUG-YearlyPB] Years to analyze: {years}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-YearlyPB] 'Draw Date_dt' column missing or not datetime type. Attempting to re-create.")
        df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'], errors='coerce')
        df_source = df_source.dropna(subset=['Draw Date_dt'])
        if df_source.empty:
            print("[ERROR-YearlyPB] Re-creation failed or resulted in empty df. Returning empty data.")
            return [], []

    recent_data = df_source[df_source['Draw Date_dt'].dt.year.isin(years)].copy()
    
    if recent_data.empty:
        print("[DEBUG-YearlyPB] recent_data is empty after filtering by years. Returning empty data.")
        return [], years

    recent_data['Year'] = recent_data['Draw Date_dt'].dt.year

    recent_data['Powerball'] = pd.to_numeric(recent_data['Powerball'], errors='coerce').fillna(0).astype(int)
    
    yearly_pb_freq_pivot = pd.pivot_table(
        recent_data,
        index='Powerball',
        columns='Year',
        values='Draw Date',
        aggfunc='count',
        fill_value=0
    )
    
    all_powerballs = pd.Series(range(1, 27))
    yearly_pb_freq_pivot = yearly_pb_freq_pivot.reindex(all_powerballs, fill_value=0)

    yearly_pb_freq_pivot = yearly_pb_freq_pivot.reindex(columns=years, fill_value=0)
    
    formatted_data = []
    for powerball_num, row in yearly_pb_freq_pivot.iterrows():
        row_dict = {'Powerball': int(powerball_num)}
        for year in years:
            row_dict[f'Year_{year}'] = int(row[year])
        formatted_data.append(row_dict)
    
    formatted_data = sorted(formatted_data, key=lambda x: x['Powerball'])

    print(f"[DEBUG-YearlyPB] Successfully computed yearly Powerball frequencies. First 3: {formatted_data[:3]}")
    return formatted_data, years

def _get_generated_picks_for_date_from_db(date_str):
    """
    Fetches generated numbers for a specific date from the database.
    Returns a list of dicts: [{'white_balls': [...], 'powerball': int}].
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{GENERATED_NUMBERS_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=False)
    
    try:
        start_of_day_dt = datetime.strptime(date_str, '%Y-%m-%d')
        end_of_day_dt = start_of_day_dt + timedelta(days=1)
        start_of_day_iso = start_of_day_dt.isoformat(timespec='seconds') + "Z"
        end_of_day_iso = end_of_day_dt.isoformat(timespec='seconds') + "Z"
    except ValueError:
        print(f"Invalid date format for generated_date_str: {date_str}")
        return []

    params = {
        'select': 'generated_date,number_1,number_2,number_3,number_4,number_5,powerball',
        'generated_date': f'gte.{start_of_day_iso}',
        'and': (f'(generated_date.lt.{end_of_day_iso})',) 
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        raw_data = response.json()
        
        formatted_picks = []
        for record in raw_data:
            white_balls = sorted([
                int(record['number_1']), int(record['number_2']), int(record['number_3']),
                int(record['number_4']), int(record['number_5'])
            ])
            formatted_picks.append({
                'white_balls': white_balls,
                'powerball': int(record['powerball'])
            })
        return formatted_picks
    except requests.exceptions.RequestException as e:
        print(f"Error fetching generated picks for date {date_str}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in _get_generated_picks_for_date_from_db: {e}")
        return []

def _get_official_draw_for_date_from_db(date_str):
    """
    Fetches a single official draw for a specific date from the database.
    Returns a dict: {'Draw Date': ..., 'Number 1': ..., 'Powerball': ...} or None.
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=False)
    
    params = {
        'select': 'Draw Date,Number 1,Number 2,Number 3,Number 4,Number 5,Powerball',
        'Draw Date': f'eq.{date_str}'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        raw_data = response.json()
        if raw_data:
            return raw_data[0] # Should only be one draw for a specific date
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching official draw for date {date_str}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in _get_official_draw_for_date_from_db: {e}")
        return None

def analyze_generated_batch_against_official_draw(generated_picks_list, official_draw):
    """
    Analyzes a batch of generated picks against a single official draw result.
    Returns a summary of matches.
    """
    summary = {
        "Match 5 White Balls + Powerball": {"count": 0},
        "Match 5 White Balls Only": {"count": 0},
        "Match 4 White Balls + Powerball": {"count": 0},
        "Match 4 White Balls Only": {"count": 0},
        "Match 3 White Balls + Powerball": {"count": 0},
        "Match 3 White Balls Only": {"count": 0},
        "Match 2 White Balls + Powerball": {"count": 0},
        "Match 1 White Ball + Powerball": {"count": 0},
        "Match Powerball Only": {"count": 0},
        "No Match": {"count": 0}
    }
    
    if not official_draw:
        return summary # Cannot analyze without an official draw

    official_white_balls = sorted([
        int(official_draw['Number 1']), int(official_draw['Number 2']), int(official_draw['Number 3']),
        int(official_draw['Number 4']), int(official_draw['Number 5'])
    ])
    official_powerball = int(official_draw['Powerball'])
    official_white_set = set(official_white_balls)

    for pick in generated_picks_list:
        generated_white_balls = sorted(pick['white_balls'])
        generated_powerball = pick['powerball']
        generated_white_set = set(generated_white_balls)

        white_matches = len(generated_white_set.intersection(official_white_set))
        powerball_match = 1 if generated_powerball == official_powerball else 0

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
        
        summary[category]["count"] += 1
    
    return summary

def save_manual_draw_to_db(draw_date, n1, n2, n3, n4, n5, pb):
    """
    Saves a manually entered official Powerball draw to the 'powerball_draws' table.
    Checks for existence of the draw date to prevent duplicates.
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=True)

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
    headers = _get_supabase_headers(is_service_key=True)

    sorted_numbers = sorted(numbers)

    check_params = {
        'select': 'id',
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

    new_generated_data = {
        'number_1': sorted_numbers[0],
        'number_2': sorted_numbers[1],
        'number_3': sorted_numbers[2],
        'number_4': sorted_numbers[3],
        'number_5': sorted_numbers[4],
        'powerball': powerball,
        'generated_date': datetime.now().isoformat()
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
        headers = _get_supabase_headers(is_service_key=False)
        
        while True:
            params = {
                'select': 'generated_date,number_1,number_2,number_3,number_4,number_5,powerball',
                'order': 'generated_date.desc',
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

        grouped_data = defaultdict(list)
        for record in all_data:
            gen_dt = datetime.fromisoformat(record['generated_date'].replace('Z', '+00:00'))
            date_key = gen_dt.strftime('%Y-%m-%d')
            
            formatted_time = gen_dt.strftime('%I:%M %p')

            generated_balls = sorted([
                int(record['number_1']), int(record['number_2']), int(record['number_3']),
                int(record['number_4']), int(record['number_5'])
            ])
            
            grouped_data[date_key].append({
                'time': formatted_time,
                'white_balls': generated_balls,
                'powerball': int(record['powerball'])
            })
        
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

    two_years_ago = datetime.now() - timedelta(days=2 * 365)
    recent_historical_data = df_historical[df_historical['Draw Date_dt'] >= two_years_ago].copy()

    if recent_historical_data.empty:
        return results

    gen_white_set = set(generated_white_balls)

    for index, row in recent_historical_data.iterrows():
        historical_white_balls = sorted([
            int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
            int(row['Number 4']), int(row['Number 5'])
        ])
        historical_powerball = int(row['Powerball'])
        historical_draw_date = row['Draw Date']

        hist_white_set = set(historical_white_balls)

        white_matches = len(gen_white_set.intersection(hist_white_set))

        powerball_match = 1 if generated_powerball == historical_powerball else 0

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
        
        # Corrected lines: access 'summary' through the 'results' dictionary
        results["summary"][category]["count"] += 1
        results["summary"][category]["draws"].append({
            "date": historical_draw_date,
            "white_balls": historical_white_balls,
            "powerball": historical_powerball
        })
    
    for category in results["summary"]:
        results["summary"][category]["draws"].sort(key=lambda x: x['date'], reverse=True)

    return results


def get_grouped_patterns_over_years(df_source):
    if df_source.empty:
        print("[DEBUG-GroupedPatterns] df_source is empty. Returning empty list.")
        return []

    df_source_copy = df_source.copy()
    if 'Draw Date_dt' not in df_source_copy.columns:
        df_source_copy['Draw Date_dt'] = pd.to_datetime(df_source_copy['Draw Date'], errors='coerce')
    df_source_copy = df_source_copy.dropna(subset=['Draw Date_dt'])
    
    if df_source_copy.empty:
        print("[DEBUG-GroupedPatterns] df_source_copy is empty after datetime conversion. Returning empty list.")
        return []

    # Ensure white ball columns are numeric
    for i in range(1, 6):
        col = f'Number {i}'
        if col in df_source_copy.columns:
            df_source_copy[col] = pd.to_numeric(df_source_copy[col], errors='coerce').fillna(0).astype(int)
        else:
            print(f"[WARN-GroupedPatterns] Column '{col}' not found in DataFrame for pattern analysis.")


    all_patterns_data = []
    
    # Iterate through unique years
    for year in sorted(df_source_copy['Draw Date_dt'].dt.year.unique()):
        yearly_df = df_source_copy[df_source_copy['Draw Date_dt'].dt.year == year]
        
        # Store counts for pairs and triplets for the current year
        year_pairs_counts = defaultdict(int)
        year_triplets_counts = defaultdict(int)

        for _, row in yearly_df.iterrows():
            # Get white balls, ensuring they are integers and handling potential NaNs after conversion
            white_balls = [int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])]
            
            for range_name, (min_val, max_val) in NUMBER_RANGES.items():
                numbers_in_current_range = sorted([num for num in white_balls if min_val <= num <= max_val])
                
                if len(numbers_in_current_range) >= 2:
                    for pair in combinations(numbers_in_current_range, 2):
                        year_pairs_counts[(range_name, tuple(sorted(pair)))] += 1
                
                if len(numbers_in_current_range) >= 3:
                    for triplet in combinations(numbers_in_current_range, 3):
                        year_triplets_counts[(range_name, tuple(sorted(triplet)))] += 1
        
        # Add to main results list
        for (range_name, pattern), count in year_pairs_counts.items():
            all_patterns_data.append({
                "year": int(year),
                "range": range_name,
                "type": "Pair",
                "pattern": list(pattern),
                "count": int(count)
            })
        
        for (range_name, pattern), count in year_triplets_counts.items():
            all_patterns_data.append({
                "year": int(year),
                "range": range_name,
                "type": "Triplet",
                "pattern": list(pattern),
                "count": int(count)
            })

    # Sort by count descending, then by year, then by range, then by pattern
    # Convert pattern lists to strings for consistent sorting as a secondary key if counts are equal
    all_patterns_data.sort(key=lambda x: (x['count'], x['year'], x['range'], str(x['pattern'])), reverse=True)
    
    print(f"[DEBUG-GroupedPatterns] Generated {len(all_patterns_data)} grouped patterns data points.")
    return all_patterns_data

# --- Data Initialization (Call after all helper functions are defined) ---
def initialize_core_data():
    global df, last_draw, historical_white_ball_sets
    print("Attempting to load core historical data...")
    try:
        df_temp = load_historical_data_from_supabase()
        if not df_temp.empty:
            df = df_temp
            last_draw = get_last_draw(df)
            # Ensure last_draw has 'Numbers' formatted for display
            if not last_draw.empty and last_draw.get('Draw Date') != 'N/A':
                # No need to convert to_datetime here, just strftime if it's not 'N/A'
                last_draw['Draw Date'] = last_draw['Draw Date_dt'].strftime('%Y-%m-%d')
                # The get_last_draw already sets 'Numbers'
            print("Core historical data loaded successfully.")
        else:
            print("Core historical data is empty after loading. df remains empty.")
            # Set default N/A for last_draw if no data is found
            last_draw = pd.Series({
                'Draw Date': 'N/A', 'Number 1': 'N/A', 'Number 2': 'N/A',
                'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A',
                'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A']
            }, dtype='object')
    except Exception as e:
        print(f"An error occurred during initial core data loading: {e}")
        import traceback
        traceback.print_exc()

initialize_core_data()


# --- Analysis Cache Management ---
def get_cached_analysis(key, compute_function, *args, **kwargs):
    global analysis_cache, last_analysis_cache_update
    
    if key in analysis_cache and (datetime.now() - last_analysis_cache_update).total_seconds() < CACHE_EXPIRATION_SECONDS:
        print(f"Serving '{key}' from cache.")
        return analysis_cache[key]
    
    print(f"Computing and caching '{key}'.")
    computed_data = compute_function(*args, **kwargs)
    
    analysis_cache[key] = computed_data
    last_analysis_cache_update = datetime.now()
    return computed_data

def invalidate_analysis_cache():
    global analysis_cache, last_analysis_cache_update
    analysis_cache = {}
    last_analysis_cache_update = datetime.min
    print("Analysis cache invalidated.")


# --- Flask Routes (Ordered for Dependency - all UI-facing routes first, then API routes) ---

# Core home page route
@app.route('/')
def index():
    last_draw_dict = last_draw.to_dict()
    return render_template('index.html', last_draw=last_draw_dict)

# Generation Routes (referenced directly in index.html forms)
@app.route('/generate', methods=['POST'])
def generate():
    if df.empty:
        flash("Cannot generate numbers: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

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
        return render_template('index.html', last_draw=last_draw.to_dict())

    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='generated')

@app.route('/generate_modified', methods=['POST'])
def generate_modified():
    if df.empty:
        flash("Cannot generate modified combination: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

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
            return render_template('index.html', last_draw=last_draw.to_dict())


        if use_common_pairs:
            common_pairs = find_common_pairs(df, top_n=20)
            if num_range:
                common_pairs = filter_common_pairs_by_range(common_pairs, num_range)
            
            if not common_pairs:
                flash("No common pairs found with the specified filter. Generating a random combination instead.", 'info')
                white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            else:
                white_balls = generate_with_common_pairs(df, common_pairs, GLOBAL_WHITE_BALL_RANGE, excluded_numbers)
                powerball = random.randint(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1])
        else:
            white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            
        max_attempts_unique = 100
        attempts_unique = 0
        while check_exact_match(white_balls) and attempts_unique < max_attempts_unique:
            if use_common_pairs:
                common_pairs_recheck = find_common_pairs(df, top_n=20)
                if num_range:
                    common_pairs_recheck = filter_common_pairs_by_range(common_pairs_recheck, num_range)
                if common_pairs_recheck:
                    white_balls = generate_with_common_pairs(df, common_pairs_recheck, GLOBAL_WHITE_BALL_RANGE, excluded_numbers)
                else:
                    white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            else:
                random_row = df.sample(1).iloc[0]
                white_balls_base = [int(random_row['Number 1']), int(random_row['Number 2']), int(random_row['Number 3']), int(random_row['Number 4']), int(random_row['Number 5'])]
                powerball_base = int(random_row['Powerball'])
                white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            attempts_unique += 1
        
        if attempts_unique == max_attempts_unique:
            flash("Could not find a unique modified combination after many attempts. Please try again.", 'error')
            return render_template('index.html', last_draw=last_draw.to_dict())

        white_balls = [int(num) for num in white_balls]
        powerball = int(powerball)

        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        return render_template('index.html', 
                            white_balls=white_balls, 
                            powerball=powerball, 
                            last_draw=last_draw.to_dict(), 
                            last_draw_dates=last_draw_dates,
                            generation_type='modified')
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())


@app.route('/generate_group_a_strategy', methods=['POST'])
def generate_group_a_strategy_route():
    if df.empty:
        flash("Cannot generate numbers with Group A strategy: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

    num_from_group_a = int(request.form.get('num_from_group_a'))
    white_ball_min = int(request.form.get('white_ball_min', 1))
    white_ball_max = int(request.form.get('white_ball_max', 69))
    white_ball_range_local = (white_ball_min, white_ball_max)
    powerball_min = int(request.form.get('powerball_min', 1))
    powerball_max = int(request.form.get('powerball_max', 26))
    powerball_range_local = (powerball_min, powerball_max)
    excluded_numbers_local = [int(num.strip()) for num in request.form.get('excluded_numbers', '').split(",") if num.strip().isdigit()] if request.form.get('excluded_numbers') else []
    
    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        white_balls, powerball = generate_from_group_a(df, num_from_group_a, white_ball_range_local, powerball_range_local, excluded_numbers_local)
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='group_a_strategy')

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

        if not (1 <= n1 <= 69 and 1 <= n2 <= 69 and 1 <= n3 <= 69 and 1 <= n4 <= 69 and 1 <= n5 <= 69 and 1 <= pb <= 26):
            flash("White balls must be between 1-69 and Powerball between 1-26.", 'error')
            return redirect(url_for('index'))
        
        submitted_white_balls = sorted([n1, n2, n3, n4, n5])
        if len(set(submitted_white_balls)) != 5:
            flash("White ball numbers must be unique within a single draw.", 'error')
            return redirect(url_for('index'))

        success, message = save_manual_draw_to_db(draw_date, n1, n2, n3, n4, n5, pb)
        if success:
            flash(message, 'info')
            initialize_core_data()
            invalidate_analysis_cache()
        else:
            flash(message, 'error')
    except ValueError:
        flash("Invalid input. Please ensure all numbers and date are correctly entered.", 'error')
    except Exception as e:
        flash(f"An error occurred: {e}", 'error')
    return redirect(url_for('index'))

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
        
        if not (all(1 <= n <= 69 for n in white_balls) and 1 <= powerball <= 26):
            flash("White balls must be between 1-69 and Powerball between 1-26 for saving.", 'error')
            return redirect(url_for('index'))

        success, message = save_generated_numbers_to_db(white_balls, powerball)
        if success:
            flash(message, 'info')
            # No need to invalidate full cache here, just ensure generated history is fresh
        else:
            flash(message, 'error')

    except ValueError:
        flash("Invalid number format for saving generated numbers.", 'error')
    except Exception as e:
        flash(f"An error occurred while saving generated numbers: {e}", 'error')
    return redirect(url_for('index'))

# All other UI-facing GET routes (from base.html navigation, and generated_numbers_history)
@app.route('/frequency_analysis')
def frequency_analysis_route():
    white_ball_freq_list, powerball_freq_list = get_cached_analysis('freq_analysis', frequency_analysis, df)
    return render_template('frequency_analysis.html', 
                           white_ball_freq=white_ball_freq_list, 
                           powerball_freq=powerball_freq_list)

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    hot_numbers_list, cold_numbers_list = get_cached_analysis('hot_cold_numbers', hot_cold_numbers, df, last_draw_date_str_for_cache)
    
    return render_template('hot_cold_numbers.html', 
                           hot_numbers=hot_numbers_list, 
                           cold_numbers=cold_numbers_list)

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    monthly_balls = get_cached_analysis('monthly_balls', monthly_white_ball_analysis, df, last_draw_date_str_for_cache)
    monthly_balls_json = json.dumps(monthly_balls)
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_balls=monthly_balls,
                           monthly_balls_json=monthly_balls_json)

@app.route('/sum_of_main_balls')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot display Sum of Main Balls: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Retrieve target_sum and sort_by from form if it's a POST request
    target_sum_display = None
    selected_sort_by = request.args.get('sort_by', 'date_desc') # Default sort

    if request.method == 'POST':
        target_sum_str = request.form.get('target_sum')
        selected_sort_by = request.form.get('sort_by', 'date_desc') # Get sort_by from form submission
        
        if target_sum_str and target_sum_str.isdigit():
            target_sum = int(target_sum_str)
            target_sum_display = target_sum
            results_df_raw = find_results_by_sum(df, target_sum)

            # Apply sorting logic
            if not results_df_raw.empty:
                # Ensure 'Draw Date_dt' exists for sorting by date
                if 'Draw Date_dt' not in results_df_raw.columns:
                    results_df_raw['Draw Date_dt'] = pd.to_datetime(results_df_raw['Draw Date'], errors='coerce')

                if selected_sort_by == 'date_desc':
                    results_df_raw = results_df_raw.sort_values(by='Draw Date_dt', ascending=False)
                elif selected_sort_by == 'date_asc':
                    results_df_raw = results_df_raw.sort_values(by='Draw Date_dt', ascending=True)
                elif selected_sort_by == 'balls_asc':
                    # Create a tuple for sorting white balls
                    results_df_raw['WhiteBallsTuple'] = results_df_raw.apply(
                        lambda row: tuple(sorted([
                            int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
                            int(row['Number 4']), int(row['Number 5'])
                        ])), axis=1
                    )
                    results_df_raw = results_df_raw.sort_values(by='WhiteBallsTuple', ascending=True)
                    results_df_raw = results_df_raw.drop(columns=['WhiteBallsTuple'])
                elif selected_sort_by == 'balls_desc':
                    results_df_raw['WhiteBallsTuple'] = results_df_raw.apply(
                        lambda row: tuple(sorted([
                            int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
                            int(row['Number 4']), int(row['Number 5'])
                        ])), axis=1
                    )
                    results_df_raw = results_df_raw.sort_values(by='WhiteBallsTuple', ascending=False)
                    results_df_raw = results_df_raw.drop(columns=['WhiteBallsTuple'])
            
            results = results_df_raw.to_dict('records')
        else:
            flash("Please enter a valid number for Target Sum.", 'error')
            results = [] # Clear results if input is invalid
            target_sum_display = None # Reset display value
    else: # GET request, no search yet or initial page load
        results = []
        target_sum_display = None # Initial state: no target sum
        selected_sort_by = 'date_desc' # Keep default sort for initial page load
    
    # Pass all necessary data to the template
    return render_template('find_results_by_sum.html', 
                           results=results,
                           target_sum=target_sum_display,
                           selected_sort_by=selected_sort_by)

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
            simulated_freq = simulate_multiple_draws(df, group_a, "Any", "No Combo", GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers, num_draws)
            simulated_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in simulated_freq.items()]
        else:
            flash("Please enter a valid number for Number of Simulations.", 'error')

    return render_template('simulate_multiple_draws.html', 
                           simulated_freq=simulated_freq_list, 
                           num_simulations=num_draws_display)

}

{
type: uploaded file
fileName: number_age_distribution.html
fullText:
{% extends "base.html" %}

{% block title %}Number Age Distribution{% endblock %}

{% block page_heading %}Draws Since Last Appearance (Number Age Distribution){% endblock %}

{% block content %}
<div class="chart-container">
    <h2 class="text-xl font-semibold mb-4">Age Distribution of All Balls</h2>
    {% if number_age_data %}
        <div id="numberAgeChart"></div>
    {% else %}
        <p class="text-gray-500">No number age distribution data available to display chart.</p>
    {% endif %}
</div>

<div class="card mt-8">
    <h3 class="text-xl font-semibold mb-4 text-gray-700">Detailed Ball Ages</h3>
    
    {% if detailed_number_ages %}
    <div class="mb-4">
        <button id="sortByAgeBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-200 ease-in-out">
            Sort by Age (Longest First)
        </button>
    </div>
    <div class="overflow-x-auto shadow-md rounded-lg">
        <table class="min-w-full divide-y divide-gray-200" id="detailedAgesTable">
            <thead class="bg-gray-50">
                <tr>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Number</th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer" data-sort="age">Age (Draws Missed) <span class="ml-1 text-gray-400">↑↓</span></th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
                {# Initial data will be sorted by number by default #}
                {% for ball in detailed_number_ages %}
                <tr class="hover:bg-gray-50">
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{{ ball['number'] }}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{ ball['type'] }}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{ ball['age'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% else %}
    <p class="text-gray-500">No detailed number age data available.</p>
    {% endif %}
</div>
{% endblock %}

{% block scripts %}
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var numberAgeData = [];
        {% if number_age_data %}
            try {
                numberAgeData = JSON.parse('{{ number_age_data | tojson | safe }}');
            } catch (e) {
                console.error("Error parsing numberAgeData JSON for chart:", e);
                numberAgeData = [];
            }
        {% endif %}
        console.log("[D3 Debug] number_age_distribution.html - numberAgeData (for chart):", numberAgeData);

        // Sort data by age for consistent chart display
        numberAgeData.sort((a, b) => a.age - b.age);

        drawBarChart(numberAgeData, "numberAgeChart", "age", "count", "Draws Since Last Appearance", "Draws Missed", "Number of Balls");

        var detailedNumberAges = [];
        {% if detailed_number_ages %}
            try {
                // Ensure the data passed is parsed correctly. Flask's tojson handles lists of dicts well.
                detailedNumberAges = {{ detailed_number_ages | tojson | safe }};
            } catch (e) {
                console.error("Error parsing detailedNumberAges JSON for table:", e);
                detailedNumberAges = [];
            }
        {% endif %}
        console.log("[D3 Debug] number_age_distribution.html - detailedNumberAges (for table):", detailedNumberAges);

        const tableBody = document.querySelector('#detailedAgesTable tbody');
        const sortByAgeBtn = document.getElementById('sortByAgeBtn');
        let isSortedAscending = false; // Tracks current sort order for the button

        function renderTable(data) {
            tableBody.innerHTML = ''; // Clear existing rows
            data.forEach(ball => {
                const row = tableBody.insertRow();
                row.className = 'hover:bg-gray-50';
                row.insertCell().textContent = ball.number;
                row.insertCell().textContent = ball.type;
                row.insertCell().textContent = ball.age;
            });
        }

        // Initial render (default sort by number, from Python side)
        // detailedNumberAges is already sorted by number (1-69, then 1-26) due to Python loop order.
        // If you want to explicitly sort by number in JS:
        // detailedNumberAges.sort((a, b) => a.number - b.number);
        renderTable(detailedNumberAges);

        sortByAgeBtn.addEventListener('click', function() {
            if (isSortedAscending) {
                // Currently sorted ascending by age, click makes it sort descending
                detailedNumberAges.sort((a, b) => b.age - a.age); // Sort descending
                sortByAgeBtn.textContent = 'Sort by Age (Shortest First)';
                isSortedAscending = false;
            } else {
                // Currently unsorted or sorted descending, click makes it sort ascending
                detailedNumberAges.sort((a, b) => a.age - b.age); // Sort ascending
                sortByAgeBtn.textContent = 'Sort by Age (Longest First)';
                isSortedAscending = true;
            }
            renderTable(detailedNumberAges);
        });
    });
</script>
{% endblock %}
Thank you again for providing the `api/index.py` and `templates/number_age_distribution.html` files. Having these current versions ensures I can make the precise changes you need.

I have thoroughly reviewed both files. The `api/index.py` correctly fetches the `last_appearance_date` within `get_number_age_distribution`. The `templates/number_age_distribution.html` currently displays `number`, `type`, and `age`.

Now, I will proceed with the refinement to add the "Last Drawn Date" column to the "Number Age Distribution" table, as discussed.

Here are the updated files:


```python
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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
SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_URL", "[https://yksxzbbcoitehdmsxqex.supabase.co](https://yksxzbbcoitehdmsxqex.supabase.co)")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NzMwNjUsImexFQI6MjA2NTM0OTA2NX0.AzUD7wjR7VbvtUH27NDqJ3AlvFW0nCWpiN9ADG8T_t4")
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

# Set to store all historical white ball combinations for fast lookup
historical_white_ball_sets = set() 

# Cache for precomputed analysis data
analysis_cache = {}
last_analysis_cache_update = datetime.min # Initialize with the earliest possible datetime

# Cache expiration time (e.g., 1 hour for analysis data)
CACHE_EXPIRATION_SECONDS = 3600

# Group A numbers (constants)
group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]
# Global default ranges - used if no specific range is provided by the user
GLOBAL_WHITE_BALL_RANGE = (1, 69)
GLOBAL_POWERBALL_RANGE = (1, 26)
excluded_numbers = []

# Define number ranges for grouped patterns analysis
NUMBER_RANGES = {
    "1-9": (1, 9),
    "10s": (10, 19),
    "20s": (20, 29),
    "30s": (30, 39),
    "40s": (40, 49),
    "50s": (50, 59),
    "60s": (60, 69)
}


# --- Core Utility Functions (All helpers defined here) ---

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
            'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A',
            'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A'] # Ensure 'Numbers' key is present
        }, dtype='object')
    
    last_row = df.iloc[-1].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Manually format 'Numbers' list if not already present
    if 'Numbers' not in last_row or not isinstance(last_row['Numbers'], list):
        last_row['Numbers'] = [
            int(last_row['Number 1']), int(last_row['Number 2']), int(last_row['Number 3']), 
            int(last_row['Number 4']), int(last_row['Number 5'])
        ]
    return last_row

def check_exact_match(white_balls):
    """
    Checks if the given white_balls combination exactly matches any historical draw.
    Uses the precomputed global historical_white_ball_sets for efficient lookup.
    """
    global historical_white_ball_sets
    return frozenset(white_balls) in historical_white_ball_sets

def generate_powerball_numbers(df_source, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None):
    if df_source.empty:
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

        last_draw_data = get_last_draw(df_source)
        if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
            last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
            if set(white_balls) == set(last_white_balls) and powerball == int(last_draw_data['Powerball']):
                attempts += 1
                continue

        if check_exact_match(white_balls): 
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

def generate_from_group_a(df_source, num_from_group_a, white_ball_range, powerball_range, excluded_numbers):
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 1000
    attempts = 0
    
    valid_group_a = [num for num in group_a if white_ball_range[0] <= num <= white_ball_range[1] and num not in excluded_numbers]
    
    remaining_pool = [num for num in range(white_ball_range[0], white_ball_range[1] + 1)
                      if num not in valid_group_a and num not in excluded_numbers]

    if len(valid_group_a) < num_from_group_a:
        raise ValueError(f"Not enough unique numbers in Group A ({len(valid_group_a)}) to pick {num_from_group_a}.")
    
    num_from_remaining = 5 - num_from_group_a
    if len(remaining_pool) < num_from_remaining:
        raise ValueError(f"Not enough unique numbers in the remaining pool ({len(remaining_pool)}) to pick {num_from_remaining}.")

    while attempts < max_attempts:
        try:
            selected_from_group_a = random.sample(valid_group_a, num_from_group_a)
            
            # Ensure numbers picked from remaining pool are not already in selected_from_group_a
            available_for_remaining = [num for num in remaining_pool if num not in selected_from_group_a]
            if len(available_for_remaining) < num_from_remaining:
                attempts += 1
                continue # Retry if not enough unique numbers for remaining

            selected_from_remaining = random.sample(available_for_remaining, num_from_remaining) 
            
            white_balls = sorted(selected_from_group_a + selected_from_remaining)
            
            powerball = random.randint(powerball_range[0], powerball_range[1])

            if check_exact_match(white_balls): 
                attempts += 1
                continue

            break
        except ValueError as e:
            print(f"Attempt failed during group_a strategy: {e}. Retrying...")
            attempts += 1
            continue
    else:
        raise ValueError("Could not generate a unique combination with Group A strategy after many attempts.")

    return white_balls, powerball


def check_historical_match(df_source, white_balls, powerball):
    if df_source.empty: return None
    for _, row in df_source.iterrows():
        historical_white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        historical_powerball = int(row['Powerball'])
        if set(white_balls) == set(historical_white_balls) and powerball == historical_powerball:
            return row['Draw Date']
    return None

def frequency_analysis(df_source):
    if df_source.empty: 
        return [], []
    white_balls = df_source[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)
    powerball_freq = df_source['Powerball'].astype(int).value_counts().reindex(range(1, 27), fill_value=0)
    
    white_ball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.items()]
    powerball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in powerball_freq.items()]

    return white_ball_freq_list, powerball_freq_list

def hot_cold_numbers(df_source, last_draw_date_str):
    if df_source.empty or last_draw_date_str == 'N/A': 
        return [], []
    
    last_draw_date = pd.to_datetime(last_draw_date_str)
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    
    recent_data = df_source[df_source['Draw Date_dt'] >= one_year_ago].copy()
    if recent_data.empty: 
        return [], []

    white_balls = recent_data[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts()

    hot_numbers = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.nlargest(14).sort_values(ascending=False).items()]
    cold_numbers = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.nsmallest(14).sort_values(ascending=True).items()]

    return hot_numbers, cold_numbers

def monthly_white_ball_analysis(df_source, last_draw_date_str):
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
        if 'Month' not in recent_data.columns:
            recent_data['Month'] = recent_data['Draw Date_dt'].dt.to_period('M')
            print(f"[DEBUG-Monthly] 'Month' column added to recent_data. First 2 months: {recent_data['Month'].head(2).tolist()}")
        
        required_cols = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
        for col in required_cols:
            if col in recent_data.columns:
                recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
            else:
                print(f"[ERROR-Monthly] Missing required column '{col}' in recent_data. Cannot perform analysis.")
                return {}

        recent_data = recent_data.dropna(subset=required_cols)
        if recent_data.empty:
            print("[DEBUG-Monthly] recent_data is empty after dropping NaN in ball columns. Returning empty dict.")
            return {}

        monthly_balls_raw = recent_data.groupby('Month')[required_cols].apply(
            lambda x: sorted([int(num) for num in x.values.flatten() if not pd.isna(num)])
        ).to_dict()

        monthly_balls_str_keys = {}
        for period_key, ball_list in monthly_balls_raw.items():
            monthly_balls_str_keys[str(period_key)] = [int(ball) for ball in ball_list]
        
        print(f"[DEBUG-Monthly] Groupby and apply successful. First item in monthly_balls_str_keys: {next(iter(monthly_balls_str_keys.items())) if monthly_balls_str_keys else 'N/A'}")

    except Exception as e:
        print(f"[ERROR-Monthly] Error during groupby/apply operation or conversion: {e}. Returning empty dict.")
        import traceback
        traceback.print_exc()
        return {}
    
    print("[DEBUG-Monthly] Successfully computed monthly_balls_str_keys.")
    return monthly_balls_str_keys


def sum_of_main_balls(df_source):
    """Calculates the sum of the five main white balls for each draw."""
    if df_source.empty:
        return pd.DataFrame(), [], 0, 0, 0.0
    
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)
    
    temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    sum_freq = temp_df['Sum'].value_counts().sort_index()
    sum_freq_list = [{'sum': int(s), 'count': int(c)} for s, c in sum_freq.items()]

    min_sum = int(temp_df['Sum'].min()) if not temp_df['Sum'].empty else 0
    max_sum = int(temp_df['Sum'].max()) if not temp_df['Sum'].empty else 0
    avg_sum = round(temp_df['Sum'].mean(), 2) if not temp_df['Sum'].empty else 0.0

    return temp_df[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum']], sum_freq_list, min_sum, max_sum, avg_sum

def find_results_by_sum(df_source, target_sum):
    if df_source.empty: return pd.DataFrame()
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']: # Include Powerball here
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)

    if 'Sum' not in temp_df.columns:
        temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    results = temp_df[temp_df['Sum'] == target_sum]
    # Return all necessary columns for rendering, including Powerball
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum']]

def simulate_multiple_draws(df_source, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
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

def winning_probability(white_ball_range_tuple, powerball_range_tuple):
    total_white_balls_in_range = white_ball_range_tuple[1] - white_ball_range_tuple[0] + 1
    white_ball_combinations = calculate_combinations(total_white_balls_in_range, 5)

    total_powerballs_in_range = powerball_range_tuple[1] - powerball_range_tuple[0] + 1

    total_combinations = white_ball_combinations * total_powerballs_in_range

    probability_1_in_x = f"1 in {total_combinations:,}" if total_combinations > 0 else "N/A"
    probability_percentage = f"{1 / total_combinations * 100:.10f}%" if total_combinations > 0 else "N/A"

    return probability_1_in_x, probability_percentage

def partial_match_probabilities(white_ball_range_tuple, powerball_range_tuple):
    total_white_balls_in_range = white_ball_range_tuple[1] - white_ball_range_tuple[0] + 1
    total_powerballs_in_range = powerball_range_tuple[1] - powerball_range_tuple[0] + 1

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
        comb_matched_w = calculate_combinations(5, data["matched_w"])
        comb_unmatched_w = calculate_combinations(total_white_balls_in_range - 5, data["unmatched_w"])

        if data["matched_p"] == 1:
            comb_p = 1
        else:
            comb_p = total_powerballs_in_range - 1
            if comb_p < 0:
                comb_p = 0
        
        numerator = comb_matched_w * comb_unmatched_w * comb_p
        
        if numerator == 0:
            probabilities[scenario] = "N/A"
        else:
            total_possible_combinations_for_draw = calculate_combinations(total_white_balls_in_range, 5) * total_powerballs_in_range
            
            probability = total_possible_combinations_for_draw / numerator
            probabilities[scenario] = f"{probability:,.0f} to 1"

    return probabilities


def export_analysis_results(df_source, file_path="analysis_results.csv"):
    df_source.to_csv(file_path, index=False)
    print(f"Analysis results saved to {file_path}")

def find_last_draw_dates_for_numbers(df_source, white_balls, powerball):
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

def modify_combination(df_source, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers):
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

def find_common_pairs(df_source, top_n=10):
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

def generate_with_common_pairs(df_source, common_pairs, white_ball_range, excluded_numbers):
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

def get_number_age_distribution(df_source):
    if df_source.empty: return [], []
    df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'])
    all_draw_dates = sorted(df_source['Draw Date_dt'].drop_duplicates().tolist())
    
    detailed_ages = []
    
    for i in range(1, 70):
        last_appearance_date = None
        last_appearance_date_str = "N/A" # Default to N/A
        temp_df_filtered = df_source[(df_source['Number 1'].astype(int) == i) | (df_source['Number 2'].astype(int) == i) |
                              (df_source['Number 3'].astype(int) == i) | (df_source['Number 4'].astype(int) == i) |
                              (df_source['Number 5'].astype(int) == i)]
        
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()
            last_appearance_date_str = last_appearance_date.strftime('%Y-%m-%d')

        miss_streak_count = 0
        if last_appearance_date is not None:
            # We need to correctly count missed draws based on *all* unique draw dates
            # subsequent to the last appearance date.
            draw_dates_after_last_appearance = [d for d in all_draw_dates if d > last_appearance_date]
            miss_streak_count = len(draw_dates_after_last_appearance)

            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': miss_streak_count, 'last_drawn_date': last_appearance_date_str})
        else:
            # If never drawn, age is the total number of draws
            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': len(all_draw_dates), 'last_drawn_date': last_appearance_date_str})

    for i in range(1, 27):
        last_appearance_date = None
        last_appearance_date_str = "N/A" # Default to N/A
        temp_df_filtered = df_source[df_source['Powerball'].astype(int) == i]
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()
            last_appearance_date_str = last_appearance_date.strftime('%Y-%m-%d')

        miss_streak_count = 0
        if last_appearance_date is not None:
            draw_dates_after_last_appearance = [d for d in all_draw_dates if d > last_appearance_date]
            miss_streak_count = len(draw_dates_after_last_appearance)

            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': miss_streak_count, 'last_drawn_date': last_appearance_date_str})
        else:
            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': len(all_draw_dates), 'last_drawn_date': last_appearance_date_str})

    all_miss_streaks_only = [item['age'] for item in detailed_ages]
    age_counts = pd.Series(all_miss_streaks_only).value_counts().sort_index()
    age_counts_list = [{'age': int(age), 'count': int(count)} for age, count in age_counts.items()]
    
    return age_counts_list, detailed_ages

def get_co_occurrence_matrix(df_source):
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

def get_powerball_position_frequency(df_source):
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

def _find_consecutive_pairs(numbers_list):
    """Identifies and returns all consecutive pairs in a sorted list of numbers."""
    pairs = []
    sorted_nums = sorted(numbers_list)
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i] + 1 == sorted_nums[i+1]:
            pairs.append([sorted_nums[i], sorted_nums[i+1]])
    return pairs

def get_consecutive_numbers_trends(df_source, last_draw_date_str):
    print("[DEBUG-ConsecutiveTrends] Inside get_consecutive_numbers_trends function.")
    if df_source.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-ConsecutiveTrends] df_source is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-ConsecutiveTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-ConsecutiveTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return {}

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-ConsecutiveTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-ConsecutiveTrends] 'Draw Date_dt' column missing or not datetime type in df_source. Returning empty list.")
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt', ascending=False)
    if recent_data.empty:
        print("[DEBUG-ConsecutiveTrends] recent_data is empty after filtering. Returning empty list.")
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        consecutive_pairs = _find_consecutive_pairs(white_balls)
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            'consecutive_present': "Yes" if consecutive_pairs else "No",
            'consecutive_pairs': consecutive_pairs
        })
    
    print(f"[DEBUG-ConsecutiveTrends] Generated {len(trend_data)} trend data points.")
    return trend_data

def get_most_frequent_triplets(df_source, top_n=10):
    print("[DEBUG-Triplets] Inside get_most_frequent_triplets function.")
    if df_source.empty:
        print("[DEBUG-Triplets] df_source is empty. Returning empty list.")
        return []

    triplet_counts = defaultdict(int)

    for idx, row in df_source.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        for triplet_combo in combinations(sorted(white_balls), 3):
            triplet_counts[triplet_combo] += 1
    
    sorted_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)
    
    formatted_triplets = []
    for triplet, count in sorted_triplets[:top_n]:
        formatted_triplets.append({
            'triplet': list(triplet),
            'count': int(count)
        })
    
    print(f"[DEBUG-Triplets] Found {len(triplet_counts)} unique triplets. Returning top {len(formatted_triplets)}.")
    return formatted_triplets


def get_odd_even_split_trends(df_source, last_draw_date_str):
    print("[DEBUG-OddEvenTrends] Inside get_odd_even_split_trends function.")
    if df_source.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-OddEvenTrends] df_source is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-OddEvenTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-OddEvenTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return {}

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-OddEvenTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-OddEvenTrends] 'Draw Date_dt' column missing or not datetime type in df_source. Returning empty list.")
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt', ascending=False)
    if recent_data.empty:
        print("[DEBUG-OddEvenTrends] recent_data is empty after filtering. Returning empty list.")
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        # Calculate WB_Sum
        wb_sum = sum(white_balls)

        # Identify group_a numbers present in the draw
        group_a_numbers_present = sorted([num for num in white_balls if num in group_a])

        even_count = sum(1 for num in white_balls if num % 2 == 0)
        odd_count = 5 - even_count

        split_category = "Other"

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
            'split_category': split_category,
            'wb_sum': wb_sum,
            'group_a_numbers': group_a_numbers_present
        })
    
    print(f"[DEBUG-OddEvenTrends] Generated {len(trend_data)} trend data points with WB_Sum and Group A numbers.")
    return trend_data

def get_powerball_frequency_by_year(df_source, num_years=5):
    """
    Calculates the frequency of each Powerball number per year for the last `num_years`.
    Returns a list of dictionaries, where each dict represents a Powerball number
    and its count for each of the last `num_years`.
    """
    print(f"[DEBUG-YearlyPB] Inside get_powerball_frequency_by_year for last {num_years} years.")
    if df_source.empty:
        print("[DEBUG-YearlyPB] df_source is empty. Returning empty data.")
        return [], []

    current_year = datetime.now().year
    
    years = [y for y in y in range(current_year - num_years + 1, current_year + 1)]
    print(f"[DEBUG-YearlyPB] Years to analyze: {years}")

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        print("[ERROR-YearlyPB] 'Draw Date_dt' column missing or not datetime type. Attempting to re-create.")
        df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'], errors='coerce')
        df_source = df_source.dropna(subset=['Draw Date_dt'])
        if df_source.empty:
            print("[ERROR-YearlyPB] Re-creation failed or resulted in empty df. Returning empty data.")
            return [], []

    recent_data = df_source[df_source['Draw Date_dt'].dt.year.isin(years)].copy()
    
    if recent_data.empty:
        print("[DEBUG-YearlyPB] recent_data is empty after filtering by years. Returning empty data.")
        return [], years

    recent_data['Year'] = recent_data['Draw Date_dt'].dt.year

    recent_data['Powerball'] = pd.to_numeric(recent_data['Powerball'], errors='coerce').fillna(0).astype(int)
    
    yearly_pb_freq_pivot = pd.pivot_table(
        recent_data,
        index='Powerball',
        columns='Year',
        values='Draw Date',
        aggfunc='count',
        fill_value=0
    )
    
    all_powerballs = pd.Series(range(1, 27))
    yearly_pb_freq_pivot = yearly_pb_freq_pivot.reindex(all_powerballs, fill_value=0)

    yearly_pb_freq_pivot = yearly_pb_freq_pivot.reindex(columns=years, fill_value=0)
    
    formatted_data = []
    for powerball_num, row in yearly_pb_freq_pivot.iterrows():
        row_dict = {'Powerball': int(powerball_num)}
        for year in years:
            row_dict[f'Year_{year}'] = int(row[year])
        formatted_data.append(row_dict)
    
    formatted_data = sorted(formatted_data, key=lambda x: x['Powerball'])

    print(f"[DEBUG-YearlyPB] Successfully computed yearly Powerball frequencies. First 3: {formatted_data[:3]}")
    return formatted_data, years

def _get_generated_picks_for_date_from_db(date_str):
    """
    Fetches generated numbers for a specific date from the database.
    Returns a list of dicts: [{'white_balls': [...], 'powerball': int}].
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{GENERATED_NUMBERS_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=False)
    
    try:
        start_of_day_dt = datetime.strptime(date_str, '%Y-%m-%d')
        end_of_day_dt = start_of_day_dt + timedelta(days=1)
        start_of_day_iso = start_of_day_dt.isoformat(timespec='seconds') + "Z"
        end_of_day_iso = end_of_day_dt.isoformat(timespec='seconds') + "Z"
    except ValueError:
        print(f"Invalid date format for generated_date_str: {date_str}")
        return []

    params = {
        'select': 'generated_date,number_1,number_2,number_3,number_4,number_5,powerball',
        'generated_date': f'gte.{start_of_day_iso}',
        'and': (f'(generated_date.lt.{end_of_day_iso})',) 
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        raw_data = response.json()
        
        formatted_picks = []
        for record in raw_data:
            white_balls = sorted([
                int(record['number_1']), int(record['number_2']), int(record['number_3']),
                int(record['number_4']), int(record['number_5'])
            ])
            formatted_picks.append({
                'white_balls': white_balls,
                'powerball': int(record['powerball'])
            })
        return formatted_picks
    except requests.exceptions.RequestException as e:
        print(f"Error fetching generated picks for date {date_str}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in _get_generated_picks_for_date_from_db: {e}")
        return []

def _get_official_draw_for_date_from_db(date_str):
    """
    Fetches a single official draw for a specific date from the database.
    Returns a dict: {'Draw Date': ..., 'Number 1': ..., 'Powerball': ...} or None.
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=False)
    
    params = {
        'select': 'Draw Date,Number 1,Number 2,Number 3,Number 4,Number 5,Powerball',
        'Draw Date': f'eq.{date_str}'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        raw_data = response.json()
        if raw_data:
            return raw_data[0] # Should only be one draw for a specific date
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching official draw for date {date_str}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in _get_official_draw_for_date_from_db: {e}")
        return None

def analyze_generated_batch_against_official_draw(generated_picks_list, official_draw):
    """
    Analyzes a batch of generated picks against a single official draw result.
    Returns a summary of matches.
    """
    summary = {
        "Match 5 White Balls + Powerball": {"count": 0},
        "Match 5 White Balls Only": {"count": 0},
        "Match 4 White Balls + Powerball": {"count": 0},
        "Match 4 White Balls Only": {"count": 0},
        "Match 3 White Balls + Powerball": {"count": 0},
        "Match 3 White Balls Only": {"count": 0},
        "Match 2 White Balls + Powerball": {"count": 0},
        "Match 1 White Ball + Powerball": {"count": 0},
        "Match Powerball Only": {"count": 0},
        "No Match": {"count": 0}
    }
    
    if not official_draw:
        return summary # Cannot analyze without an official draw

    official_white_balls = sorted([
        int(official_draw['Number 1']), int(official_draw['Number 2']), int(official_draw['Number 3']),
        int(official_draw['Number 4']), int(official_draw['Number 5'])
    ])
    official_powerball = int(official_draw['Powerball'])
    official_white_set = set(official_white_balls)

    for pick in generated_picks_list:
        generated_white_balls = sorted(pick['white_balls'])
        generated_powerball = pick['powerball']
        generated_white_set = set(generated_white_balls)

        white_matches = len(generated_white_set.intersection(official_white_set))
        powerball_match = 1 if generated_powerball == official_powerball else 0

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
        
        summary[category]["count"] += 1
    
    return summary

def save_manual_draw_to_db(draw_date, n1, n2, n3, n4, n5, pb):
    """
    Saves a manually entered official Powerball draw to the 'powerball_draws' table.
    Checks for existence of the draw date to prevent duplicates.
    """
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=True)

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
    headers = _get_supabase_headers(is_service_key=True)

    sorted_numbers = sorted(numbers)

    check_params = {
        'select': 'id',
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

    new_generated_data = {
        'number_1': sorted_numbers[0],
        'number_2': sorted_numbers[1],
        'number_3': sorted_numbers[2],
        'number_4': sorted_numbers[3],
        'number_5': sorted_numbers[4],
        'powerball': powerball,
        'generated_date': datetime.now().isoformat()
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
        headers = _get_supabase_headers(is_service_key=False)
        
        while True:
            params = {
                'select': 'generated_date,number_1,number_2,number_3,number_4,number_5,powerball',
                'order': 'generated_date.desc',
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

        grouped_data = defaultdict(list)
        for record in all_data:
            gen_dt = datetime.fromisoformat(record['generated_date'].replace('Z', '+00:00'))
            date_key = gen_dt.strftime('%Y-%m-%d')
            
            formatted_time = gen_dt.strftime('%I:%M %p')

            generated_balls = sorted([
                int(record['number_1']), int(record['number_2']), int(record['number_3']),
                int(record['number_4']), int(record['number_5'])
            ])
            
            grouped_data[date_key].append({
                'time': formatted_time,
                'white_balls': generated_balls,
                'powerball': int(record['powerball'])
            })
        
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

    two_years_ago = datetime.now() - timedelta(days=2 * 365)
    recent_historical_data = df_historical[df_historical['Draw Date_dt'] >= two_years_ago].copy()

    if recent_historical_data.empty:
        return results

    gen_white_set = set(generated_white_balls)

    for index, row in recent_historical_data.iterrows():
        historical_white_balls = sorted([
            int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
            int(row['Number 4']), int(row['Number 5'])
        ])
        historical_powerball = int(row['Powerball'])
        historical_draw_date = row['Draw Date']

        hist_white_set = set(historical_white_balls)

        white_matches = len(gen_white_set.intersection(hist_white_set))

        powerball_match = 1 if generated_powerball == historical_powerball else 0

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
        
        # Corrected lines: access 'summary' through the 'results' dictionary
        results["summary"][category]["count"] += 1
        results["summary"][category]["draws"].append({
            "date": historical_draw_date,
            "white_balls": historical_white_balls,
            "powerball": historical_powerball
        })
    
    for category in results["summary"]:
        results["summary"][category]["draws"].sort(key=lambda x: x['date'], reverse=True)

    return results


def get_grouped_patterns_over_years(df_source):
    if df_source.empty:
        print("[DEBUG-GroupedPatterns] df_source is empty. Returning empty list.")
        return []

    df_source_copy = df_source.copy()
    if 'Draw Date_dt' not in df_source_copy.columns:
        df_source_copy['Draw Date_dt'] = pd.to_datetime(df_source_copy['Draw Date'], errors='coerce')
    df_source_copy = df_source_copy.dropna(subset=['Draw Date_dt'])
    
    if df_source_copy.empty:
        print("[DEBUG-GroupedPatterns] df_source_copy is empty after datetime conversion. Returning empty list.")
        return []

    # Ensure white ball columns are numeric
    for i in range(1, 6):
        col = f'Number {i}'
        if col in df_source_copy.columns:
            df_source_copy[col] = pd.to_numeric(df_source_copy[col], errors='coerce').fillna(0).astype(int)
        else:
            print(f"[WARN-GroupedPatterns] Column '{col}' not found in DataFrame for pattern analysis.")


    all_patterns_data = []
    
    # Iterate through unique years
    for year in sorted(df_source_copy['Draw Date_dt'].dt.year.unique()):
        yearly_df = df_source_copy[df_source_copy['Draw Date_dt'].dt.year == year]
        
        # Store counts for pairs and triplets for the current year
        year_pairs_counts = defaultdict(int)
        year_triplets_counts = defaultdict(int)

        for _, row in yearly_df.iterrows():
            # Get white balls, ensuring they are integers and handling potential NaNs after conversion
            white_balls = [int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])]
            
            for range_name, (min_val, max_val) in NUMBER_RANGES.items():
                numbers_in_current_range = sorted([num for num in white_balls if min_val <= num <= max_val])
                
                if len(numbers_in_current_range) >= 2:
                    for pair in combinations(numbers_in_current_range, 2):
                        year_pairs_counts[(range_name, tuple(sorted(pair)))] += 1
                
                if len(numbers_in_current_range) >= 3:
                    for triplet in combinations(numbers_in_current_range, 3):
                        year_triplets_counts[(range_name, tuple(sorted(triplet)))] += 1
        
        # Add to main results list
        for (range_name, pattern), count in year_pairs_counts.items():
            all_patterns_data.append({
                "year": int(year),
                "range": range_name,
                "type": "Pair",
                "pattern": list(pattern),
                "count": int(count)
            })
        
        for (range_name, pattern), count in year_triplets_counts.items():
            all_patterns_data.append({
                "year": int(year),
                "range": range_name,
                "type": "Triplet",
                "pattern": list(pattern),
                "count": int(count)
            })

    # Sort by count descending, then by year, then by range, then by pattern
    # Convert pattern lists to strings for consistent sorting as a secondary key if counts are equal
    all_patterns_data.sort(key=lambda x: (x['count'], x['year'], x['range'], str(x['pattern'])), reverse=True)
    
    print(f"[DEBUG-GroupedPatterns] Generated {len(all_patterns_data)} grouped patterns data points.")
    return all_patterns_data

# --- Data Initialization (Call after all helper functions are defined) ---
def initialize_core_data():
    global df, last_draw, historical_white_ball_sets
    print("Attempting to load core historical data...")
    try:
        df_temp = load_historical_data_from_supabase()
        if not df_temp.empty:
            df = df_temp
            last_draw = get_last_draw(df)
            # Ensure last_draw has 'Numbers' formatted for display
            if not last_draw.empty and last_draw.get('Draw Date') != 'N/A':
                # No need to convert to_datetime here, just strftime if it's not 'N/A'
                last_draw['Draw Date'] = last_draw['Draw Date_dt'].strftime('%Y-%m-%d')
                # The get_last_draw already sets 'Numbers'
            print("Core historical data loaded successfully.")
        else:
            print("Core historical data is empty after loading. df remains empty.")
            # Set default N/A for last_draw if no data is found
            last_draw = pd.Series({
                'Draw Date': 'N/A', 'Number 1': 'N/A', 'Number 2': 'N/A',
                'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A',
                'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A']
            }, dtype='object')
    except Exception as e:
        print(f"An error occurred during initial core data loading: {e}")
        import traceback
        traceback.print_exc()

initialize_core_data()


# --- Analysis Cache Management ---
def get_cached_analysis(key, compute_function, *args, **kwargs):
    global analysis_cache, last_analysis_cache_update
    
    if key in analysis_cache and (datetime.now() - last_analysis_cache_update).total_seconds() < CACHE_EXPIRATION_SECONDS:
        print(f"Serving '{key}' from cache.")
        return analysis_cache[key]
    
    print(f"Computing and caching '{key}'.")
    computed_data = compute_function(*args, **kwargs)
    
    analysis_cache[key] = computed_data
    last_analysis_cache_update = datetime.now()
    return computed_data

def invalidate_analysis_cache():
    global analysis_cache, last_analysis_cache_update
    analysis_cache = {}
    last_analysis_cache_update = datetime.min
    print("Analysis cache invalidated.")


# --- Flask Routes (Ordered for Dependency - all UI-facing routes first, then API routes) ---

# Core home page route
@app.route('/')
def index():
    last_draw_dict = last_draw.to_dict()
    return render_template('index.html', last_draw=last_draw_dict)

# Generation Routes (referenced directly in index.html forms)
@app.route('/generate', methods=['POST'])
def generate():
    if df.empty:
        flash("Cannot generate numbers: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

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
        return render_template('index.html', last_draw=last_draw.to_dict())

    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='generated')

@app.route('/generate_modified', methods=['POST'])
def generate_modified():
    if df.empty:
        flash("Cannot generate modified combination: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

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
            return render_template('index.html', last_draw=last_draw.to_dict())


        if use_common_pairs:
            common_pairs = find_common_pairs(df, top_n=20)
            if num_range:
                common_pairs = filter_common_pairs_by_range(common_pairs, num_range)
            
            if not common_pairs:
                flash("No common pairs found with the specified filter. Generating a random combination instead.", 'info')
                white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            else:
                white_balls = generate_with_common_pairs(df, common_pairs, GLOBAL_WHITE_BALL_RANGE, excluded_numbers)
                powerball = random.randint(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1])
        else:
            white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            
        max_attempts_unique = 100
        attempts_unique = 0
        while check_exact_match(white_balls) and attempts_unique < max_attempts_unique:
            if use_common_pairs:
                common_pairs_recheck = find_common_pairs(df, top_n=20)
                if num_range:
                    common_pairs_recheck = filter_common_pairs_by_range(common_pairs_recheck, num_range)
                if common_pairs_recheck:
                    white_balls = generate_with_common_pairs(df, common_pairs_recheck, GLOBAL_WHITE_BALL_RANGE, excluded_numbers)
                else:
                    white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            else:
                random_row = df.sample(1).iloc[0]
                white_balls_base = [int(random_row['Number 1']), int(random_row['Number 2']), int(random_row['Number 3']), int(random_row['Number 4']), int(random_row['Number 5'])]
                powerball_base = int(random_row['Powerball'])
                white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers)
            attempts_unique += 1
        
        if attempts_unique == max_attempts_unique:
            flash("Could not find a unique modified combination after many attempts. Please try again.", 'error')
            return render_template('index.html', last_draw=last_draw.to_dict())

        white_balls = [int(num) for num in white_balls]
        powerball = int(powerball)

        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        return render_template('index.html', 
                            white_balls=white_balls, 
                            powerball=powerball, 
                            last_draw=last_draw.to_dict(), 
                            last_draw_dates=last_draw_dates,
                            generation_type='modified')
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())


@app.route('/generate_group_a_strategy', methods=['POST'])
def generate_group_a_strategy_route():
    if df.empty:
        flash("Cannot generate numbers with Group A strategy: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

    num_from_group_a = int(request.form.get('num_from_group_a'))
    white_ball_min = int(request.form.get('white_ball_min', 1))
    white_ball_max = int(request.form.get('white_ball_max', 69))
    white_ball_range_local = (white_ball_min, white_ball_max)
    powerball_min = int(request.form.get('powerball_min', 1))
    powerball_max = int(request.form.get('powerball_max', 26))
    powerball_range_local = (powerball_min, powerball_max)
    excluded_numbers_local = [int(num.strip()) for num in request.form.get('excluded_numbers', '').split(",") if num.strip().isdigit()] if request.form.get('excluded_numbers') else []
    
    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        white_balls, powerball = generate_from_group_a(df, num_from_group_a, white_ball_range_local, powerball_range_local, excluded_numbers_local)
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict())

    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='group_a_strategy')

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

        if not (1 <= n1 <= 69 and 1 <= n2 <= 69 and 1 <= n3 <= 69 and 1 <= n4 <= 69 and 1 <= n5 <= 69 and 1 <= pb <= 26):
            flash("White balls must be between 1-69 and Powerball between 1-26.", 'error')
            return redirect(url_for('index'))
        
        submitted_white_balls = sorted([n1, n2, n3, n4, n5])
        if len(set(submitted_white_balls)) != 5:
            flash("White ball numbers must be unique within a single draw.", 'error')
            return redirect(url_for('index'))

        success, message = save_manual_draw_to_db(draw_date, n1, n2, n3, n4, n5, pb)
        if success:
            flash(message, 'info')
            initialize_core_data()
            invalidate_analysis_cache()
        else:
            flash(message, 'error')
    except ValueError:
        flash("Invalid input. Please ensure all numbers and date are correctly entered.", 'error')
    except Exception as e:
        flash(f"An error occurred: {e}", 'error')
    return redirect(url_for('index'))

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
        
        if not (all(1 <= n <= 69 for n in white_balls) and 1 <= powerball <= 26):
            flash("White balls must be between 1-69 and Powerball between 1-26 for saving.", 'error')
            return redirect(url_for('index'))

        success, message = save_generated_numbers_to_db(white_balls, powerball)
        if success:
            flash(message, 'info')
            # No need to invalidate full cache here, just ensure generated history is fresh
        else:
            flash(message, 'error')

    except ValueError:
        flash("Invalid number format for saving generated numbers.", 'error')
    except Exception as e:
        flash(f"An error occurred while saving generated numbers: {e}", 'error')
    return redirect(url_for('index'))

# All other UI-facing GET routes (from base.html navigation, and generated_numbers_history)
@app.route('/frequency_analysis')
def frequency_analysis_route():
    white_ball_freq_list, powerball_freq_list = get_cached_analysis('freq_analysis', frequency_analysis, df)
    return render_template('frequency_analysis.html', 
                           white_ball_freq=white_ball_freq_list, 
                           powerball_freq=powerball_freq_list)

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    hot_numbers_list, cold_numbers_list = get_cached_analysis('hot_cold_numbers', hot_cold_numbers, df, last_draw_date_str_for_cache)
    
    return render_template('hot_cold_numbers.html', 
                           hot_numbers=hot_numbers_list, 
                           cold_numbers=cold_numbers_list)

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    monthly_balls = get_cached_analysis('monthly_balls', monthly_white_ball_analysis, df, last_draw_date_str_for_cache)
    monthly_balls_json = json.dumps(monthly_balls)
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_balls=monthly_balls,
                           monthly_balls_json=monthly_balls_json)

@app.route('/sum_of_main_balls')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot display Sum of Main Balls: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Retrieve target_sum and sort_by from form if it's a POST request
    target_sum_display = None
    selected_sort_by = request.args.get('sort_by', 'date_desc') # Default sort

    if request.method == 'POST':
        target_sum_str = request.form.get('target_sum')
        selected_sort_by = request.form.get('sort_by', 'date_desc') # Get sort_by from form submission
        
        if target_sum_str and target_sum_str.isdigit():
            target_sum = int(target_sum_str)
            target_sum_display = target_sum
            results_df_raw = find_results_by_sum(df, target_sum)

            # Apply sorting logic
            if not results_df_raw.empty:
                # Ensure 'Draw Date_dt' exists for sorting by date
                if 'Draw Date_dt' not in results_df_raw.columns:
                    results_df_raw['Draw Date_dt'] = pd.to_datetime(results_df_raw['Draw Date'], errors='coerce')

                if selected_sort_by == 'date_desc':
                    results_df_raw = results_df_raw.sort_values(by='Draw Date_dt', ascending=False)
                elif selected_sort_by == 'date_asc':
                    results_df_raw = results_df_raw.sort_values(by='Draw Date_dt', ascending=True)
                elif selected_sort_by == 'balls_asc':
                    # Create a tuple for sorting white balls
                    results_df_raw['WhiteBallsTuple'] = results_df_raw.apply(
                        lambda row: tuple(sorted([
                            int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
                            int(row['Number 4']), int(row['Number 5'])
                        ])), axis=1
                    )
                    results_df_raw = results_df_raw.sort_values(by='WhiteBallsTuple', ascending=True)
                    results_df_raw = results_df_raw.drop(columns=['WhiteBallsTuple'])
                elif selected_sort_by == 'balls_desc':
                    results_df_raw['WhiteBallsTuple'] = results_df_raw.apply(
                        lambda row: tuple(sorted([
                            int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
                            int(row['Number 4']), int(row['Number 5'])
                        ])), axis=1
                    )
                    results_df_raw = results_df_raw.sort_values(by='WhiteBallsTuple', ascending=False)
                    results_df_raw = results_df_raw.drop(columns=['WhiteBallsTuple'])
            
            results = results_df_raw.to_dict('records')
        else:
            flash("Please enter a valid number for Target Sum.", 'error')
            results = [] # Clear results if input is invalid
            target_sum_display = None # Reset display value
    else: # GET request, no search yet or initial page load
        results = []
        target_sum_display = None # Initial state: no target sum
        selected_sort_by = 'date_desc' # Keep default sort for initial page load
    
    # Pass all necessary data to the template
    return render_template('find_results_by_sum.html', 
                           results=results,
                           target_sum=target_sum_display,
                           selected_sort_by=selected_sort_by)

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
            simulated_freq = simulate_multiple_draws(df, group_a, "Any", "No Combo", GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, excluded_numbers, num_draws)
            simulated_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in simulated_freq.items()]
        else:
            flash("Please enter a valid number for Number of Simulations.", 'error')

    return render_template('simulate_multiple_draws.html', 
                           simulated_freq=simulated_freq_list, 
                           num_simulations=num_draws_display)

}
