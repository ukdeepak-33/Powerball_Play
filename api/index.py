import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
import random
from itertools import combinations
import math
import os
from collections import defaultdict
from datetime import datetime
import requests
import json
import numpy as np

# --- Flask App Initialization with Template Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = 'supersecretkey'

# --- Supabase Configuration ---
SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NzMwNjUsImV4cCI6MjA2NTM0OTA2NX0.AzUD7wjR7VbvtUH27NDqJ3AlvFW0nCWpiN9ADG8T_t4")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "YOUR_SUPABASE_SERVICE_ROLE_KEY")

SUPABASE_TABLE_NAME = 'powerball_draws'

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

        df = pd.DataFrame(all_data)
        df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'], errors='coerce')
        df = df.dropna(subset=['Draw Date_dt'])

        numeric_cols = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0).astype(int)
            else:
                print(f"Warning: Column '{col}' not found in fetched data. Skipping conversion for this column.")

        df['Draw Date'] = df['Draw Date_dt'].dt.strftime('%Y-%m-%d')
        
        print(f"Successfully loaded and processed {len(df)} records from Supabase.")
        return df

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
        if not last_draw_data.empty and last_draw_data.get('Number 1') != 'N/A':
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

def frequency_analysis(df):
    if df.empty: return pd.Series([], dtype=int), pd.Series([], dtype=int)
    white_balls = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)
    powerball_freq = df['Powerball'].astype(int).value_counts().reindex(range(1, 27), fill_value=0)
    return white_ball_freq, powerball_freq

def hot_cold_numbers(df, last_draw_date_str):
    if df.empty or last_draw_date_str == 'N/A': return pd.Series([], dtype=int), pd.Series([], dtype=int)
    last_draw_date = pd.to_datetime(last_draw_date_str)
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    
    recent_data = df[df['Draw Date_dt'] >= one_year_ago].copy()
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

def monthly_white_ball_analysis(df, last_draw_date_str):
    print("[DEBUG-Monthly] Inside monthly_white_ball_analysis function.")
    if df.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-Monthly] df is empty or last_draw_date_str is N/A. Returning empty dict.")
        return {}

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-Monthly] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-Monthly] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty dict.")
        return {}

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-Monthly] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Draw Date_dt']):
        print("[ERROR-Monthly] 'Draw Date_dt' column missing or not datetime type in df. Attempting to re-create it.")
        try:
            df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'], errors='coerce')
            df = df.dropna(subset=['Draw Date_dt'])
            if df.empty:
                print("[ERROR-Monthly] Re-creating 'Draw Date_dt' resulted in empty DataFrame. Returning empty dict.")
                return {}
            print("[DEBUG-Monthly] Successfully re-created 'Draw Date_dt' column.")
        except Exception as e_recreate:
            print(f"[ERROR-Monthly] Failed to re-create 'Draw Date_dt' column: {e_recreate}. Returning empty dict.")
            return {}


    recent_data = df[df['Draw Date_dt'] >= six_months_ago].copy()
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


def sum_of_main_balls(df):
    """Calculates the sum of the five main white balls for each draw."""
    if df.empty:
        return pd.DataFrame(), [], 0, 0, 0.0
    
    temp_df = df.copy()
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

def find_results_by_sum(df, target_sum):
    if df.empty: return pd.DataFrame()
    temp_df = df.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)

    if 'Sum' not in temp_df.columns:
        temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    results = temp_df[temp_df['Sum'] == target_sum]
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Sum']]

def simulate_multiple_draws(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
    if df.empty: return pd.Series([], dtype=int)
    results = []
    for _ in range(num_draws):
        try:
            white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
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

    total_winning_white_comb = calculate_combinations(5, data["matched_w"])

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
            comb_p = calculate_combinations(1, 1)
        else:
            comb_p = calculate_combinations(total_powerballs_in_range - 1, 1)
            if total_powerballs_in_range == 1:
                comb_p = 0
        
        numerator = comb_matched_w * comb_unmatched_w * comb_p
        
        if numerator == 0:
            probabilities[scenario] = "N/A"
        else:
            total_possible_combinations = total_winning_white_comb * total_powerballs_in_range
            probability = total_possible_combinations / numerator
            probabilities[scenario] = f"{probability:,.0f} to 1"

    return probabilities


def export_analysis_results(df, file_path="analysis_results.csv"):
    df.to_csv(file_path, index=False)
    print(f"Analysis results saved to {file_path}")

def find_last_draw_dates_for_numbers(df, white_balls, powerball):
    if df.empty: return {}
    last_draw_dates = {}
    
    sorted_df = df.sort_values(by='Draw Date_dt', ascending=False)

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

def modify_combination(df, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers):
    if df.empty:
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

def find_common_pairs(df, top_n=10):
    if df.empty: return []
    pair_count = defaultdict(int)
    for _, row in df.iterrows():
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

def generate_with_common_pairs(df, common_pairs, white_ball_range, excluded_numbers):
    if df.empty:
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

def get_number_age_distribution(df):
    if df.empty: return [], [] # Return both lists
    df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'])
    all_draw_dates = sorted(df['Draw Date_dt'].drop_duplicates().tolist())
    
    detailed_ages = []
    
    # Process White Balls
    for i in range(1, 70):
        last_appearance_date = None
        temp_df_filtered = df[(df['Number 1'].astype(int) == i) | (df['Number 2'].astype(int) == i) |
                              (df['Number 3'].astype(int) == i) | (df['Number 4'].astype(int) == i) |
                              (df['Number 5'].astype(int) == i)]
        
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
        temp_df_filtered = df[df['Powerball'].astype(int) == i]
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

def get_co_occurrence_matrix(df):
    if df.empty: return [], 0
    co_occurrence = defaultdict(int)
    
    for index, row in df.iterrows():
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

def get_powerball_position_frequency(df):
    if df.empty: return []
    position_frequency_data = []
    
    for index, row in df.iterrows():
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
def get_consecutive_numbers_trends(df, last_draw_date_str):
    print("[DEBUG-ConsecutiveTrends] Inside get_consecutive_numbers_trends function.")
    if df.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-ConsecutiveTrends] df is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-ConsecutiveTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-ConsecutiveTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return []

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-ConsecutiveTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Draw Date_dt']):
        print("[ERROR-ConsecutiveTrends] 'Draw Date_dt' column missing or not datetime type in df. Returning empty list.")
        return []

    recent_data = df[df['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt') # Ensure chronological order
    print(f"[DEBUG-ConsecutiveTrends] recent_data shape after filtering: {recent_data.shape}")
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

# NEW FUNCTION: Get most frequent triplets
def get_most_frequent_triplets(df, top_n=10):
    print("[DEBUG-Triplets] Inside get_most_frequent_triplets function.")
    if df.empty:
        print("[DEBUG-Triplets] df is empty. Returning empty list.")
        return []

    triplet_counts = defaultdict(int)

    # Iterate through each draw
    for idx, row in df.iterrows():
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


def get_odd_even_split_trends(df, last_draw_date_str):
    print("[DEBUG-OddEvenTrends] Inside get_odd_even_split_trends function.")
    if df.empty or last_draw_date_str == 'N/A':
        print("[DEBUG-OddEvenTrends] df is empty or last_draw_date_str is N/A. Returning empty list.")
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
        print(f"[DEBUG-OddEvenTrends] last_draw_date: {last_draw_date}")
    except Exception as e:
        print(f"[ERROR-OddEvenTrends] Failed to convert last_draw_date_str '{last_draw_date_str}' to datetime: {e}. Returning empty list.")
        return []

    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    print(f"[DEBUG-OddEvenTrends] six_months_ago: {six_months_ago}")

    if 'Draw Date_dt' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Draw Date_dt']):
        print("[ERROR-OddEvenTrends] 'Draw Date_dt' column missing or not datetime type in df. Returning empty list.")
        return []

    recent_data = df[df['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt') # Ensure chronological order
    print(f"[DEBUG-OddEvenTrends] recent_data shape after filtering: {recent_data.shape}")
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
        
        # Now, let's process trend_data to get counts per category per date for plotting multiple lines
        # Or, just append the draw date and its split
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


# --- Global Data Loading and Pre-computation ---
df = pd.DataFrame()
last_draw = pd.Series(dtype='object') 

precomputed_white_ball_freq_list = []
precomputed_powerball_freq_list = []
precomputed_last_draw_date_str = "N/A"
precomputed_hot_numbers_list = []
precomputed_cold_numbers_list = []
precomputed_monthly_balls = {}
precomputed_number_age_counts = [] 
precomputed_detailed_number_ages = [] 
precomputed_co_occurrence_data = []
precomputed_max_co_occurrence = 0
precomputed_powerball_position_data = []
precomputed_odd_even_trends = [] 
precomputed_consecutive_trends = [] 
precomputed_triplets = [] 


# This function will be called once after app initialization to load data
def initialize_app_data():
    global df, last_draw, precomputed_white_ball_freq_list, precomputed_powerball_freq_list, \
           precomputed_last_draw_date_str, precomputed_hot_numbers_list, precomputed_cold_numbers_list, \
           precomputed_monthly_balls, precomputed_number_age_counts, precomputed_detailed_number_ages, \
           precomputed_co_occurrence_data, precomputed_max_co_occurrence, precomputed_powerball_position_data, \
           precomputed_odd_even_trends, precomputed_consecutive_trends, precomputed_triplets
    
    print("Attempting to load and pre-compute data...")
    try:
        df_temp = load_historical_data_from_supabase() # Load data into a temp DataFrame
        
        if not df_temp.empty: # Only update global df if data was successfully loaded
            df = df_temp # Assign to global df
            last_draw = get_last_draw(df)

            white_ball_freq, powerball_freq = frequency_analysis(df)
            precomputed_white_ball_freq_list.clear() # Clear before extending
            precomputed_white_ball_freq_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.items()])
            
            precomputed_powerball_freq_list.clear() # Clear before extending
            precomputed_powerball_freq_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in powerball_freq.items()])
            
            precomputed_last_draw_date_str = last_draw['Draw Date']
            
            hot_numbers, cold_numbers = hot_cold_numbers(df, precomputed_last_draw_date_str)
            precomputed_hot_numbers_list.clear() # Clear before extending
            precomputed_hot_numbers_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in hot_numbers.items()])
            
            precomputed_cold_numbers_list.clear() # Clear before extending
            precomputed_cold_numbers_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in cold_numbers.items()])
            
            precomputed_monthly_balls = monthly_white_ball_analysis(df, precomputed_last_draw_date_str)
            
            age_counts_data, detailed_ages_data = get_number_age_distribution(df)
            precomputed_number_age_counts.clear() # For the chart
            precomputed_number_age_counts.extend(age_counts_data)
            precomputed_detailed_number_ages.clear() # For the table
            precomputed_detailed_number_ages.extend(detailed_ages_data)
            
            co_occurrence_data, max_co_occurrence = get_co_occurrence_matrix(df)
            precomputed_co_occurrence_data.clear() # Clear before extending
            precomputed_co_occurrence_data.extend(co_occurrence_data)
            precomputed_max_co_occurrence = max_co_occurrence

            precomputed_powerball_position_data.clear() # Clear before extending
            precomputed_powerball_position_data.extend(get_powerball_position_frequency(df))

            precomputed_odd_even_trends = get_odd_even_split_trends(df, precomputed_last_draw_date_str)
            
            precomputed_consecutive_trends = get_consecutive_numbers_trends(df, precomputed_last_draw_date_str)
            
            precomputed_triplets = get_most_frequent_triplets(df)


            print("\n--- DEBUG: Precomputed Analysis Data Status ---")
            print(f"precomputed_white_ball_freq_list is empty: {not bool(precomputed_white_ball_freq_list)}")
            print(f"precomputed_white_ball_freq_list sample: {precomputed_white_ball_freq_list[:5]}")
            print(f"precomputed_powerball_freq_list is empty: {not bool(precomputed_powerball_freq_list)}")
            print(f"precomputed_powerball_freq_list sample: {precomputed_powerball_freq_list[:5]}")
            print(f"precomputed_hot_numbers_list is empty: {not bool(precomputed_hot_numbers_list)}")
            print(f"precomputed_hot_numbers_list sample: {precomputed_hot_numbers_list[:5]}")
            print(f"precomputed_cold_numbers_list is empty: {not bool(precomputed_cold_numbers_list)}")
            print(f"precomputed_cold_numbers_list sample: {precomputed_cold_numbers_list[:5]}")
            print(f"precomputed_monthly_balls is empty: {not bool(precomputed_monthly_balls)}")
            print(f"precomputed_monthly_balls sample: {list(precomputed_monthly_balls.keys())[:2] if precomputed_monthly_balls else 'N/A'}")
            print(f"precomputed_number_age_counts is empty: {not bool(precomputed_number_age_counts)}")
            print(f"precomputed_number_age_counts sample: {precomputed_number_age_counts[:2] if precomputed_number_age_counts else 'N/A'}")
            print(f"precomputed_detailed_number_ages is empty: {not bool(precomputed_detailed_number_ages)}")
            print(f"precomputed_detailed_number_ages sample: {precomputed_detailed_number_ages[:2] if precomputed_detailed_number_ages else 'N/A'}")
            print(f"precomputed_co_occurrence_data is empty: {not bool(precomputed_co_occurrence_data)}")
            print(f"precomputed_co_occurrence_data sample: {precomputed_co_occurrence_data[:2] if precomputed_co_occurrence_data else 'N/A'}")
            print(f"precomputed_powerball_position_data is empty: {not bool(precomputed_powerball_position_data)}")
            print(f"precomputed_powerball_position_data sample: {precomputed_powerball_position_data[:2] if precomputed_powerball_position_data else 'N/A'}")
            print(f"precomputed_odd_even_trends is empty: {not bool(precomputed_odd_even_trends)}")
            print(f"precomputed_odd_even_trends sample: {precomputed_odd_even_trends[:2] if precomputed_odd_even_trends else 'N/A'}")
            print(f"precomputed_consecutive_trends is empty: {not bool(precomputed_consecutive_trends)}")
            print(f"precomputed_consecutive_trends sample: {precomputed_consecutive_trends[:2] if precomputed_consecutive_trends else 'N/A'}")
            print(f"precomputed_triplets is empty: {not bool(precomputed_triplets)}")
            print(f"precomputed_triplets sample: {precomputed_triplets[:2] if precomputed_triplets else 'N/A'}")
            print("--- END DEBUG ---")
        else:
            print("DataFrame is empty after loading. Skipping pre-computation and leaving precomputed lists/dicts empty.")

    except Exception as e:
        print(f"An error occurred during initial data loading or pre-computation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for initialization errors

# Call the initialization function once when the module is loaded
initialize_app_data()


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
        return redirect(url_for('index'))

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
    return render_template('frequency_analysis.html', 
                           white_ball_freq=precomputed_white_ball_freq_list, 
                           powerball_freq=precomputed_powerball_freq_list)

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    return render_template('hot_cold_numbers.html', 
                           hot_numbers=precomputed_hot_numbers_list, 
                           cold_numbers=precomputed_cold_numbers_list)

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    # The precomputed data is already available from initialize_app_data
    monthly_balls_json = json.dumps(precomputed_monthly_balls)
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_balls=precomputed_monthly_balls,
                           monthly_balls_json=monthly_balls_json)


@app.route('/sum_of_main_balls')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot display Sum of Main Balls: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    sums_data_df, sum_freq_list, min_sum, max_sum, avg_sum = sum_of_main_balls(df)
    
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
    probability_1_in_x, probability_percentage = winning_probability(white_ball_range, powerball_range)

    return render_template('winning_probability.html', 
                           probability_1_in_x=probability_1_in_x, 
                           probability_percentage=probability_percentage)

@app.route('/partial_match_probabilities')
def partial_match_probabilities_route():
    probabilities = partial_match_probabilities(white_ball_range, powerball_range)

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
    # Pass both the aggregated data (for the chart) and the detailed data (for the table)
    return render_template('number_age_distribution.html',
                           number_age_data=precomputed_number_age_counts, # For the chart (renamed for clarity)
                           detailed_number_ages=precomputed_detailed_number_ages) # For the new table

@app.route('/co_occurrence_analysis')
def co_occurrence_analysis_route():
    return render_template('co_occurrence_analysis.html',
                           co_occurrence_data=precomputed_co_occurrence_data,
                           max_co_occurrence=precomputed_max_co_occurrence)

@app.route('/powerball_position_frequency')
def powerball_position_frequency_route():
    return render_template('powerball_position_frequency.html',
                           powerball_position_data=precomputed_powerball_position_data)

@app.route('/odd_even_trends')
def odd_even_trends_route():
    return render_template('odd_even_trends.html',
                           odd_even_trends=precomputed_odd_even_trends)

@app.route('/consecutive_trends')
def consecutive_trends_route():
    return render_template('consecutive_trends.html',
                           consecutive_trends=precomputed_consecutive_trends)

@app.route('/triplets_analysis')
def triplets_analysis_route():
    return render_template('triplets_analysis.html',
                           triplets_data=precomputed_triplets)

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
            
            global df, last_draw, precomputed_white_ball_freq_list, precomputed_powerball_freq_list, \
                   precomputed_last_draw_date_str, precomputed_hot_numbers_list, precomputed_cold_numbers_list, \
                   precomputed_monthly_balls, precomputed_number_age_counts, precomputed_detailed_number_ages, \
                   precomputed_co_occurrence_data, precomputed_max_co_occurrence, precomputed_powerball_position_data, \
                   precomputed_odd_even_trends, precomputed_consecutive_trends, precomputed_triplets

            df = load_historical_data_from_supabase()
            last_draw = get_last_draw(df)

            if not df.empty:
                white_ball_freq, powerball_freq = frequency_analysis(df)
                precomputed_white_ball_freq_list.clear()
                precomputed_white_ball_freq_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.items()])
                
                precomputed_powerball_freq_list.clear()
                precomputed_powerball_freq_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in powerball_freq.items()])
                
                precomputed_last_draw_date_str = last_draw['Draw Date']
                
                hot_numbers, cold_numbers = hot_cold_numbers(df, precomputed_last_draw_date_str)
                precomputed_hot_numbers_list.clear()
                precomputed_hot_numbers_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in hot_numbers.items()])
                
                precomputed_cold_numbers_list.clear()
                precomputed_cold_numbers_list.extend([{'Number': int(k), 'Frequency': int(v)} for k, v in cold_numbers.items()])
                
                # Update precomputed_monthly_balls after new data
                precomputed_monthly_balls = monthly_white_ball_analysis(df, precomputed_last_draw_date_str)
                
                # Update number age data to get both counts and detailed list
                age_counts_data, detailed_ages_data = get_number_age_distribution(df)
                precomputed_number_age_counts.clear() # For the chart
                precomputed_number_age_counts.extend(age_counts_data)
                precomputed_detailed_number_ages.clear() # For the table
                precomputed_detailed_number_ages.extend(detailed_ages_data)
                
                co_occurrence_data, max_co_occurrence = get_co_occurrence_matrix(df)
                precomputed_co_occurrence_data.clear()
                precomputed_co_occurrence_data.extend(co_occurrence_data)
                precomputed_max_co_occurrence = max_co_occurrence

                precomputed_powerball_position_data.clear()
                precomputed_powerball_position_data.extend(get_powerball_position_frequency(df))

                precomputed_odd_even_trends = get_odd_even_split_trends(df, precomputed_last_draw_date_str)
                
                precomputed_consecutive_trends = get_consecutive_numbers_trends(df, precomputed_last_draw_date_str)

                precomputed_triplets = get_most_frequent_triplets(df)


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
