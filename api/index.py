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
import traceback 

# --- Supabase Configuration ---
SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "YOUR_ACTUAL_SUPABASE_ANON_KEY_GOES_HERE") 
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "YOUR_ACTUAL_SUPABASE_SERVICE_ROLE_KEY_GOES_HERE") 

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

historical_white_ball_sets = set() 
white_ball_co_occurrence_lookup = {}

analysis_cache = {}
last_analysis_cache_update = datetime.min 

CACHE_EXPIRATION_SECONDS = 3600

group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]
GLOBAL_WHITE_BALL_RANGE = (1, 69)
GLOBAL_POWERBALL_RANGE = (1, 26)
excluded_numbers = [] # Global excluded numbers, can be overridden by form input

NUMBER_RANGES = {
    "1-9": (1, 9),
    "10s": (10, 19),
    "20s": (20, 29),
    "30s": (30, 39),
    "40s": (40, 49),
    "50s": (50, 59),
    "60s": (60, 69)
}

ASCENDING_GEN_RANGES = [
    (10, 19), 
    (20, 29), 
    (30, 39), 
    (40, 49), 
    (50, 59), 
    (60, 69)  
]

SUM_RANGES = {
    "Any": None, 
    "Zone A (60-99)": (60, 99),
    "Zone B (100-129)": (100, 129),
    "Zone C (130-159)": (130, 159), 
    "Zone D (160-189)": (160, 189),
    "Zone E (190-220)": (190, 220),
    "Zone F (221-249)": (221, 249), 
    "Zone G (250-300)": (250, 300)  
}

LOW_NUMBER_MAX = 34 
HIGH_NUMBER_MIN = 35 

POWERBALL_DRAW_DAYS = ['Monday', 'Wednesday', 'Saturday']

BOUNDARY_PAIRS_TO_ANALYZE = [
    (9, 10), (19, 20), (29, 30), (39, 40), (49, 50), (59, 60)
]

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


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
            'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A'] 
        }, dtype='object')
    
    last_row = df.iloc[-1].copy() 
    
    if 'Numbers' not in last_row or not isinstance(last_row['Numbers'], list):
        last_row['Numbers'] = [
            int(last_row['Number 1']), int(last_row['Number 2']), int(last_row['Number 3']), 
            int(last_row['Number 4']), int(last_row['Number 5'])
        ]
    return last_row

def check_exact_match(white_balls):
    global historical_white_ball_sets
    return frozenset(white_balls) in historical_white_ball_sets

def generate_powerball_numbers(df_source, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None, selected_sum_range_tuple=None, is_simulation=False):
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 5000 # Increased attempts for stricter adherence, especially with odd/even
    attempts = 0
    
    # Pre-filter available numbers for white balls
    base_available_white_balls = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
    if len(base_available_white_balls) < 5:
        raise ValueError("Not enough available white balls after exclusions and range constraints.")

    while attempts < max_attempts:
        
        # 1. Generate candidate white balls
        white_balls_candidate = sorted(random.sample(base_available_white_balls, 5))

        # 2. **STRICT ODD/EVEN CHECK (PRIORITY)**
        even_count = sum(1 for num in white_balls_candidate if num % 2 == 0)
        odd_count = 5 - even_count

        if odd_even_choice == "All Even" and even_count != 5:
            attempts += 1
            continue
        elif odd_even_choice == "All Odd" and odd_count != 5:
            attempts += 1
            continue
        elif odd_even_choice == "3 Even / 2 Odd" and (even_count != 3 or odd_count != 2):
            attempts += 1
            continue
        elif odd_even_choice == "2 Even / 3 Odd" and (even_count != 2 or odd_count != 3):
            attempts += 1
            continue
        elif odd_even_choice == "1 Even / 4 Odd" and (even_count != 1 or odd_count != 4):
            attempts += 1
            continue
        elif odd_even_choice == "4 Even / 1 Odd" and (even_count != 4 or odd_count != 1):
            attempts += 1
            continue
        # If odd_even_choice is "Any", this block is skipped, which is fine.

        # 3. Check Sum Range constraint
        if selected_sum_range_tuple:
            current_sum = sum(white_balls_candidate)
            if not (selected_sum_range_tuple[0] <= current_sum <= selected_sum_range_tuple[1]):
                attempts += 1
                continue 

        # 4. Check Group A constraint (reverting to original logic from index (2).py, as it does not pass num_from_group_a directly to this function for 'simulate_multiple_draws')
        group_a_numbers = [num for num in white_balls_candidate if num in group_a]
        if len(group_a_numbers) < 2: # This condition might make generation very hard if group_a is small or numbers are excluded.
                                     # The original index (2).py had this, so keeping it.
            attempts += 1
            continue
            
        # 5. Check High/Low balance
        if high_low_balance is not None:
            low_numbers_count = sum(1 for num in white_balls_candidate if num <= LOW_NUMBER_MAX)
            high_numbers_count = sum(1 for num in white_balls_candidate if num >= HIGH_NUMBER_MIN)
            if low_numbers_count != high_low_balance[0] or high_numbers_count != high_low_balance[1]:
                attempts += 1
                continue
        
        # 6. Check against last draw (if not simulation)
        if not is_simulation:
            last_draw_data = get_last_draw(df_source)
            if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
                # Ensure last_draw_data numbers are properly converted to int for comparison
                last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
                if set(white_balls_candidate) == set(last_white_balls): # Powerball comparison removed here for flexibility
                    attempts += 1
                    continue

            # 7. Check for exact historical white ball match
            if check_exact_match(white_balls_candidate): 
                attempts += 1
                continue

        # If all checks pass, generate powerball and return
        powerball = random.randint(powerball_range[0], powerball_range[1])
        return white_balls_candidate, powerball

    # If loop finishes without returning, it means we couldn't find a valid combination
    raise ValueError("Could not generate a unique combination meeting all criteria after many attempts. Try adjusting filters or increasing max_attempts.")


def generate_from_group_a(df_source, num_from_group_a, white_ball_range, powerball_range, excluded_numbers, selected_sum_range_tuple=None):
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 2000 
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
            
            available_for_remaining = [num for num in remaining_pool if num not in selected_from_group_a]
            if len(available_for_remaining) < num_from_remaining:
                attempts += 1
                continue 

            selected_from_remaining = random.sample(available_for_remaining, num_from_remaining) 
            
            white_balls = sorted(selected_from_group_a + selected_from_remaining)
            
            if selected_sum_range_tuple:
                current_sum = sum(white_balls)
                if not (selected_sum_range_tuple[0] <= current_sum <= selected_sum_range_tuple[1]):
                    attempts += 1
                    continue 

            powerball = random.randint(powerball_range[0], powerball_range[1])

            if check_exact_match(white_balls): 
                attempts += 1
                continue

            break
        except ValueError as e:
            attempts += 1
            continue
        except IndexError: 
            attempts += 1
            continue
    else:
        raise ValueError("Could not generate a unique combination with Group A strategy meeting all criteria after many attempts. Try adjusting filters.")

    return white_balls, powerball

def generate_with_user_provided_pair(num1, num2, white_ball_range, powerball_range, excluded_numbers, df_source, selected_sum_range_tuple=None):
    """
    Generates a Powerball combination starting with two user-provided white balls.
    The remaining three numbers are generated in ascending order from specific tens ranges (20s-60s).
    """
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    if not (white_ball_range[0] <= num1 <= white_ball_range[1] and
            white_ball_range[0] <= num2 <= white_ball_range[1]):
        raise ValueError(f"Provided numbers ({num1}, {num2}) must be within the white ball range ({white_ball_range[0]}-{white_ball_range[1]}).")
    
    if num1 == num2:
        raise ValueError("The two provided white balls must be unique.")
    
    if num1 in excluded_numbers or num2 in excluded_numbers:
        raise ValueError(f"One or both provided numbers ({num1}, {num2}) are in the excluded list.")

    initial_white_balls = sorted([num1, num2]) 
    
    max_attempts_overall = 2000 
    attempts_overall = 0

    while attempts_overall < max_attempts_overall:
        candidate_white_balls_generated = []
        temp_current_min = initial_white_balls[-1] + 1 
        
        try:
            for i in range(3): 
                possible_nums_for_slot = []
                
                start_range_idx = -1
                for idx, (range_min, range_max) in enumerate(ASCENDING_GEN_RANGES):
                    if temp_current_min <= range_max and temp_current_min >= range_min: 
                        start_range_idx = idx
                        break
                    elif temp_current_min < range_min: 
                        start_range_idx = idx
                        break
                
                if start_range_idx == -1: 
                    raise ValueError("Not enough space in ascending ranges to complete combination.")

                eligible_ranges = ASCENDING_GEN_RANGES[start_range_idx:]
                
                for range_min, range_max in eligible_ranges:
                    actual_start_val = max(temp_current_min, range_min) 
                    
                    for num in range(actual_start_val, range_max + 1):
                        if num not in excluded_numbers and \
                           num not in initial_white_balls and \
                           num not in candidate_white_balls_generated:
                            possible_nums_for_slot.append(num)
                
                if not possible_nums_for_slot:
                    raise ValueError(f"No available numbers for slot {i+3}. Current min: {temp_current_min}, initial: {initial_white_balls}, generated: {candidate_white_balls_generated}")

                picked_num = random.choice(possible_nums_for_slot)
                candidate_white_balls_generated.append(picked_num)
                temp_current_min = picked_num + 1 
            
            final_white_balls = sorted(initial_white_balls + candidate_white_balls_generated)
            
            if selected_sum_range_tuple:
                current_sum = sum(final_white_balls)
                if not (selected_sum_range_tuple[0] <= current_sum <= selected_sum_range_tuple[1]):
                    attempts_overall += 1
                    continue 

            powerball = random.randint(powerball_range[0], powerball_range[1])

            if check_exact_match(final_white_balls):
                attempts_overall += 1
                continue 

            last_draw_data = get_last_draw(df_source)
            if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
                last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
                if set(final_white_balls) == set(last_white_balls) and powerball == int(last_draw_data['Powerball']):
                    attempts_overall += 1
                    continue

            return final_white_balls, powerball

        except ValueError as e:
            attempts_overall += 1
            continue
        except IndexError: 
            attempts_overall += 1
            continue
    else:
        raise ValueError("Could not generate a unique combination with the provided pair and ascending range constraint meeting all criteria after many attempts. Try adjusting filters.")


def check_historical_match(white_balls, powerball):
    global historical_white_ball_sets
    return frozenset(white_balls) in historical_white_ball_sets

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

def get_monthly_white_ball_analysis_data(dataframe, num_top_wb=69, num_top_pb=3, num_months_for_top_display=6):
    if dataframe.empty:
        return {'monthly_data': [], 'streak_numbers': {'3_month_streaks': [], '4_month_streaks': [], '5_month_streaks': []}}

    df_sorted = dataframe.sort_values(by='Draw Date_dt', ascending=False).copy()
    df_sorted['YearMonth'] = df_sorted['Draw Date_dt'].dt.to_period('M')
    unique_months_periods = sorted(df_sorted['YearMonth'].unique(), reverse=True)

    monthly_display_data = [] 
    current_period = pd.Period(datetime.now(), freq='M')
    
    processed_months_count = 0
    for period in unique_months_periods:
        if processed_months_count >= num_months_for_top_display:
            if not (period == current_period and processed_months_count < num_months_for_top_display):
                break

        month_df = df_sorted[df_sorted['YearMonth'] == period]
        if month_df.empty:
            continue

        is_current_month_flag = (period == current_period)

        drawn_white_balls_set = set()
        wb_monthly_counts = defaultdict(int) 
        for _, row in month_df.iterrows():
            for i in range(1, 6):
                num = int(row[f'Number {i}'])
                drawn_white_balls_set.add(num)
                wb_monthly_counts[num] += 1
        
        drawn_wb_with_counts = sorted([{'number': n, 'count': wb_monthly_counts[n]} for n in drawn_white_balls_set], key=lambda x: x['number'])

        all_possible_white_balls = set(range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1))
        not_picked_white_balls = sorted(list(all_possible_white_balls - drawn_white_balls_set))

        drawn_powerballs_set = set()
        pb_monthly_counts = defaultdict(int) 
        for _, row in month_df.iterrows():
            pb_num = int(row['Powerball'])
            drawn_powerballs_set.add(pb_num)
            pb_monthly_counts[pb_num] += 1

        sorted_pb_freq = sorted(pb_monthly_counts.items(), key=lambda item: (-item[1], item[0])) 
        top_pb = [{'number': int(n), 'count': int(c)} for n, c in sorted_pb_freq[:num_top_pb]]

        all_possible_powerballs = set(range(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1] + 1))
        not_picked_powerballs = sorted(list(all_possible_powerballs - drawn_powerballs_set))

        monthly_display_data.append({
            'month': period.strftime('%B %Y'),
            'drawn_white_balls_with_counts': drawn_wb_with_counts,
            'not_picked_white_balls': not_picked_white_balls,
            'top_powerballs': top_pb,
            'not_picked_powerballs': not_picked_powerballs,
            'is_current_month': is_current_month_flag
        })
        processed_months_count += 1
    
    monthly_display_data.sort(key=lambda x: datetime.strptime(x['month'], '%B %Y'))

    numbers_per_completed_month = defaultdict(set)
    for period in unique_months_periods:
        if period == current_period: 
            continue
        month_df = df_sorted[df_sorted['YearMonth'] == period]
        if not month_df.empty:
            for _, row in month_df.iterrows():
                for i in range(1, 6):
                    numbers_per_completed_month[period].add(int(row[f'Number {i}']))
                numbers_per_completed_month[period].add(int(row['Powerball']))
    
    completed_months_sorted = sorted([p for p in unique_months_periods if p != current_period])

    streak_numbers = {'3_month_streaks': [], '4_month_streaks': [], '5_month_streaks': []}

    all_possible_numbers = set(range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1)) \
                           .union(set(range(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1] + 1)))

    for num in all_possible_numbers:
        current_streak_length = 0
        
        for i in range(len(completed_months_sorted) - 1, -1, -1): 
            month_period = completed_months_sorted[i]
            if num in numbers_per_completed_month[month_period]:
                current_streak_length += 1
            else:
                break 
        
        if current_streak_length >= 5:
            streak_numbers['5_month_streaks'].append(int(num))
        if current_streak_length >= 4:
            streak_numbers['4_month_streaks'].append(int(num))
        if current_streak_length >= 3:
            streak_numbers['3_month_streaks'].append(int(num))
    
    streak_numbers['3_month_streaks'] = sorted(list(set(streak_numbers['3_month_streaks'])))
    streak_numbers['4_month_streaks'] = sorted(list(set(streak_numbers['4_month_streaks'])))
    streak_numbers['5_month_streaks'] = sorted(list(set(streak_numbers['5_month_streaks'])))

    return {
        'monthly_data': monthly_display_data, 
        'streak_numbers': streak_numbers
    }


def sum_of_main_balls(df_source):
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

    return temp_df[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum', 'Draw Date_dt']], sum_freq_list, min_sum, max_sum, avg_sum

def find_results_by_sum(df_source, target_sum):
    if df_source.empty: return pd.DataFrame()
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']: 
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)

    if 'Sum' not in temp_df.columns:
        temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    results = temp_df[temp_df['Sum'] == target_sum]
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum', 'Draw Date_dt']]

def simulate_multiple_draws(df_source, group_a, odd_even_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
    if df_source.empty: 
        return {'white_ball_freq': [], 'powerball_freq': []}
    
    white_ball_results = defaultdict(int)
    powerball_results = defaultdict(int)

    for _ in range(num_draws):
        try:
            # Pass the selected_sum_range_tuple as None here, as it's not a direct filter for simulate_multiple_draws
            # Also, high_low_balance is not used in this context.
            white_balls, powerball = generate_powerball_numbers(
                df_source, group_a, odd_even_choice, "No Combo", 
                white_ball_range, powerball_range, excluded_numbers, 
                high_low_balance=None, selected_sum_range_tuple=None, is_simulation=True
            )
            
            for wb in white_balls:
                white_ball_results[wb] += 1
            powerball_results[powerball] += 1

        except ValueError: # Catch ValueErrors from generate_powerball_numbers if constraints too tight
            pass 
    
    full_white_ball_range_list = list(range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1))
    simulated_white_ball_freq_list = sorted([
        {'Number': n, 'Frequency': white_ball_results[n]} for n in full_white_ball_range_list
    ], key=lambda x: x['Number'])

    full_powerball_range_list = list(range(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1] + 1))
    simulated_powerball_freq_list = sorted([
        {'Number': n, 'Frequency': powerball_results[n]} for n in full_powerball_range_list
    ], key=lambda x: x['Number'])

    return {'white_ball_freq': simulated_white_ball_freq_list, 'powerball_freq': simulated_powerball_freq_list}


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

def _get_last_drawn_date_for_single_number(df_source, number):
    if df_source.empty:
        return "N/A"

    if 'Draw Date_dt' not in df_source.columns:
        df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'], errors='coerce')
        df_source = df_source.dropna(subset=['Draw Date_dt'])
        if df_source.empty: return "N/A" 

    sorted_df = df_source.sort_values(by='Draw Date_dt', ascending=False)

    for col_idx in range(1, 6):
        col_name = f'Number {col_idx}'
        if col_name in sorted_df.columns:
            matching_rows = sorted_df[sorted_df[col_name].astype(int) == number]
            if not matching_rows.empty:
                return matching_rows['Draw Date'].iloc[0] 

    if 'Powerball' in sorted_df.columns:
        matching_rows_pb = sorted_df[sorted_df['Powerball'].astype(int) == number]
        if not matching_rows_pb.empty:
            return matching_rows_pb['Draw Date'].iloc[0] 

    return "N/A" 

def _get_last_co_occurrence_date_for_pattern(df_source, pattern_numbers):
    if not pattern_numbers:
        return "N/A"

    target_pattern_set = frozenset(pattern_numbers)

    latest_date = "N/A"
    latest_datetime = datetime.min

    for historical_white_balls_set, draw_date_str in white_ball_co_occurrence_lookup.items():
        if target_pattern_set.issubset(historical_white_balls_set):
            try:
                current_draw_datetime = datetime.strptime(draw_date_str, '%Y-%m-%d')
                if current_draw_datetime > latest_datetime:
                    latest_datetime = current_draw_datetime
                    latest_date = draw_date_str
            except ValueError:
                pass
                
    return latest_date 

def get_number_age_distribution(df_source):
    if df_source.empty: return [], []
    df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'])
    all_draw_dates = sorted(df_source['Draw Date_dt'].drop_duplicates().tolist())
    
    detailed_ages = []
    
    for i in range(1, 70):
        last_appearance_date = None
        last_appearance_date_str = "N/A" 
        temp_df_filtered = df_source[(df_source['Number 1'].astype(int) == i) | (df_source['Number 2'].astype(int) == i) |
                              (df_source['Number 3'].astype(int) == i) | (df_source['Number 4'].astype(int) == i) |
                              (df_source['Number 5'].astype(int) == i)]
        
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()
            last_appearance_date_str = last_appearance_date.strftime('%Y-%m-%d')

        miss_streak_count = 0
        if last_appearance_date is not None:
            draw_dates_after_last_appearance = [d for d in all_draw_dates if d > last_appearance_date]
            miss_streak_count = len(draw_dates_after_last_appearance)

            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': miss_streak_count, 'last_drawn_date': last_appearance_date_str})
        else:
            detailed_ages.append({'number': int(i), 'type': 'White Ball', 'age': len(all_draw_dates), 'last_drawn_date': last_appearance_date_str})

    for i in range(1, 27):
        last_appearance_date = None
        last_appearance_date_str = "N/A" 
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
            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': len(all_draw_dates), 'last_drawn_date': len(all_draw_dates)}) 

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

def _find_consecutive_sequences(numbers_list):
    """
    Identifies and returns all consecutive sequences (pairs, triplets, etc.)
    from a list of numbers.
    Example: [1, 2, 3, 5, 6] -> [[1, 2, 3], [5, 6]]
    """
    sequences = []
    if not numbers_list:
        return sequences

    sorted_nums = sorted(list(set(numbers_list))) # Ensure unique and sorted
    if not sorted_nums:
        return sequences

    current_sequence = [sorted_nums[0]]
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] == current_sequence[-1] + 1:
            current_sequence.append(sorted_nums[i])
        else:
            if len(current_sequence) >= 2: # Only add sequences of 2 or more
                sequences.append(current_sequence)
            current_sequence = [sorted_nums[i]]
    
    # Add the last sequence if it's long enough
    if len(current_sequence) >= 2:
        sequences.append(current_sequence)
        
    return sequences

def get_consecutive_numbers_trends(df_source, last_draw_date_str):
    if df_source.empty or last_draw_date_str == 'N/A':
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
    except Exception as e:
        return []

    # Change this line from 6 months to 12 months (1 year)
    one_year_ago = last_draw_date - pd.DateOffset(years=1) # Changed from months=6 to years=1

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        return []

    # Filter data for the last 12 months
    recent_data = df_source[df_source['Draw Date_dt'] >= one_year_ago].copy() # Changed to one_year_ago
    recent_data = recent_data.sort_values(by='Draw Date_dt', ascending=False)
    if recent_data.empty:
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        # Call the new function _find_consecutive_sequences
        consecutive_sequences = _find_consecutive_sequences(white_balls)
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            # Use the new variable name 'consecutive_sequences'
            'consecutive_present': "Yes" if consecutive_sequences else "No",
            # Use the new key 'consecutive_sequences'
            'consecutive_sequences': consecutive_sequences
        })
    
    return trend_data

def get_most_frequent_triplets(df_source): 
    if df_source.empty:
        return []

    triplet_counts = defaultdict(int)

    for idx, row in df_source.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        for triplet_combo in combinations(sorted(white_balls), 3):
            triplet_counts[triplet_combo] += 1
    
    sorted_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)
    
    formatted_triplets = []
    for triplet, count in sorted_triplets: 
        formatted_triplets.append({
            'triplet': list(triplet),
            'count': int(count)
        })
    
    return formatted_triplets


def get_odd_even_split_trends(df_source, last_draw_date_str):
    if df_source.empty or last_draw_date_str == 'N/A':
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
    except Exception as e:
        return []

    six_months_ago = last_draw_date - pd.DateOffset(months=6)

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= six_months_ago].copy()
    recent_data = recent_data.sort_values(by='Draw Date_dt', ascending=False)
    if recent_data.empty:
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        
        wb_sum = sum(white_balls)

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
    
    return trend_data
    
def get_powerball_frequency_by_year(df_source):
    if df_source.empty:
        return [], []

    current_year = datetime.now().year
    # Dynamically set start_year to ensure a rolling 10-year window (current year + 9 previous years)
    start_year = max(2017, current_year - 9) # Ensure it doesn't go before 2017 if data starts then
    years = [y for y in range(start_year, current_year + 1)]

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        df_source['Draw Date_dt'] = pd.to_datetime(df_source['Draw Date'], errors='coerce')
        df_source = df_source.dropna(subset=['Draw Date_dt'])
        if df_source.empty:
            return [], []

    recent_data = df_source[df_source['Draw Date_dt'].dt.year.isin(years)].copy()
    
    if recent_data.empty:
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

    return formatted_data, years

def _get_generated_picks_for_date_from_db(date_str):
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{GENERATED_NUMBERS_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=False)
    
    try:
        start_of_day_dt = datetime.strptime(date_str, '%Y-%m-%d')
        end_of_day_dt = start_of_day_dt + timedelta(days=1)
        start_of_day_iso = start_of_day_dt.isoformat(timespec='seconds') + "Z"
        end_of_day_iso = end_of_day_dt.isoformat(timespec='seconds') + "Z"
    except ValueError:
        return []

    params = {
        'select': 'generated_date,number_1,number_2,number_3,number_4,number_5,powerball',
        'order': 'generated_date.desc', 
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
                'time': datetime.fromisoformat(record['generated_date'].replace('Z', '+00:00')).strftime('%I:%M %p'), # Add time for display
                'white_balls': white_balls,
                'powerball': int(record['powerball'])
            })
        return formatted_picks
    except requests.exceptions.RequestException as e:
        return []
    except Exception as e:
        return []

def _get_official_draw_for_date_from_db(date_str):
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
            return raw_data[0] 
        return None
    except requests.exceptions.RequestException as e:
        return None
    except Exception as e:
        return []

def analyze_generated_batch_against_official_draw(generated_picks_list, official_draw):
    summary = {
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
    
    if not official_draw:
        return summary 

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
        summary[category]["draws"].append({
            "date": official_draw['Draw Date'], 
            "white_balls": official_white_balls,
            "powerball": official_powerball
        })

    return summary

def save_manual_draw_to_db(draw_date, n1, n2, n3, n4, n5, pb):
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=True)

    check_params = {'select': 'Draw Date', 'Draw Date': f'eq.{draw_date}'}
    check_response = requests.get(url, headers=headers, params=check_params)
    check_response.raise_for_status()
    existing_draws = check_response.json()

    if existing_draws:
        print(f"Draw for date {draw_date} already exists in {SUPABASE_TABLE_NAME}.")
        return False, f"Draw for {draw_date} already exists."

    sorted_white_balls = sorted([n1, n2, n3, n4, n5])

    new_draw_data = {
        'Draw Date': draw_date,
        'Number 1': sorted_white_balls[0], 
        'Number 2': sorted_white_balls[1],
        'Number 3': sorted_white_balls[2],
        'Number 4': sorted_white_balls[3],
        'Number 5': sorted_white_balls[4],
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
        if hasattr(e, 'response') and e.response is not None:
            pass
        return {}
    except json.JSONDecodeError as e:
        if 'response' in locals() and response is not None:
            pass
        return {}
    except Exception as e:
        traceback.print_exc()
        return {}


def check_generated_against_history(generated_white_balls, generated_powerball, df_historical):
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
        return []

    df_source_copy = df_source.copy()
    if 'Draw Date_dt' not in df_source_copy.columns:
        df_source_copy['Draw Date_dt'] = pd.to_datetime(df_source_copy['Draw Date'], errors='coerce')
    df_source_copy = df_source_copy.dropna(subset=['Draw Date_dt'])
    
    if df_source_copy.empty:
        return []

    for i in range(1, 6):
        col = f'Number {i}'
        if col in df_source_copy.columns:
            df_source_copy[col] = pd.to_numeric(df_source_copy[col], errors='coerce').fillna(0).astype(int)
        else:
            pass


    all_patterns_data = []
    
    for year in sorted(df_source_copy['Draw Date_dt'].dt.year.unique()):
        yearly_df = df_source_copy[df_source_copy['Draw Date_dt'].dt.year == year]
        
        year_pairs_counts = defaultdict(int)
        year_triplets_counts = defaultdict(int)

        for _, row in yearly_df.iterrows():
            white_balls = [int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])]
            
            for range_name, (min_val, max_val) in NUMBER_RANGES.items():
                numbers_in_current_range = sorted([num for num in white_balls if min_val <= num <= max_val])
                
                if len(numbers_in_current_range) >= 2:
                    for pair in combinations(numbers_in_current_range, 2):
                        year_pairs_counts[(range_name, tuple(sorted(pair)))] += 1
                
                if len(numbers_in_current_range) >= 3:
                    for triplet_combo in combinations(numbers_in_current_range, 3):
                        year_triplets_counts[(range_name, tuple(sorted(triplet_combo)))] += 1
        
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

    all_patterns_data.sort(key=lambda x: (x['count'], x['year'], x['range'], str(x['pattern'])), reverse=True)
    
    return all_patterns_data

def get_sum_trends_and_gaps_data(df_source):
    if df_source.empty:
        return {
            'min_possible_sum': 15,
            'max_possible_sum': 335,
            'appeared_sums_details': [],
            'missing_sums': [],
            'grouped_sums_analysis': {}
        }

    df_copy = df_source.copy()
    
    df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])
    for i in range(1, 6):
        col = f'Number {i}'
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)

    df_copy['Sum'] = df_copy[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)

    last_appearance_by_sum_df = df_copy.groupby('Sum')['Draw Date_dt'].max().reset_index()
    last_appearance_by_sum_df['last_drawn_date'] = last_appearance_by_sum_df['Draw Date_dt'].dt.strftime('%Y-%m-%d')
    last_drawn_dates_map = last_appearance_by_sum_df.set_index('Sum')['last_drawn_date'].to_dict()

    sum_freq_series = df_copy['Sum'].value_counts()
    sum_counts_map = sum_freq_series.to_dict()

    appeared_sums_details = sorted([
        {'sum': int(s), 'last_drawn_date': last_drawn_dates_map.get(s, 'N/A'), 'count': sum_counts_map.get(s, 0)}
        for s in sum_freq_series.index
    ], key=lambda x: x['sum'])

    min_possible_sum = 1 + 2 + 3 + 4 + 5 
    max_possible_sum = 69 + 68 + 67 + 66 + 65 

    all_possible_sums = set(range(min_possible_sum, max_possible_sum + 1))
    actual_appeared_sums = set(df_copy['Sum'].unique())
    missing_sums = sorted(list(all_possible_sums - actual_appeared_sums))

    grouped_sums_analysis = {}
    for range_name, range_tuple in SUM_RANGES.items():
        if range_tuple is None: 
            continue
        
        range_min, range_max = range_tuple 
        
        sums_in_current_range = sorted(list(set(range(range_min, range_max + 1)).intersection(all_possible_sums)))

        appeared_in_range_details = [
            {'sum': s_data['sum'], 'last_drawn_date': s_data['last_drawn_date'], 'count': s_data['count']}
            for s_data in appeared_sums_details if range_min <= s_data['sum'] <= range_max
        ]
        
        most_frequent_sums = sorted(appeared_in_range_details, key=lambda x: (-x['count'], x['sum']))[:5]
        
        least_frequent_sums = sorted([s for s in appeared_in_range_details if s['count'] > 0], key=lambda x: (x['count'], x['sum']))[:5]

        total_freq_in_range = sum(s['count'] for s in appeared_in_range_details)
        if appeared_in_range_details:
            avg_freq_in_range = round(total_freq_in_range / len(appeared_in_range_details), 2)
        else:
            avg_freq_in_range = 0.0

        draw_dates_for_range = df_copy[(df_copy['Sum'] >= range_min) & (df_copy['Sum'] <= range_max)]['Draw Date_dt']
        last_drawn_date_for_range = draw_dates_for_range.max().strftime('%Y-%m-%d') if not draw_dates_for_range.empty else 'N/A'

        grouped_sums_analysis[range_name] = {
            'total_possible_in_range': len(sums_in_current_range),
            'appeared_in_range_count': len(appeared_in_range_details),
            'missing_in_range_count': len([s for s in missing_sums if range_min <= s <= range_max]),
            'last_drawn_date_for_range': last_drawn_date_for_range,
            'average_frequency_in_range': avg_freq_in_range,
            'most_frequent_sums_in_range': most_frequent_sums,
            'least_frequent_sums_in_range': least_frequent_sums,
            'all_appeared_sums_in_range': appeared_in_range_details 
        }
    
    return {
        'min_possible_sum': min_possible_sum,
        'max_possible_sum': max_possible_sum,
        'appeared_sums_details': appeared_sums_details,
        'missing_sums': missing_sums,
        'grouped_sums_analysis': grouped_sums_analysis
    }

def get_weekday_draw_trends(df_source, group_a_numbers_def=None): 
    if df_source.empty:
        return {}

    df_copy = df_source.copy()
    df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])

    if df_copy.empty:
        return {}

    df_copy['Weekday'] = df_copy['Draw Date_dt'].dt.day_name()

    weekday_stats = defaultdict(lambda: {
        'total_draws': 0,
        'total_low_balls': 0,
        'total_high_balls': 0,
        'total_odd_balls': 0,
        'total_even_balls': 0,
        'total_sum': 0,
        'total_group_a_balls': 0,
        'consecutive_draws_count': 0,
        'low_high_splits': defaultdict(int),
        'odd_even_splits': defaultdict(int)
    })

    for _, row in df_copy.iterrows():
        day_name = row['Weekday']
        if day_name not in POWERBALL_DRAW_DAYS:
            continue
        
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6)]) 
        
        low_count = sum(1 for num in white_balls if LOW_NUMBER_MAX >= num >= 1)
        high_count = sum(1 for num in white_balls if HIGH_NUMBER_MIN <= num <= GLOBAL_WHITE_BALL_RANGE[1])
        
        odd_count = sum(1 for num in white_balls if num % 2 != 0)
        even_count = 5 - odd_count # Correct calculation for even count

        current_sum = sum(white_balls)

        group_a_to_use = group_a_numbers_def if group_a_numbers_def is not None else group_a
        current_group_a_count = sum(1 for num in white_balls if num in group_a_to_use)

        consecutive_present = False
        for i in range(len(white_balls) - 1):
            if white_balls[i] + 1 == white_balls[i+1]:
                consecutive_present = True
                break

        weekday_stats[day_name]['total_draws'] += 1
        weekday_stats[day_name]['total_low_balls'] += low_count
        weekday_stats[day_name]['total_high_balls'] += high_count
        weekday_stats[day_name]['total_odd_balls'] += odd_count
        weekday_stats[day_name]['total_even_balls'] += even_count
        weekday_stats[day_name]['total_sum'] += current_sum
        weekday_stats[day_name]['total_group_a_balls'] += current_group_a_count
        if consecutive_present:
            weekday_stats[day_name]['consecutive_draws_count'] += 1
        
        low_high_split_key = f"{low_count} Low / {high_count} High"
        weekday_stats[day_name]['low_high_splits'][low_high_split_key] += 1

        odd_even_split_key = f"{odd_count} Odd / {even_count} Even"
        weekday_stats[day_name]['odd_even_splits'][odd_even_split_key] += 1
    
    final_results = {}
    for day in POWERBALL_DRAW_DAYS: 
        if day in weekday_stats and weekday_stats[day]['total_draws'] > 0:
            data = weekday_stats[day]
            total_draws = data['total_draws']

            final_results[day] = {
                'total_draws': total_draws,
                'avg_low_balls': round(data['total_low_balls'] / total_draws, 2),
                'avg_high_balls': round(data['total_high_balls'] / total_draws, 2),
                'avg_odd_balls': round(data['total_odd_balls'] / total_draws, 2),
                'avg_even_balls': round(data['total_even_balls'] / total_draws, 2),
                'avg_sum': round(data['total_sum'] / total_draws, 2),
                'avg_group_a_balls': round(data['total_group_a_balls'] / total_draws, 2),
                'consecutive_present_percentage': round((data['consecutive_draws_count'] / total_draws) * 100, 2),
                'low_high_splits': sorted([{'split': k, 'count': v} for k, v in data['low_high_splits'].items()], key=lambda item: (-item['count'], item['split'])),
                'odd_even_splits': sorted([{'split': k, 'count': v} for k, v in data['odd_even_splits'].items()], key=lambda item: (-item['count'], item['split']))
            }
        else:
            final_results[day] = { 
                'total_draws': 0, 'avg_low_balls': 0.0, 'avg_high_balls': 0.0,
                'avg_odd_balls': 0.0, 'avg_even_balls': 0.0, 'avg_sum': 0.0,
                'avg_group_a_balls': 0.0, 'consecutive_present_percentage': 0.0,
                'low_high_splits': [], 'odd_even_splits': []
            }
            
    return final_results

def _get_yearly_patterns_for_range(df_source, selected_range_name):
    """
    Analyzes grouped patterns (pairs and triplets) within a specific number range,
    grouped by year. Returns data structured for yearly comparison in the frontend.
    """
    if df_source.empty:
        return []

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])
    
    if df_copy.empty:
        return []

    for i in range(1, 6):
        col = f'Number {i}'
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)
        else:
            pass

    # Filter for the selected range only
    min_val, max_val = NUMBER_RANGES.get(selected_range_name, (1, 69))

    yearly_data_structured = []
    
    for year in sorted(df_copy['Draw Date_dt'].dt.year.unique()):
        yearly_df = df_copy[df_copy['Draw Date_dt'].dt.year == year]
        
        year_pairs = defaultdict(int)
        year_triplets = defaultdict(int)

        for _, row in yearly_df.iterrows():
            white_balls = [int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])]
            
            numbers_in_current_range = sorted([num for num in white_balls if min_val <= num <= max_val])
            
            if len(numbers_in_current_range) >= 2:
                for pair in combinations(numbers_in_current_range, 2):
                    year_pairs[tuple(sorted(pair))] += 1
            
            if len(numbers_in_current_range) >= 3:
                for triplet_combo in combinations(numbers_in_current_range, 3):
                    year_triplets[tuple(sorted(triplet_combo))] += 1
        
        # Convert defaultdicts to lists of dicts for JSON serialization
        formatted_pairs = [{'pattern': list(p), 'count': c} for p, c in year_pairs.items()]
        formatted_triplets = [{'pattern': list(t), 'count': c} for t, c in year_triplets.items()]

        yearly_data_structured.append({
            "year": int(year),
            "pairs": formatted_pairs,
            "triplets": formatted_triplets,
            "total_unique_patterns": len(formatted_pairs) + len(formatted_triplets)
        })

    # Sort the overall yearly data by year
    yearly_data_structured.sort(key=lambda x: x['year'])
    
    return yearly_data_structured

def get_boundary_crossing_pairs_trends(df_source, selected_pair_tuple=None):
    if df_source.empty:
        return {
            'all_boundary_patterns_summary': [],
            'yearly_data_for_selected_pattern': [],
            'all_years_in_data': []
        }

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])
    
    if df_copy.empty:
        return {
            'all_boundary_patterns_summary': [],
            'yearly_data_for_selected_pattern': [],
            'all_years_in_data': []
        }

    for i in range(1, 6):
        col = f'Number {i}'
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)

    current_year = datetime.now().year
    # Dynamically set start_year for a rolling 10-year window
    start_year = max(2017, current_year - 9) 
    all_years = sorted([y for y in df_copy['Draw Date_dt'].dt.year.unique().tolist() if y >= start_year])


    yearly_boundary_pair_counts = defaultdict(lambda: defaultdict(int)) 
    overall_boundary_pair_counts = defaultdict(int) 

    for _, row in df_copy.iterrows():
        year = row['Draw Date_dt'].year
        if year < start_year: # Filter out years outside the rolling window
            continue
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])])
        current_draw_pairs = set(combinations(white_balls, 2))
        
        for bp in BOUNDARY_PAIRS_TO_ANALYZE:
            if bp in current_draw_pairs:
                yearly_boundary_pair_counts[year][bp] += 1
                overall_boundary_pair_counts[bp] += 1
    
    all_boundary_patterns_summary = sorted([
        {'pattern': list(p), 'total_count': c} for p, c in overall_boundary_pair_counts.items()
    ], key=lambda x: (-x['total_count'], str(x['pattern'])))

    yearly_data_for_selected_pattern = []
    if selected_pair_tuple:
        for year in all_years:
            count = yearly_boundary_pair_counts[year].get(selected_pair_tuple, 0)
            yearly_data_for_selected_pattern.append({'year': int(year), 'count': int(count)})
        yearly_data_for_selected_pattern.sort(key=lambda x: x['year']) 

    return {
        'all_boundary_patterns_summary': all_boundary_patterns_summary,
        'yearly_data_for_selected_pattern': yearly_data_for_selected_pattern,
        'all_years_in_data': all_years
    }


def get_special_patterns_analysis(df_source):
    """
    Analyzes historical draws for special number patterns (tens-apart, same last digit,
    repeating digit) and returns their counts per year,
    and also recent trends (last 12 months) indicating pattern presence per draw.

    Args:
        df_source (pd.DataFrame): The historical Powerball draw data, expected to have
                                  'Draw Date_dt' (datetime) and 'Number 1' through 'Number 5'.

    Returns:
        dict: A dictionary containing:
              - 'yearly_data': A list of dictionaries, each representing a year's pattern analysis.
              - 'recent_trends': A list of dictionaries for draws in the last 12 months,
                                 indicating if each pattern type was present.
              Example:
              {
                  'yearly_data': [
                      {
                          'year': 2025,
                          'total_draws': 50,
                          'tens_apart_patterns': [{'pattern': [10, 20], 'count': 5}, ...],
                          'same_last_digit_patterns': [{'pattern': [1, 11], 'count': 3}, ...],
                          'repeating_digit_patterns': [{'pattern': [11, 22], 'count': 2}, ...]
                      },
                      ...
                  ],
                  'recent_trends': [
                      {'draw_date': '2024-07-31', 'tens_apart': 'Yes', 'same_last_digit': 'No', 'repeating_digit': 'Yes'},
                      ...
                  ]
              }
    """
    if df_source.empty:
        return {'yearly_data': [], 'recent_trends': []}

    # Get unique years from the data, limited from 2017 to current year
    current_year = datetime.now().year
    years_in_data = sorted([y for y in df_source['Draw Date_dt'].dt.year.unique() if y >= 2017 and y <= current_year])

    all_yearly_patterns_data = []
    recent_special_trends = []

    # Pre-calculate all possible tens-apart pairs for efficiency
    all_tens_apart_pairs = set()
    for n1 in range(1, 60):
        for diff in [10, 20, 30, 40, 50]:
            n2 = n1 + diff
            if n2 <= 69:
                all_tens_apart_pairs.add(tuple(sorted((n1, n2))))

    # Pre-group numbers by last digit
    same_last_digit_groups_full = defaultdict(list)
    for i in range(1, 70):
        last_digit = i % 10
        same_last_digit_groups_full[last_digit].append(i)
    
    # Pre-define repeating digit numbers
    repeating_digit_numbers = [11, 22, 33, 44, 55, 66]

    # --- Yearly Data Calculation ---
    for year in years_in_data:
        yearly_df = df_source[df_source['Draw Date_dt'].dt.year == year].copy()
        
        if yearly_df.empty:
            all_yearly_patterns_data.append({
                'year': int(year),
                'total_draws': 0,
                'tens_apart_patterns': [],
                'same_last_digit_patterns': [],
                'repeating_digit_patterns': []
            })
            continue

        tens_apart_counts = defaultdict(int)
        same_last_digit_counts = defaultdict(int)
        repeating_digit_counts = defaultdict(int)
        
        total_draws_in_year = len(yearly_df)

        for idx, row in yearly_df.iterrows():
            white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])])
            white_ball_set = set(white_balls)
            
            # Tens-Apart
            for pair in combinations(white_balls, 2):
                sorted_pair = tuple(sorted(pair))
                if sorted_pair in all_tens_apart_pairs:
                    tens_apart_counts[sorted_pair] += 1

            # Same Last Digit
            for last_digit, full_group_numbers in same_last_digit_groups_full.items():
                intersection_with_draw = white_ball_set.intersection(set(full_group_numbers))
                if len(intersection_with_draw) >= 2:
                    for r in range(2, len(intersection_with_draw) + 1):
                        for pattern_combo in combinations(sorted(list(intersection_with_draw)), r):
                            same_last_digit_counts[pattern_combo] += 1

            # Repeating Digit
            drawn_repeating_digits = [n for n in repeating_digit_numbers if n in white_ball_set]
            if len(drawn_repeating_digits) >= 2:
                for r in range(2, len(drawn_repeating_digits) + 1):
                    for pattern_combo in combinations(sorted(drawn_repeating_digits), r):
                        repeating_digit_counts[pattern_combo] += 1

        formatted_tens_apart = [{'pattern': list(p), 'count': c} for p, c in tens_apart_counts.items()]
        formatted_tens_apart.sort(key=lambda x: (-x['count'], str(x['pattern'])))

        formatted_same_last_digit = [{'pattern': list(p), 'count': c} for p, c in same_last_digit_counts.items()]
        formatted_same_last_digit.sort(key=lambda x: (-x['count'], str(x['pattern'])))

        formatted_repeating_digit = [{'pattern': list(p), 'count': c} for p, c in repeating_digit_counts.items()]
        formatted_repeating_digit.sort(key=lambda x: (-x['count'], str(x['pattern'])))

        all_yearly_patterns_data.append({
            'year': int(year),
            'total_draws': total_draws_in_year,
            'tens_apart_patterns': formatted_tens_apart,
            'same_last_digit_patterns': formatted_same_last_digit,
            'repeating_digit_patterns': formatted_repeating_digit
        })
    
    all_yearly_patterns_data.sort(key=lambda x: x['year'], reverse=True)

    # --- Recent Trends (Last 12 Months) Calculation ---
    one_year_ago = datetime.now() - timedelta(days=365)
    recent_data_df = df_source[df_source['Draw Date_dt'] >= one_year_ago].copy()
    recent_data_df = recent_data_df.sort_values(by='Draw Date_dt', ascending=False) # Sort newest first

    for idx, row in recent_data_df.iterrows():
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])])
        white_ball_set = set(white_balls)
        draw_date_str = row['Draw Date_dt'].strftime('%Y-%m-%d')

        tens_apart_present = "No"
        for pair in combinations(white_balls, 2):
            if tuple(sorted(pair)) in all_tens_apart_pairs:
                tens_apart_present = "Yes"
                break
        
        same_last_digit_present = "No"
        for last_digit, full_group_numbers in same_last_digit_groups_full.items():
            intersection_with_draw = white_ball_set.intersection(set(full_group_numbers))
            if len(intersection_with_draw) >= 2:
                same_last_digit_present = "Yes"
                break
        
        repeating_digit_present = "No"
        drawn_repeating_digits = [n for n in repeating_digit_numbers if n in white_ball_set]
        if len(drawn_repeating_digits) >= 2:
            repeating_digit_present = "Yes"

        recent_special_trends.append({
            'draw_date': draw_date_str,
            'tens_apart': tens_apart_present,
            'same_last_digit': same_last_digit_present,
            'repeating_digit': repeating_digit_present
        })

    return {
        'yearly_data': all_yearly_patterns_data,
        'recent_trends': recent_special_trends
    }

def get_white_ball_frequency_by_period(df_source, period_type='year'):
    """
    Calculates the frequency of each white ball (1-69) per specified period (year, half_year, quarter).
    Defaults to a rolling 10-year window.
    """
    if df_source.empty:
        return {}, []

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])

    if df_copy.empty:
        return {}, []

    current_year = datetime.now().year
    # Dynamically set start_year for a rolling 10-year window (current year + 9 previous years)
    start_year = max(2017, current_year - 9) 
    years_to_analyze = range(start_year, current_year + 1)

    # Filter data to only include relevant years
    df_filtered_years = df_copy[df_copy['Draw Date_dt'].dt.year.isin(years_to_analyze)].copy()

    # Determine period_label based on period_type
    if period_type == 'year':
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str)
    elif period_type == 'half_year':
        df_filtered_years['half'] = (df_filtered_years['Draw Date_dt'].dt.month - 1) // 6 + 1
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str) + ' H' + df_filtered_years['half'].astype(str)
    elif period_type == 'quarter':
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str) + ' Q' + df_filtered_years['Draw Date_dt'].dt.quarter.astype(str)
    else:
        # Default to year if an invalid period_type is provided
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str)
        period_type = 'year' # Reset period_type to 'year' for consistent behavior

    # Get all unique period labels in chronological order
    all_period_labels = sorted(df_filtered_years['period_label'].unique().tolist())

    # Initialize a nested dictionary for all white balls and all periods
    period_freq_data = {wb: {label: 0 for label in all_period_labels} for wb in range(1, 70)}

    # Iterate through each draw and populate frequencies
    for _, row in df_filtered_years.iterrows():
        period_label = row['period_label']
        white_balls = [int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])]
        for wb in white_balls:
            if 1 <= wb <= 69: # Ensure number is within valid range
                period_freq_data[wb][period_label] += 1
    
    # Format the data for JSON/template
    formatted_data = {}
    for wb_num, period_counts in period_freq_data.items():
        formatted_data[wb_num] = sorted([
            {'period_label': label, 'frequency': count} 
            for label, count in period_counts.items()
        ], key=lambda x: x['period_label']) # Sort by period label for chronological order
        
    return formatted_data, all_period_labels # Also return all_period_labels for frontend chart axes

def get_consecutive_numbers_yearly_trends(df_source):
    """
    Calculates the percentage of draws containing consecutive numbers for each year
    within a rolling 10-year window.
    Also generates a flat list of all unique consecutive sequences found in the data,
    with their total counts and all associated draw dates.
    """
    if df_source.empty:
        return {'yearly_data': [], 'years': [], 'all_consecutive_pairs_flat': []}

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])
    
    if df_copy.empty:
        return {'yearly_data': [], 'years': [], 'all_consecutive_pairs_flat': []}

    current_year = datetime.now().year
    start_year = max(2017, current_year - 9) # Rolling 10-year window
    years_to_analyze = range(start_year, current_year + 1)

    yearly_trends = []
    # This will store {sequence_tuple: {'count': X, 'dates': [date1, date2, ...]}}
    all_consecutive_sequences_aggregated = defaultdict(lambda: {'count': 0, 'dates': []}) 

    for year in years_to_analyze:
        yearly_df = df_copy[df_copy['Draw Date_dt'].dt.year == year].copy()
        
        total_draws_in_year = len(yearly_df)
        consecutive_draws_count = 0
        
        if total_draws_in_year > 0:
            for _, row in yearly_df.iterrows():
                white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])])
                draw_date_str = row['Draw Date_dt'].strftime('%Y-%m-%d')
                
                current_draw_consecutive_sequences = _find_consecutive_sequences(white_balls)
                
                if current_draw_consecutive_sequences:
                    consecutive_draws_count += 1
                    for sequence in current_draw_consecutive_sequences: # Iterate through sequences
                        sequence_tuple = tuple(sequence) # Use tuple for dict key
                        all_consecutive_sequences_aggregated[sequence_tuple]['count'] += 1
                        all_consecutive_sequences_aggregated[sequence_tuple]['dates'].append(draw_date_str)
                        
            percentage = round((consecutive_draws_count / total_draws_in_year) * 100, 2)
        else:
            percentage = 0.0
        
        yearly_trends.append({
            'year': int(year),
            'percentage': percentage,
            'total_draws': total_draws_in_year,
            'consecutive_draws': consecutive_draws_count
        })

    # Convert the aggregated dictionary to a flat list of dictionaries
    flat_consecutive_sequences_list = []
    for sequence_tuple, data in all_consecutive_sequences_aggregated.items(): 
        flat_consecutive_sequences_list.append({
            'sequence': list(sequence_tuple), 
            'count': data['count'],
            'dates': sorted(list(set(data['dates'])), reverse=True) # Deduplicate and sort dates descending
        })
    
    # Sort the flat list by count (descending) then by sequence (ascending)
    flat_consecutive_sequences_list.sort(key=lambda x: (-x['count'], x['sequence'])) 
    
    yearly_trends.sort(key=lambda x: x['year'])
    
    return {
        'yearly_data': yearly_trends, 
        'years': list(years_to_analyze), 
        'all_consecutive_pairs_flat': flat_consecutive_sequences_list # Also corrected return key
    }

def get_powerball_position_frequency(df_source):
    if df_source.empty:
        return {}
    
    position_freq = defaultdict(lambda: defaultdict(int))

    for _, row in df_source.iterrows():
        powerball = int(row['Powerball'])
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6)])
        
        # Determine the "position" of the Powerball relative to the white balls
        # This is a conceptual position, not an actual draw position.
        # For simplicity, we can categorize it into ranges or relative to the white balls.
        # For now, let's just count its frequency. If a more complex "position" is needed,
        # we'd need a more specific definition.
        
        # Example: Is Powerball lower than all white balls?
        if white_balls and powerball < white_balls[0]:
            position_freq[powerball]['Lower than all WB'] += 1
        # Is Powerball higher than the highest white ball?
        elif white_balls and powerball > white_balls[-1]:
            position_freq[powerball]['Higher than all WB'] += 1
        else:
            position_freq[powerball]['Within WB Range'] += 1 # This needs refinement for real positional analysis
        
        # For now, let's just return the raw frequency of the Powerball
        # as the concept of "position" without more specific criteria is ambiguous.
        # If the user wants "Powerball frequency by its value", the existing powerball_freq
        # function is better.
        # If they want "Powerball relative to white balls", we need specific categories.
        
        # Let's assume for this function, they want the raw frequency of Powerball numbers
        # without complex positional logic, as the name might imply.
        # This function name is a bit misleading given its current implementation.
        # I'll provide a basic frequency for now.
        position_freq[powerball]['Total Draws'] += 1

    formatted_data = []
    for pb_num in sorted(position_freq.keys()):
        total_draws = position_freq[pb_num]['Total Draws']
        formatted_data.append({
            'Powerball': int(pb_num),
            'Total Draws': int(total_draws)
            # Add more specific positional data if defined later
        })
    return formatted_data


def initialize_core_data():
    global df, last_draw, historical_white_ball_sets, white_ball_co_occurrence_lookup
    print("Attempting to load core historical data...")
    try:
        df_temp = load_historical_data_from_supabase()
        if not df_temp.empty:
            df = df_temp
            last_draw = get_last_draw(df)
            if not last_draw.empty and 'Draw Date_dt' in last_draw and pd.notna(last_draw['Draw Date_dt']):
                last_draw['Draw Date'] = last_draw['Draw Date_dt'].strftime('%Y-%m-%d')
            else:
                 last_draw['Draw Date'] = 'N/A' 
            
            historical_white_ball_sets.clear() 
            white_ball_co_occurrence_lookup.clear() 
            for _, row in df.iterrows():
                white_balls_tuple = tuple(sorted([
                    int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), 
                    int(row['Number 4']), int(row['Number 5'])
                ]))
                frozenset_white_balls = frozenset(white_balls_tuple)
                historical_white_ball_sets.add(frozenset_white_balls)
                
                current_draw_date = row['Draw Date']
                if frozenset_white_balls not in white_ball_co_occurrence_lookup or \
                   datetime.strptime(current_draw_date, '%Y-%m-%d') > datetime.strptime(white_ball_co_occurrence_lookup[frozenset_white_balls], '%Y-%m-%d'):
                    white_ball_co_occurrence_lookup[frozenset_white_balls] = current_draw_date

            print("Core historical data loaded successfully and co-occurrence lookup populated.")
        else:
            print("Core historical data is empty after loading. df remains empty.")
            last_draw = pd.Series({
                'Draw Date': 'N/A', 'Number 1': 'N/A', 'Number 2': 'N/A',
                'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A',
                'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A']
            }, dtype='object')
    except Exception as e:
        print(f"An error occurred during initial core data loading: {e}")
        traceback.print_exc()


def get_cached_analysis(key, compute_function, *args, **kwargs):
    global analysis_cache, last_analysis_cache_update
    
    # Filter out DataFrame objects from args/kwargs for JSON serialization in cache key
    # The actual compute_function will still receive all original args/kwargs
    serializable_args = [arg for arg in args if not isinstance(arg, pd.DataFrame)]
    serializable_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, pd.DataFrame)}

    cache_key_full = f"{key}_{json.dumps(serializable_args)}_{json.dumps(serializable_kwargs)}"

    if cache_key_full in analysis_cache and (datetime.now() - last_analysis_cache_update).total_seconds() < CACHE_EXPIRATION_SECONDS:
        print(f"Serving '{cache_key_full}' from cache.")
        return analysis_cache[cache_key_full]
    
    print(f"Computing and caching '{cache_key_full}'.")
    computed_data = compute_function(*args, **kwargs) # Pass original args/kwargs to the compute function
    
    analysis_cache[cache_key_full] = computed_data
    last_analysis_cache_update = datetime.now()
    return computed_data

def invalidate_analysis_cache():
    global analysis_cache, last_analysis_cache_update
    analysis_cache = {}
    last_analysis_cache_update = datetime.min
    print("Analysis cache invalidated.")

def _summarize_for_ai(df_source):
    if df_source.empty:
        return "No historical data available for detailed analysis. Please ensure the database is populated."

    summary_parts = []

    # 1. Overall Frequency
    white_ball_freq_list, powerball_freq_list = frequency_analysis(df_source)
    top_wb_freq = sorted(white_ball_freq_list, key=lambda x: x['Frequency'], reverse=True)[:10]
    top_pb_freq = sorted(powerball_freq_list, key=lambda x: x['Frequency'], reverse=True)[:5]
    summary_parts.append("Overall Most Frequent White Balls: " + ", ".join([f"{n['Number']} ({n['Frequency']} times)" for n in top_wb_freq]))
    summary_parts.append("Overall Most Frequent Powerballs: " + ", ".join([f"{n['Number']} ({n['Frequency']} times)" for n in top_pb_freq]))

    # 2. Hot/Cold Numbers (Last Year)
    last_draw_date_str = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else datetime.now().strftime('%Y-%m-%d')
    hot_nums, cold_nums = hot_cold_numbers(df_source, last_draw_date_str)
    if hot_nums:
        summary_parts.append("Hot Numbers (most frequent in last year): " + ", ".join([f"{n['Number']} ({n['Frequency']} times)" for n in hot_nums[:10]])) 
    if cold_nums:
        summary_parts.append("Cold Numbers (least frequent in last year): " + ", ".join([f"{n['Number']} (missed {n['Frequency']} draws)" for n in cold_nums[:10]])) # Changed to 'missed draws' for cold

    # 3. Co-occurring Pairs and Triplets
    co_occurrence_data, _ = get_co_occurrence_matrix(df_source)
    sorted_co_occurrence = sorted(co_occurrence_data, key=lambda x: x['count'], reverse=True)
    if sorted_co_occurrence:
        summary_parts.append("Top 10 Co-occurring Pairs (most frequent overall): " + ", ".join([f"({p['x']}, {p['y']}) - {p['count']} times" for p in sorted_co_occurrence[:10]]))

    triplets_data = get_most_frequent_triplets(df_source)
    if triplets_data:
        summary_parts.append("Top 5 Co-occurring Triplets (most frequent overall): " + ", ".join([f"({', '.join(map(str, t['triplet']))}) - {t['count']} times" for t in triplets_data[:5]]))

    # 4. Number Age (Miss Streak)
    _, detailed_ages = get_number_age_distribution(df_source) 
    white_ball_ages = sorted([d for d in detailed_ages if d['type'] == 'White Ball'], key=lambda x: x['age'], reverse=True)
    powerball_ages = sorted([d for d in detailed_ages if d['type'] == 'Powerball'], key=lambda x: x['age'], reverse=True)
    
    if white_ball_ages:
        summary_parts.append("White Balls with longest 'miss streak' (coldest by age): " + ", ".join([f"{n['number']} (missed {n['age']} draws)" for n in white_ball_ages[:10]]))
    if powerball_ages:
        summary_parts.append("Powerballs with longest 'miss streak' (coldest by age): " + ", ".join([f"{n['number']} (missed {n['age']} draws)" for n in powerball_ages[:5]])) 

    # 5. Monthly Trends (Recent Activity)
    # Using the new flexible function for monthly trends for AI summary
    # Call without start_year, so it uses the 10-year rolling window
    monthly_trends_data, _ = get_white_ball_frequency_by_period(df_source, period_type='half_year') 
    
    recent_monthly_numbers_wb = defaultdict(int)
    for wb_num, periods_data in monthly_trends_data.items():
        for period_info in periods_data:
            recent_monthly_numbers_wb[wb_num] += period_info['frequency']

    sorted_recent_monthly_wb = sorted(recent_monthly_numbers_wb.items(), key=lambda item: item[1], reverse=True)
    if sorted_recent_monthly_wb:
        summary_parts.append("White Balls frequently drawn in recent half-years: " + ", ".join([f"{n} ({c} times)" for n, c in sorted_recent_monthly_wb[:15]])) 

    # 6. Odd/Even Split Trends
    odd_even_trends_list = get_odd_even_split_trends(df_source, last_draw_date_str) 
    overall_odd_even_counts = defaultdict(int)
    for draw_detail in odd_even_trends_list:
        overall_odd_even_counts[draw_detail['split_category']] += 1
    
    most_common_odd_even = sorted(overall_odd_even_counts.items(), key=lambda item: item[1], reverse=True)
    if most_common_odd_even:
        summary_parts.append("Most common Odd/Even splits in recent draws (last 6 months): " + ", ".join([f"{s} ({c} times)" for s, c in most_common_odd_even[:3]])) 

    # 7. Sum Range Trends
    sum_data = get_sum_trends_and_gaps_data(df_source)
    if sum_data['grouped_sums_analysis']:
        sum_range_summaries = []
        for range_name, data in sum_data['grouped_sums_analysis'].items():
            if data['most_frequent_sums_in_range']:
                top_sums_str = ", ".join([f"{s['sum']} ({s['count']} times)" for s in data['most_frequent_sums_in_range']])
                sum_range_summaries.append(f"{range_name} (Top sums: {top_sums_str})")
        if sum_range_summaries:
            summary_parts.append("Key Sum Range Trends: " + "; ".join(sum_range_summaries))

    # 8. Consecutive Numbers Trends
    consecutive_trends_list = get_consecutive_numbers_trends(df_source, last_draw_date_str)
    consecutive_present_count = sum(1 for t in consecutive_trends_list if t['consecutive_present'] == 'Yes')
    if consecutive_trends_list:
        consecutive_percentage = (consecutive_present_count / len(consecutive_trends_list)) * 100
        summary_parts.append(f"Consecutive numbers appeared in {consecutive_percentage:.2f}% of recent draws (last 6 months).")

    return "\n".join(summary_parts)
    
# Function to get consecutive trends for a specific DataFrame (e.g., filtered by year)
def get_consecutive_trends_for_df(df_to_analyze):
    if df_to_analyze.empty:
        return []

    trend_data = []
    for idx, row in df_to_analyze.iterrows():
        # Ensure numbers are integers before sorting and finding pairs
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])])
        
        consecutive_sequences = _find_consecutive_sequences(white_balls)
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            'consecutive_present': "Yes" if consecutive_sequences else "No", 
            'consecutive_sequences': consecutive_sequences # Changed key to sequences
        })
    return trend_data

initialize_core_data() 


# --- Flask Routes (Ordered for Dependency - all UI-facing routes first, then API routes) ---

@app.route('/')
def index():
    last_draw_dict = last_draw.to_dict()
    return render_template('index.html', 
                           last_draw=last_draw_dict, 
                           sum_ranges=SUM_RANGES,
                           selected_odd_even_choice="Any", 
                           selected_sum_range="Any", 
                           num_sets_to_generate=1 
                          )

@app.route('/generate', methods=['POST'])
def generate():
    if df.empty:
        flash("Cannot generate numbers: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)

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

    selected_sum_range_label = request.form.get('sum_range_filter', 'Any')
    selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)

    num_sets_to_generate_str = request.form.get('num_sets_to_generate', '1')
    try:
        num_sets_to_generate = int(num_sets_to_generate_str)
        if not (1 <= num_sets_to_generate <= 10):
            flash("Number of sets to generate must be between 1 and 10.", 'error')
            num_sets_to_generate = 1 
    except ValueError:
        flash("Invalid number of sets. Please enter an integer.", 'error')
        num_sets_to_generate = 1 

    generated_sets = []
    last_draw_dates = {} 

    for i in range(num_sets_to_generate):
        try:
            white_balls, powerball = generate_powerball_numbers(
                df, group_a, odd_even_choice, combo_choice, white_ball_range_local, powerball_range_local, 
                excluded_numbers_local, high_low_balance, selected_sum_range_tuple, is_simulation=False
            )
            generated_sets.append({'white_balls': white_balls, 'powerball': powerball})
            
            if i == num_sets_to_generate - 1:
                last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        except ValueError as e:
            flash(f"Error generating set {i+1}: {str(e)}", 'error')
            break 
        except Exception as e:
            flash(f"An unexpected error occurred during generation of set {i+1}: {e}", 'error')
            break

    return render_template('index.html', 
                           generated_sets=generated_sets, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='generated',
                           sum_ranges=SUM_RANGES, 
                           selected_sum_range=selected_sum_range_label, 
                           selected_odd_even_choice=odd_even_choice, 
                           num_sets_to_generate=num_sets_to_generate 
                          )


@app.route('/generate_with_user_pair', methods=['POST'])
def generate_with_user_pair_route():
    if df.empty:
        flash("Cannot generate with provided pair: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)

    user_pair_str = request.form.get('user_pair')
    selected_sum_range_label = request.form.get('sum_range_filter_pair', 'Any') 
    selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)

    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        if not user_pair_str:
            raise ValueError("Please enter two numbers for the starting pair.")
        
        pair_parts = [int(num.strip()) for num in user_pair_str.split(',') if num.strip().isdigit()]
        if len(pair_parts) != 2:
            raise ValueError("Please enter exactly two numbers for the pair, separated by a comma (e.g., '18, 19').")
        
        num1, num2 = pair_parts

        white_balls, powerball = generate_with_user_provided_pair(
            num1, num2, GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, 
            excluded_numbers, df, selected_sum_range_tuple
        )
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        generated_sets = [{'white_balls': white_balls, 'powerball': powerball}]

        return render_template('index.html', 
                               generated_sets=generated_sets, 
                               powerball=powerball, 
                               last_draw=last_draw.to_dict(), 
                               last_draw_dates=last_draw_dates,
                               generation_type='user_pair',
                               sum_ranges=SUM_RANGES, 
                               selected_sum_range_pair=selected_sum_range_label) 
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)
    except Exception as e:
        flash(f"An unexpected error occurred during pair-based generation: {e}", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)


@app.route('/generate_group_a_strategy', methods=['POST'])
def generate_group_a_strategy_route():
    if df.empty:
        flash("Cannot generate numbers with Group A strategy: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)

    num_from_group_a = int(request.form.get('num_from_group_a'))
    white_ball_min = int(request.form.get('white_ball_min', 1))
    white_ball_max = int(request.form.get('white_ball_max', 69))
    white_ball_range_local = (white_ball_min, white_ball_max)
    powerball_min = int(request.form.get('powerball_min', 1))
    powerball_max = int(request.form.get('powerball_max', 26))
    powerball_range_local = (powerball_min, powerball_max)
    excluded_numbers_local = [int(num.strip()) for num in request.form.get('excluded_numbers', '').split(",") if num.strip().isdigit()] if request.form.get('excluded_numbers') else []
    
    selected_sum_range_label = request.form.get('sum_range_filter_group_a', 'Any') 
    selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)

    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        white_balls, powerball = generate_from_group_a(
            df, num_from_group_a, white_ball_range_local, powerball_range_local, 
            excluded_numbers_local, selected_sum_range_tuple
        )
        generated_sets = [{'white_balls': white_balls, 'powerball': powerball}]

        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)

    return render_template('index.html', 
                           generated_sets=generated_sets, 
                           powerball=powerball, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='group_a_strategy',
                           sum_ranges=SUM_RANGES, 
                           selected_sum_range_group_a=selected_sum_range_label)

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

# --- All other routes (unchanged from your provided index (11).py, or with minimal fixes) ---

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
        else:
            flash(message, 'error')

    except ValueError:
        flash("Invalid number format for saving generated numbers.", 'error')
    except Exception as e:
        flash(f"An error occurred while saving generated numbers: {e}", 'error')
    return redirect(url_for('index'))

# NEW ROUTE: Save multiple generated picks
@app.route('/save_multiple_generated_picks', methods=['POST'])
def save_multiple_generated_picks_route():
    try:
        picks_to_save = request.json.get('picks', [])
        
        if not picks_to_save:
            return jsonify({"success": False, "message": "No picks provided to save."}), 400

        saved_count = 0
        failed_count = 0
        messages = []

        for pick in picks_to_save:
            white_balls = pick.get('white_balls')
            powerball = pick.get('powerball')

            if not white_balls or len(white_balls) != 5 or powerball is None:
                messages.append(f"Skipping invalid pick: {pick}")
                failed_count += 1
                continue
            
            # Ensure numbers are integers and sorted for consistency
            try:
                white_balls = sorted([int(n) for n in white_balls])
                powerball = int(powerball)
            except ValueError:
                messages.append(f"Skipping pick due to invalid number format: {pick}")
                failed_count += 1
                continue

            success, message = save_generated_numbers_to_db(white_balls, powerball)
            if success:
                saved_count += 1
                messages.append(f"Saved: {', '.join(map(str, white_balls))} + {powerball}")
            else:
                failed_count += 1
                messages.append(f"Failed to save {', '.join(map(str, white_balls))} + {powerball}: {message}")
        
        status_message = f"Successfully saved {saved_count} pick(s). Failed to save {failed_count} pick(s)."
        return jsonify({"success": True, "message": status_message, "details": messages}), 200

    except Exception as e:
        print(f"Error in save_multiple_generated_picks_route: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/frequency_analysis')
def frequency_analysis_route():
    # Pass df directly to the function, not for cache key serialization
    white_ball_freq_list, powerball_freq_list = get_cached_analysis('freq_analysis', frequency_analysis, df)
    return render_template('frequency_analysis.html', 
                           white_ball_freq=white_ball_freq_list, 
                           powerball_freq=powerball_freq_list)

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    # Pass df directly to the function, not for cache key serialization
    hot_numbers_list, cold_numbers_list = get_cached_analysis('hot_cold_numbers', hot_cold_numbers, df, last_draw_date_str_for_cache)
    
    return render_template('hot_cold_numbers.html', 
                           hot_numbers=hot_numbers_list, 
                           cold_numbers=cold_numbers_list)

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    if df.empty:
        flash("Cannot perform monthly trends analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'

    # Pass df directly to the function, not for cache key serialization
    # Removed num_months_for_top_display from get_white_ball_frequency_by_period call
    monthly_trends_data = get_cached_analysis(
        'monthly_trends_and_streaks', 
        get_monthly_white_ball_analysis_data, 
        df, 
        num_top_wb=69, 
        num_top_pb=3 
    )
    
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_data=monthly_trends_data['monthly_data'], 
                           streak_numbers=monthly_trends_data['streak_numbers'])

@app.route('/sum_of_main_balls_analysis')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot display Sum of Main Balls Analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Pass df directly to the function, not for cache key serialization
    sums_data_df, sum_freq_list, min_sum, max_sum, avg_sum = get_cached_analysis('sum_of_main_balls_data', sum_of_main_balls, df)
    
    sums_data = sums_data_df.to_dict('records') 
    
    sum_freq_json = json.dumps(sum_freq_list)

    return render_template('sum_of_main_balls.html', 
                           sums_data=sums_data,
                           sum_freq_json=sum_freq_json,
                           min_sum=min_sum,
                           max_sum=max_sum,
                           avg_sum=avg_sum)

@app.route('/find_results_by_sum', methods=['GET', 'POST'])
def find_results_by_sum_route():
    if df.empty:
        flash("Cannot display Search by Sum: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    results = []
    target_sum_display = None
    selected_sort_by = request.args.get('sort_by', 'date_desc') 

    if request.method == 'POST':
        target_sum_str = request.form.get('target_sum')
        selected_sort_by = request.form.get('sort_by', 'date_desc') 
        
        if target_sum_str and target_sum_str.isdigit():
            target_sum = int(target_sum_str)
            target_sum_display = target_sum
            results_df_raw = find_results_by_sum(df, target_sum)

            if not results_df_raw.empty:
                if 'Draw Date_dt' not in results_df_raw.columns:
                    results_df_raw['Draw Date_dt'] = pd.to_datetime(results_df_raw['Draw Date'], errors='coerce')

                if selected_sort_by == 'date_desc':
                    results_df_raw = results_df_raw.sort_values(by='Draw Date_dt', ascending=False)
                elif selected_sort_by == 'date_asc':
                    results_df_raw = results_df_raw.sort_values(by='Draw Date_dt', ascending=True)
                elif selected_sort_by == 'balls_asc':
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
            results = [] 
            target_sum_display = None 
    
    return render_template('find_results_by_sum.html', 
                           results=results,
                           target_sum=target_sum_display,
                           selected_sort_by=selected_sort_by)

@app.route('/simulate_multiple_draws', methods=['GET', 'POST'])
def simulate_multiple_draws_route():
    if df.empty:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"error": "Historical data not loaded or is empty."}), 500
        else:
            flash("Cannot run simulation: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
            return redirect(url_for('index'))

    simulated_white_ball_freq_list = []
    simulated_powerball_freq_list = []
    num_draws_display = None
    odd_even_choice_display = "Any" 

    if request.method == 'POST':
        num_draws_str = request.form.get('num_draws')
        odd_even_choice_display = request.form.get('odd_even_choice', 'Any') 

        if num_draws_str and num_draws_str.isdigit():
            num_draws = int(num_draws_str)
            num_draws_display = num_draws
            
            sim_results = simulate_multiple_draws(
                df, group_a, odd_even_choice_display, 
                GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, 
                excluded_numbers, num_draws
            )
            
            simulated_white_ball_freq_list = sim_results['white_ball_freq']
            simulated_powerball_freq_list = sim_results['powerball_freq']
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": "Please enter a valid number for Number of Simulations."}), 400
            else:
                flash("Please enter a valid number for Number of Simulations.", 'error')


        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # This is an AJAX request, return JSON
            return jsonify({
                "simulated_white_ball_freq": simulated_white_ball_freq_list,
                "simulated_powerball_freq": simulated_powerball_freq_list,
                "num_simulations": num_draws_display
            })
        else:
            # Not an AJAX request, render the full template
            return render_template('simulate_multiple_draws.html', 
                                simulated_white_ball_freq=simulated_white_ball_freq_list, 
                                simulated_powerball_freq=simulated_powerball_freq_list,   
                                num_simulations=num_draws_display,
                                selected_odd_even_choice=odd_even_choice_display)

    # For GET requests, render the initial template
    return render_template('simulate_multiple_draws.html', 
                           simulated_white_ball_freq=[], # Start with empty data for GET
                           simulated_powerball_freq=[],   # Start with empty data for GET
                           num_simulations=100, # Default value for display
                           selected_odd_even_choice="Any")

# NEW API endpoint for generating a single draw for animation purposes
@app.route('/api/generate_single_draw', methods=['GET'])
def generate_single_draw_api():
    try:
        # Generate one random set quickly without complex validation
        white_balls = sorted(random.sample(range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1), 5))
        powerball = random.randint(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1])
        return jsonify({'success': True, 'white_balls': white_balls, 'powerball': powerball})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/number_age_distribution')
def number_age_distribution_route():
    # Pass df directly to the function, not for cache key serialization
    number_age_counts, detailed_number_ages = get_cached_analysis('number_age_distribution', get_number_age_distribution, df)
    return render_template('number_age_distribution.html',
                           number_age_data=number_age_counts,
                           detailed_number_ages=detailed_number_ages)

@app.route('/co_occurrence_analysis')
def co_occurrence_analysis_route():
    # Pass df directly to the function, not for cache key serialization
    co_occurrence_data, max_co_occurrence = get_cached_analysis('co_occurrence_analysis', get_co_occurrence_matrix, df)
    return render_template('co_occurrence_analysis.html',
                           co_occurrence_data=co_occurrence_data,
                           max_co_occurrence=max_co_occurrence)

@app.route('/powerball_position_frequency')
def powerball_position_frequency_route():
    # Pass df directly to the function, not for cache key serialization
    powerball_position_data = get_cached_analysis('powerball_position_frequency', get_powerball_position_frequency, df)
    return render_template('powerball_position_frequency.html',
                           powerball_position_data=powerball_position_data)

@app.route('/powerball_frequency_by_year')
def powerball_frequency_by_year_route():
    # Pass df directly to the function, not for cache key serialization
    yearly_pb_freq_data, years = get_cached_analysis('yearly_pb_freq', get_powerball_frequency_by_year, df)
    return render_template('powerball_frequency_by_year.html',
                           yearly_pb_freq_data=yearly_pb_freq_data,
                           years=years)

@app.route('/odd_even_trends')
def odd_even_trends_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    # Pass df directly to the function, not for cache key serialization
    odd_even_trends = get_cached_analysis('odd_even_trends', get_odd_even_split_trends, df, last_draw_date_str_for_cache)
    return render_template('odd_even_trends.html',
                           odd_even_trends=odd_even_trends)

@app.route('/consecutive_trends')
def consecutive_trends_route():
    if df.empty:
        flash("Cannot display Consecutive Trends: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    
    # Existing recent trends data
    consecutive_trends = get_cached_analysis('consecutive_trends', get_consecutive_numbers_trends, df, last_draw_date_str_for_cache)
    
    # New yearly trends data and years for dropdown
    yearly_consecutive_data_full = get_cached_analysis('consecutive_yearly_trends', get_consecutive_numbers_yearly_trends, df)
    
    # Extract data for template
    yearly_consecutive_percentage_data = yearly_consecutive_data_full['yearly_data']
    years_for_dropdown = yearly_consecutive_data_full['years']
    all_consecutive_pairs_flat = yearly_consecutive_data_full['all_consecutive_pairs_flat']

    return render_template('consecutive_trends.html',
                           consecutive_trends=consecutive_trends,
                           yearly_consecutive_percentage_data=yearly_consecutive_percentage_data,
                           years_for_dropdown=years_for_dropdown,
                           all_consecutive_pairs_flat=all_consecutive_pairs_flat)

# NEW API endpoint for yearly consecutive trends (now returns flat list)
@app.route('/api/consecutive_yearly_trends')
def api_consecutive_yearly_trends_route():
    if df.empty:
        return jsonify({"error": "Historical data not loaded or is empty."}), 500
    
    yearly_consecutive_data_full = get_cached_analysis('consecutive_yearly_trends', get_consecutive_numbers_yearly_trends, df)
    
    return jsonify({
        'data': yearly_consecutive_data_full['yearly_data'],
        'years': yearly_consecutive_data_full['years'],
        'all_consecutive_pairs_flat': yearly_consecutive_data_full['all_consecutive_pairs_flat']
    })
# Add this new function to your index.py file
@app.route('/consecutive_trends_by_year/<int:year>')
def consecutive_trends_by_year(year):
    # Assuming 'df' is your global DataFrame of historical data
    df_year = df[(df['Draw Date_dt'].dt.year == year)].copy()

    # Call your existing function to calculate the consecutive trends for the filtered data
    trends = get_consecutive_trends_for_df(df_year) # Corrected function call

    # Return the data as a JSON object
    return jsonify(trends)

@app.route('/triplets_analysis')
def triplets_analysis_route():
    # Pass df directly to the function, not for cache key serialization
    triplets_data = get_cached_analysis('triplets_analysis', get_most_frequent_triplets, df) 
    return render_template('triplets_analysis.html',
                           triplets_data=triplets_data)

@app.route('/grouped_patterns_analysis')
def grouped_patterns_analysis_route():
    # Pass df directly to the function, not for cache key serialization
    patterns_data = get_cached_analysis('grouped_patterns', get_grouped_patterns_over_years, df)
    return render_template('grouped_patterns_analysis.html', patterns_data=patterns_data)

@app.route('/grouped_patterns_yearly_comparison', methods=['GET', 'POST'])
def grouped_patterns_yearly_comparison_route():
    if df.empty:
        flash("Cannot display Grouped Patterns Yearly Comparison: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    selected_range_label = request.form.get('selected_range', '20s') 
    
    if selected_range_label not in NUMBER_RANGES:
        flash(f"Invalid number range selected: {selected_range_label}. Displaying data for default range '20s'.", 'error')
        selected_range_label = '20s' 

    cache_key = f'yearly_patterns_{selected_range_label}'
    
    # Pass df directly to the function, not for cache key serialization
    yearly_patterns_data = get_cached_analysis(
        cache_key,
        _get_yearly_patterns_for_range,
        df,
        selected_range_label
    )

    return render_template('grouped_patterns_yearly_comparison.html',
                           yearly_patterns_data=yearly_patterns_data,
                           number_ranges=NUMBER_RANGES, 
                           selected_range=selected_range_label) 

@app.route('/boundary_crossing_pairs_trends', methods=['GET', 'POST'])
def boundary_crossing_pairs_trends_route():
    if df.empty:
        flash("Cannot display Boundary Crossing Pairs Trends: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    selected_pair = request.form.get('selected_pair') 
    selected_pair_tuple = None
    if selected_pair:
        try:
            parts = [int(p.strip()) for p in selected_pair.split(',')]
            if len(parts) == 2:
                selected_pair_tuple = tuple(sorted(parts))
        except ValueError:
            flash("Invalid pair format. Please select a valid pair from the dropdown.", 'error')
            selected_pair = None 

    # Pass df directly to the function, not for cache key serialization
    boundary_trends_data = get_cached_analysis(
        f'boundary_crossing_trends_{selected_pair}', 
        get_boundary_crossing_pairs_trends, 
        df, 
        selected_pair_tuple
    )
    
    all_boundary_patterns_summary = boundary_trends_data['all_boundary_patterns_summary']
    yearly_data_for_selected_pattern = boundary_trends_data['yearly_data_for_selected_pattern']
    all_years_in_data = boundary_trends_data['all_years_in_data']

    boundary_pairs_for_dropdown = [f"{p[0]}, {p[1]}" for p in BOUNDARY_PAIRS_TO_ANALYZE]

    return render_template('boundary_crossing_pairs_trends.html',
                           all_boundary_patterns_summary=all_boundary_patterns_summary,
                           yearly_data_for_selected_pattern=yearly_data_for_selected_pattern,
                           all_years_in_data=all_years_in_data,
                           boundary_pairs_for_dropdown=boundary_pairs_for_dropdown,
                           selected_pair=selected_pair) 

@app.route('/special_patterns_analysis')
def special_patterns_analysis_route():
    if df.empty:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"error": "Historical data not loaded or is empty."}), 500
        else:
            flash("Cannot display Special Patterns Analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
            return redirect(url_for('index'))
    
    # Pass df directly to the function, not for cache key serialization
    special_patterns_data = get_cached_analysis('special_patterns_analysis', get_special_patterns_analysis, df)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # This is an AJAX request, return JSON
        return jsonify(special_patterns_data)
    else:
        # Not an AJAX request, render the full template
        return render_template('special_patterns_analysis.html',
                            special_patterns_data=special_patterns_data)


@app.route('/find_results_by_first_white_ball', methods=['GET', 'POST'])
def find_results_by_first_white_ball():
    if df.empty:
        flash("Cannot find results: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    results_dict = []
    white_ball_number_display = None
    selected_sort_by = 'date_desc'

    if request.method == 'POST':
        white_ball_number_str = request.form.get('white_ball_number')
        selected_sort_by = request.form.get('sort_by', 'date_desc')

        if white_ball_number_str and white_ball_number_str.isdigit():
            white_ball_number = int(white_ball_number_str)
            white_ball_number_display = white_ball_number
            
            if 'Draw Date_dt' not in df.columns:
                 df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'], errors='coerce')

            results = df[df['Number 1'].astype(int) == white_ball_number].copy()

            if selected_sort_by == 'date_desc':
                results = results.sort_values(by='Draw Date_dt', ascending=False)
            elif selected_sort_by == 'date_asc':
                results = results.sort_values(by='Draw Date_dt', ascending=True)
            elif selected_sort_by == 'balls_asc':
                results['WhiteBallsTuple'] = results.apply(
                    lambda row: tuple(sorted([
                        int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
                        int(row['Number 4']), int(row['Number 5'])
                    ])), axis=1
                )
                results = results.sort_values(by='WhiteBallsTuple', ascending=True)
                results = results.drop(columns=['WhiteBallsTuple'])
            elif selected_sort_by == 'balls_desc':
                results['WhiteBallsTuple'] = results.apply(
                    lambda row: tuple(sorted([
                        int(row['Number 1']), int(row['Number 2']), int(row['Number 3']),
                        int(row['Number 4']), int(row['Number 5'])
                    ])), axis=1
                )
                results = results.sort_values(by='WhiteBallsTuple', ascending=False)
                results = results.drop(columns=['WhiteBallsTuple'])

            results_dict = results.to_dict('records')
        else:
            flash("Please enter a valid number for First White Ball Number.", 'error')

    return render_template('find_results_by_first_white_ball.html', 
                           results_by_first_white_ball=results_dict, 
                           white_ball_number=white_ball_number_display,
                           selected_sort_by=selected_sort_by)

def supabase_search_draws(query_params):
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=False) 

    try:
        response = requests.get(url, headers=headers, params=query_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            pass
        return []
    except Exception as e:
        traceback.print_exc()
        return []

@app.route('/strict_positional_search', methods=['GET', 'POST'])
def strict_positional_search_route():
    entered_numbers = {
        'white_ball_1': '', 'white_ball_2': '', 'white_ball_3': '', 
        'white_ball_4': '', 'white_ball_5': '', 'powerball_pos': ''
    }
    search_results = []
    total_results = 0
    
    if request.method == 'POST':
        entered_numbers['white_ball_1'] = request.form.get('white_ball_1', '').strip()
        entered_numbers['white_ball_2'] = request.form.get('white_ball_2', '').strip()
        entered_numbers['white_ball_3'] = request.form.get('white_ball_3', '').strip()
        entered_numbers['white_ball_4'] = request.form.get('white_ball_4', '').strip()
        entered_numbers['white_ball_5'] = request.form.get('white_ball_5', '').strip()
        entered_numbers['powerball_pos'] = request.form.get('powerball_pos', '').strip()

        if df.empty:
            flash("Historical data not loaded or is empty. Please check Supabase connection before searching.", 'error')
            return render_template('strict_positional_search.html', 
                                   entered_numbers=entered_numbers, 
                                   search_results=[],
                                   total_results=0)

        query_params = {'select': 'Draw Date,Number 1,Number 2,Number 3,Number 4,Number 5,Powerball'}
        filter_count = 0

        for i in range(1, 6):
            key = f'white_ball_{i}'
            col_name = f'Number {i}'
            if entered_numbers[key]:
                try:
                    num = int(entered_numbers[key])
                    if not (1 <= num <= 69):
                        flash(f"White ball {i} must be between 1 and 69. Please correct your input.", 'error')
                        return render_template('strict_positional_search.html', 
                                               entered_numbers=entered_numbers, 
                                               search_results=[],
                                               total_results=0)
                    query_params[col_name] = f'eq.{num}'
                    filter_count += 1
                except ValueError:
                    flash(f"White ball {i} must be a valid number. Please correct your input.", 'error')
                    return render_template('strict_positional_search.html', 
                                           entered_numbers=entered_numbers, 
                                           search_results=[],
                                           total_results=0)

        if entered_numbers['powerball_pos']:
            try:
                pb_num = int(entered_numbers['powerball_pos'])
                if not (1 <= pb_num <= 26):
                    flash("Powerball must be between 1 and 26. Please correct your input.", 'error')
                    return render_template('strict_positional_search.html', 
                                           entered_numbers=entered_numbers, 
                                           search_results=[],
                                           total_results=0)
                query_params['Powerball'] = f'eq.{pb_num}'
                filter_count += 1
            except ValueError:
                flash("Powerball must be a valid number. Please correct your input.", 'error')
                return render_template('strict_positional_search.html', 
                                       entered_numbers=entered_numbers, 
                                       search_results=[],
                                       total_results=0)
        
        if filter_count == 0:
            flash("Please enter at least one number to perform a search.", 'info')
            return render_template('strict_positional_search.html', 
                                   entered_numbers=entered_numbers, 
                                   search_results=[],
                                   total_results=0)

        draws = supabase_search_draws(query_params)

        if draws:
            search_results = sorted(draws, key=lambda x: x.get('Draw Date', ''), reverse=True)
            total_results = len(search_results)
            if total_results == 0:
                flash("No draws found matching your criteria.", 'info')
            else:
                flash(f"Found {total_results} draw(s) matching your criteria.", 'success')
        else:
            flash("Error fetching data from Supabase. Please try again later.", 'error')

    return render_template('strict_positional_search.html', 
                           entered_numbers=entered_numbers, 
                           search_results=search_results,
                           total_results=total_results)


@app.route('/generated_numbers_history')
def generated_numbers_history_route():
    generated_history = get_cached_analysis('generated_history', get_generated_numbers_history)
    
    official_draw_dates = []
    if not df.empty:
        official_draw_dates = sorted(df['Draw Date'].unique(), reverse=True)

    last_draw_for_template = last_draw.to_dict()

    return render_template('generated_numbers_history.html', 
                           generated_history=generated_history,
                           official_draw_dates=official_draw_dates,
                           last_official_draw=last_draw_for_template) 


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
        
        simulated_draw_date_dt = datetime.now()
        simulated_draw_date = simulated_draw_date_dt.strftime('%Y-%m-%d')
        simulated_numbers_list = sorted(random.sample(range(1, 70), 5))
        simulated_powerball = random.randint(1, 26)

        new_draw_data = {
            'Draw Date': simulated_draw_date,
            'Number 1': simulated_numbers_list[0],
            'Number 2': simulated_numbers_list[1],
            'Number 3': simulated_numbers_list[2], 
            'Number 4': simulated_numbers_list[3], 
            'Number 5': simulated_numbers_list[4], 
            'Powerball': simulated_powerball
        }
        
        if new_draw_data['Draw Date'] == last_db_draw_date:
            return "No new draw data. Database is up-to-date.", 200
        
        url_insert = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        insert_response = requests.post(url_insert, headers=service_headers, data=json.dumps(new_draw_data))
        insert_response.raise_for_status()

        if insert_response.status_code == 201:
            initialize_core_data() 
            invalidate_analysis_cache()

            return f"Data updated successfully with draw for {simulated_draw_date}.", 200
        else:
            return f"Error updating data: {insert_response.status_code} - {insert_response.text}", 500

    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            pass
        return f"Network or HTTP error: {e}", 500
    except json.JSONDecodeError as e:
        if 'insert_response' in locals() and insert_response is not None:
            pass
        return f"JSON parsing error: {e}", 500
    except Exception as e:
        traceback.print_exc()
        return f"An internal error occurred: {e}", 500


@app.route('/export_analysis_results')
def export_analysis_results_route():
    if df.empty:
        flash("Cannot export results: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    export_analysis_results(df) 
    flash("Analysis results exported to analysis_results.csv (this file is temporary on Vercel's serverless environment).", 'info')
    return redirect(url_for('index'))


@app.route('/analyze_batch_vs_official', methods=['POST'])
def analyze_batch_vs_official_route():
    try:
        data = request.get_json()
        generated_date_str = data.get('generated_date')
        official_draw_date_str = data.get('official_draw_date')

        if not generated_date_str or not official_draw_date_str:
            return jsonify({"error": "Missing generated_date or official_draw_date"}), 400

        generated_picks = _get_generated_picks_for_date_from_db(generated_date_str)
        if not generated_picks:
            return jsonify({"error": f"No generated picks found for date: {generated_date_str}"}), 404

        official_draw = _get_official_draw_for_date_from_db(official_draw_date_str)
        if not official_draw:
            return jsonify({"error": f"No official draw found for date: {official_draw_date_str}. Please ensure it is added to the database."}), 404

        analysis_summary = analyze_generated_batch_against_official_draw(generated_picks, official_draw)
        
        return jsonify({
            "success": True,
            "generated_date": generated_date_str,
            "official_draw_date": official_draw_date_str,
            "total_generated_picks_in_batch": len(generated_picks),
            "summary": analysis_summary
        })

    except Exception as e:
        traceback.print_exc() 
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/analyze_generated_historical_matches', methods=['POST'])
def analyze_generated_historical_matches_route():
    if df.empty:
        return jsonify({"success": False, "error": "Historical data not loaded or is empty."}), 500
    
    try:
        data = request.get_json() 
        generated_white_balls_str = data.get('generated_white_balls')
        generated_powerball_str = data.get('generated_powerball')

        if not generated_white_balls_str or not generated_powerball_str: 
            return jsonify({"success": False, "error": "Missing generated_white_balls or generated_powerball"}), 400

        generated_white_balls = sorted([int(x.strip()) for x in generated_white_balls_str.split(',') if x.strip().isdigit()])
        generated_powerball = int(generated_powerball_str)

        if len(generated_white_balls) != 5:
            return jsonify({"success": False, "error": "Invalid generated white balls format. Expected 5 numbers."}), 400

        historical_match_results = check_generated_against_history(generated_white_balls, generated_powerball, df)
        
        return jsonify({
            "success": True,
            "generated_numbers_for_analysis": generated_white_balls, 
            "generated_powerball_for_analysis": generated_powerball,
            "match_summary": historical_match_results['summary']
        })

    except ValueError:
        return jsonify({"success": False, "error": "Invalid number format for historical analysis."}), 400
    except Exception as e:
        traceback.print_exc() 
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/ai_assistant')
def ai_assistant_route():
    return render_template('ai_assistant.html')

@app.route('/chat_with_ai', methods=['POST'])
def chat_with_ai_route():
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({"response": "Please provide a message."}), 400

    chat_history = []
    
    # Get the detailed historical summary
    historical_summary = _summarize_for_ai(df)

    initial_prompt_text = f"""
    You are a highly knowledgeable Powerball Lottery Analysis AI Assistant. 
    Your purpose is to help users understand Powerball data, trends, and probabilities.
    You should be helpful, informative, and concise.
    
    Here's some general knowledge about Powerball:
    - White Balls: 5 numbers are drawn from a pool of 1 to 69.
    - Powerball: 1 number is drawn from a separate pool of 1 to 26.
    - Odds of winning the jackpot (matching 5 White Balls + Powerball): 1 in 292,201,338.
    
    Here is a summary of the *current and recent Powerball historical data and trends* from the user's database:
    ---
    {historical_summary}
    ---

    When answering, refer to the provided historical data summary where relevant.
    If a user asks to "generate numbers," or requests specific real-time data that you don't have access to (like "what are the current hot numbers?" *if not explicitly in the summary*), you should gently explain that you are an analytical assistant and cannot perform live data lookups or generate numbers, but can explain the *concepts* behind them and refer to the data provided in the summary.

    User's question: """ + user_message

    chat_history.append({
        "role": "user",
        "parts": [{"text": initial_prompt_text}]
    })
    
    try:
        apiKey = GEMINI_API_KEY 
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

        payload = {
            "contents": chat_history,
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 500
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        }
        
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() 
        
        result = response.json()

        if result and result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            ai_response = result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"response": ai_response})
        else:
            error_message = "I'm sorry, I couldn't generate a response. The AI model did not return valid content."
            return jsonify({"response": error_message}), 500

    except requests.exceptions.RequestException as e:
        traceback.print_exc() 
        return jsonify({"response": f"Error communicating with AI: {e}"}), 500
    except json.JSONDecodeError as e:
        traceback.print_exc() 
        return jsonify({"response": f"Error parsing AI response: {e}"}), 500
    except Exception as e:
        traceback.print_exc() 
        return jsonify({"response": f"An internal error occurred: {e}"}), 500

@app.route('/my_jackpot_pick')
def my_jackpot_pick_route():
    try:
        return render_template('my_jackpot_pick.html')
    except Exception as e:
        traceback.print_exc() 
        flash("An error occurred loading the Jackpot Pick page. Please try again.", 'error')
        return redirect(url_for('index')) 

@app.route('/analyze_manual_pick', methods=['POST'])
def analyze_manual_pick_route():
    if df.empty:
        return jsonify({"error": "Historical data not loaded or is empty."}), 500
    
    try:
        data = request.get_json()
        white_balls = data.get('white_balls')
        powerball = data.get('powerball')

        if not white_balls or len(white_balls) != 5 or powerball is None:
            return jsonify({"error": "Invalid input. Please provide 5 white balls and 1 powerball."}), 400
        
        white_balls = sorted([int(n) for n in white_balls])
        powerball = int(powerball)

        historical_match_results = check_generated_against_history(white_balls, powerball, df)
        
        last_drawn_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        return jsonify({
            "success": True,
            "generated_numbers": white_balls, 
            "generated_powerball": powerball,
            "match_summary": historical_match_results['summary'],
            "last_drawn_dates": last_drawn_dates
        })

    except ValueError:
        return jsonify({"error": "Invalid number format provided."}), 400
    except Exception as e:
        traceback.print_exc() 
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/save_manual_pick', methods=['POST']) 
def save_manual_pick_route():
    try:
        data = request.get_json()
        white_balls = data.get('white_balls')
        powerball = data.get('powerball')

        if not white_balls or len(white_balls) != 5 or powerball is None:
            return jsonify({"success": False, "error": "Invalid input. Please provide 5 white balls and 1 powerball."}), 400
        
        white_balls = sorted([int(n) for n in white_balls])
        powerball = int(powerball)

        success, message = save_generated_numbers_to_db(white_balls, powerball)

        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"success": False, "error": message}), 400

    except ValueError:
        return jsonify({"success": False, "error": "Invalid number format provided."}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/sum_trends_and_gaps')
def sum_trends_and_gaps_route():
    if df.empty:
        flash("Cannot display Sum Trends and Gaps: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Pass df directly to the function, not for cache key serialization
    sum_data = get_cached_analysis('sum_trends_and_gaps', get_sum_trends_and_gaps_data, df)
    
    return render_template('sum_trends_and_gaps.html', 
                           min_possible_sum=sum_data['min_possible_sum'],
                           max_possible_sum=sum_data['max_possible_sum'],
                           appeared_sums_details=sum_data['appeared_sums_details'],
                           missing_sums=sum_data['missing_sums'],
                           grouped_sums_analysis=sum_data['grouped_sums_analysis'])

@app.route('/weekday_trends')
def weekday_trends_route():
    if df.empty:
        flash("Cannot display Weekday Trends: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Pass df directly to the function, not for cache key serialization
    weekday_data = get_cached_analysis('weekday_all_trends', get_weekday_draw_trends, df, group_a_numbers_def=group_a)
    
    return render_template('weekday_trends.html', 
                           weekday_trends=weekday_data)

@app.route('/yearly_white_ball_trends')
def yearly_white_ball_trends_route():
    if df.empty:
        flash("Cannot display Yearly White Ball Trends: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Get the list of years for the dropdown/chart labels (up to current year + 1 for future-proofing)
    current_year = datetime.now().year
    # Calculate the start year for a 10-year rolling window
    start_year_for_display = max(2017, current_year - 9) 
    years_for_display = list(range(start_year_for_display, current_year + 1))

    return render_template('yearly_white_ball_trends.html',
                           years=years_for_display) # Pass years for initial dropdown values

# Renamed from /api/yearly_white_ball_data
@app.route('/api/white_ball_trends')
def api_white_ball_trends_route():
    if df.empty:
        return jsonify({"error": "Historical data not loaded or is empty."}), 500
    
    period_type = request.args.get('period', 'year') # Default to 'year'

    # Cache the data based on the period type
    # Pass df directly to the function, not for cache key serialization
    # Removed start_year parameter, it's now dynamically calculated within the function
    white_ball_data, period_labels = get_cached_analysis(
        f'white_ball_frequency_{period_type}', 
        get_white_ball_frequency_by_period, 
        df, 
        period_type=period_type
    )
    
    return jsonify({
        'data': white_ball_data,
        'period_labels': period_labels
    })

# --- Smart Pick Generator Logic (New Functions and Route) ---

def _get_current_month_hot_numbers(df_source):
    """
    Identifies numbers that have appeared more than once in the current (incomplete) month's draws.
    Returns a set of these hot numbers.
    """
    if df_source.empty:
        return set()

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])

    if df_copy.empty:
        return set()

    current_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    current_month_df = df_copy[df_copy['Draw Date_dt'] >= current_month_start]

    if current_month_df.empty:
        return set()

    monthly_counts = defaultdict(int)
    for _, row in current_month_df.iterrows():
        for i in range(1, 6):
            num = int(row[f'Number {i}'])
            monthly_counts[num] += 1
        monthly_counts[int(row['Powerball'])] += 1 # Include Powerball in hot numbers consideration

    hot_numbers = {num for num, count in monthly_counts.items() if count > 1}
    return hot_numbers

def _score_pick_for_patterns(white_balls, criteria_data):
    """
    Scores a generated white ball pick based on how well it aligns with various
    pattern-based preferences (soft constraints). Higher score means better alignment.
    """
    score = 0
    wb_set = set(white_balls)
    sorted_wb = sorted(white_balls)

    # 1. Grouped Patterns Score
    if criteria_data['prioritize_grouped_patterns'] and criteria_data['most_frequent_grouped_patterns']:
        for pattern_info in criteria_data['most_frequent_grouped_patterns']:
            pattern_set = set(pattern_info['pattern'])
            if pattern_set.issubset(wb_set):
                # Give higher score for more frequent patterns
                score += pattern_info['count'] * 0.1 # Adjust multiplier as needed

    # 2. Special Patterns Score
    if criteria_data['prioritize_special_patterns'] and criteria_data['most_frequent_special_patterns']:
        # Combine all special patterns for scoring
        all_special_patterns = []
        all_special_patterns.extend(criteria_data['most_frequent_special_patterns']['tens_apart_patterns'])
        all_special_patterns.extend(criteria_data['most_frequent_special_patterns']['same_last_digit_patterns'])
        all_special_patterns.extend(criteria_data['most_frequent_special_patterns']['repeating_digit_patterns'])
        
        for pattern_info in all_special_patterns:
            pattern_set = set(pattern_info['pattern'])
            if pattern_set.issubset(wb_set):
                score += pattern_info['count'] * 0.05 # Smaller multiplier for special patterns

    # 3. Consecutive Trends Score
    if criteria_data['prioritize_consecutive_patterns']:
        consecutive_sequences = _find_consecutive_sequences(white_balls) # Use the correct function
        score += len(consecutive_sequences) * 5 # Score for each consecutive sequence found
        # Add bonus for triplets if present (check if any sequence is length 3)
        if any(len(s) >= 3 for s in consecutive_sequences):
            score += 10 # Bonus for a triplet

    # 4. Monthly Hot Numbers Score
    if criteria_data['prioritize_monthly_hot'] and criteria_data['current_month_hot_numbers']:
        hot_count = len(wb_set.intersection(criteria_data['current_month_hot_numbers']))
        score += hot_count * 2 # Score for each hot number included

    return score

def generate_smart_picks(df_source, num_sets, excluded_numbers, num_from_group_a, odd_even_choice, sum_range_tuple, prioritize_monthly_hot, prioritize_grouped_patterns, prioritize_special_patterns, prioritize_consecutive_patterns, force_specific_pattern):
    """
    Generates Powerball picks based on a combination of hard and soft criteria.
    """
    if df_source.empty:
        raise ValueError("Historical data is empty. Cannot generate smart picks.")

    generated_sets = []
    max_overall_attempts = 5000 * num_sets # Increased attempts for complex criteria

    # Pre-calculate historical data needed for soft constraints
    # These are cached, so calling them here is efficient
    all_grouped_patterns = get_cached_analysis('grouped_patterns', get_grouped_patterns_over_years, df_source)
    all_special_patterns = get_cached_analysis('special_patterns_analysis', get_special_patterns_analysis, df_source)
    
    # For scoring, we need a flat list of most frequent patterns
    most_frequent_grouped_patterns = sorted(all_grouped_patterns, key=lambda x: x['count'], reverse=True)[:50] # Top 50 grouped patterns
    
    # Combine all special patterns into one list for easier scoring
    most_frequent_special_patterns = {
        'tens_apart_patterns': sorted(all_special_patterns['tens_apart_patterns'], key=lambda x: x['count'], reverse=True)[:20],
        'same_last_digit_patterns': sorted(all_special_patterns['same_last_digit_patterns'], key=lambda x: x['count'], reverse=True)[:20],
        'repeating_digit_patterns': sorted(all_special_patterns['repeating_digit_patterns'], key=lambda x: x['count'], reverse=True)[:20]
    }

    current_month_hot_numbers = set()
    if prioritize_monthly_hot:
        current_month_hot_numbers = _get_current_month_hot_numbers(df_source)

    # Prepare criteria data for scoring function
    criteria_for_scoring = {
        'prioritize_monthly_hot': prioritize_monthly_hot,
        'current_month_hot_numbers': current_month_hot_numbers,
        'prioritize_grouped_patterns': prioritize_grouped_patterns,
        'most_frequent_grouped_patterns': most_frequent_grouped_patterns,
        'prioritize_special_patterns': prioritize_special_patterns,
        'most_frequent_special_patterns': most_frequent_special_patterns,
        'prioritize_consecutive_patterns': prioritize_consecutive_patterns,
    }

    for _ in range(num_sets):
        best_pick_white_balls = None
        best_pick_powerball = None
        highest_score = -1
        current_set_attempts = 0
        max_attempts_per_set = max_overall_attempts // num_sets # Distribute attempts

        while current_set_attempts < max_attempts_per_set:
            current_set_attempts += 1
            
            candidate_white_balls = []
            candidate_powerball = random.randint(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1])
            
            # 1. Handle Forced Specific Pattern (Hard Constraint)
            remaining_to_pick = 5
            temp_excluded = set(excluded_numbers)
            
            if force_specific_pattern:
                for num in force_specific_pattern:
                    if not (GLOBAL_WHITE_BALL_RANGE[0] <= num <= GLOBAL_WHITE_BALL_RANGE[1]) or num in temp_excluded:
                        # If forced number is invalid or excluded, this attempt fails
                        continue 
                candidate_white_balls.extend(force_specific_pattern)
                temp_excluded.update(force_specific_pattern)
                remaining_to_pick -= len(force_specific_pattern)
                
            # Ensure we have enough numbers left to pick
            if remaining_to_pick < 0: # Should not happen with valid input
                continue
            
            available_pool = [n for n in range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1)
                              if n not in temp_excluded and n not in candidate_white_balls]

            if len(available_pool) < remaining_to_pick:
                continue # Not enough numbers to complete the pick

            # 2. Handle Group A Numbers (Hard Constraint)
            # Determine how many more Group A numbers are needed
            current_group_a_count = sum(1 for num in candidate_white_balls if num in group_a)
            needed_from_group_a = num_from_group_a - current_group_a_count

            temp_available_pool = list(available_pool) # Copy to modify
            
            if needed_from_group_a > 0:
                possible_group_a_from_pool = [n for n in temp_available_pool if n in group_a]
                if len(possible_group_a_from_pool) < needed_from_group_a:
                    continue # Not enough Group A numbers available
                
                try:
                    selected_group_a = random.sample(possible_group_a_from_pool, needed_from_group_a)
                    candidate_white_balls.extend(selected_group_a)
                    temp_excluded.update(selected_group_a)
                    remaining_to_pick -= needed_from_group_a
                    
                    # Update available pool after selecting Group A numbers
                    available_pool = [n for n in available_pool if n not in selected_group_a]
                except ValueError: # Not enough elements to sample
                    continue
            elif needed_from_group_a < 0: # Already have too many Group A numbers from forced pattern
                # This scenario means the forced pattern already violates the Group A count.
                # We should probably raise an error earlier or ensure the UI prevents this.
                # For now, just skip this candidate.
                continue

            # 3. Fill remaining spots randomly from available pool
            if remaining_to_pick > 0:
                if len(available_pool) < remaining_to_pick:
                    continue # Not enough numbers left
                try:
                    random_fill = random.sample(available_pool, remaining_to_pick)
                    candidate_white_balls.extend(random_fill)
                except ValueError: # Not enough elements to sample
                    continue
            
            # Ensure 5 unique white balls
            if len(set(candidate_white_balls)) != 5:
                continue
            
            candidate_white_balls = sorted(candidate_white_balls)

            # 4. Check Odd/Even Split (Hard Constraint)
            even_count = sum(1 for num in candidate_white_balls if num % 2 == 0)
            odd_count = 5 - even_count
            
            # Adjusted for expected format from form (e.g., "3 Even / 2 Odd")
            expected_odd_even_split = odd_even_choice
            
            # If a specific odd/even choice is made, apply it strictly
            if expected_odd_even_split != "Any":
                current_split_str = f"{odd_count} Odd / {even_count} Even"
                if current_split_str != expected_odd_even_split:
                    # Special handling for "All Even" or "All Odd" which are specific cases
                    if expected_odd_even_split == "All Even" and even_count != 5: continue
                    if expected_odd_even_split == "All Odd" and odd_count != 5: continue
                    # For other named splits, check for exact match
                    if expected_odd_even_split not in ["All Even", "All Odd"] and current_split_str != expected_odd_even_split: continue

            # 5. Check Sum Range (Hard Constraint)
            current_sum = sum(candidate_white_balls)
            if sum_range_tuple and not (sum_range_tuple[0] <= current_sum <= sum_range_tuple[1]):
                continue

            # 6. Check against last draw and historical exact matches (Hard Constraint)
            if check_exact_match(candidate_white_balls): 
                continue
            
            last_draw_data = get_last_draw(df_source)
            if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
                last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
                if set(candidate_white_balls) == set(last_white_balls) and candidate_powerball == int(last_draw_data['Powerball']):
                    continue

            # All hard constraints met, now score for soft constraints
            current_score = _score_pick_for_patterns(candidate_white_balls, criteria_for_scoring)

            if current_score > highest_score:
                highest_score = current_score
                best_pick_white_balls = candidate_white_balls
                best_pick_powerball = candidate_powerball
            
            # If we found a perfect pick (can define "perfect" as score > threshold or just the first valid one)
            # For now, if a pick has a positive score and meets all hard constraints, we can consider it "good enough"
            # to prevent excessive iterations if many options exist, especially for lower `num_sets`.
            if highest_score > 0 and current_set_attempts > max_attempts_per_set / 2: # Found a good one relatively early
                 break
        
        if best_pick_white_balls:
            generated_sets.append({'white_balls': best_pick_white_balls, 'powerball': best_pick_powerball})
        else:
            raise ValueError(f"Could not generate a smart pick meeting all criteria after {max_attempts_per_set} attempts. Try adjusting filters or reducing strictness.")
            
    return generated_sets


@app.route('/smart_pick_generator')
def smart_pick_generator_route():
    if df.empty:
        flash("Cannot load Smart Pick Generator: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)

@app.route('/generate_smart_picks_route', methods=['POST'])
def generate_smart_picks_route():
    if df.empty:
        flash("Cannot generate smart picks: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)

    try:
        num_sets_to_generate = int(request.form.get('num_sets_to_generate', 1))
        excluded_numbers_input = request.form.get('excluded_numbers', '')
        excluded_numbers_local = [int(num.strip()) for num in excluded_numbers_input.split(',') if num.strip().isdigit()] if excluded_numbers_input else []
        
        num_from_group_a = int(request.form.get('num_from_group_a', 0))
        odd_even_choice = request.form.get('odd_even_choice', 'Any')
        
        selected_sum_range_label = request.form.get('sum_range_filter', 'Any')
        selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)

        prioritize_monthly_hot = 'prioritize_monthly_hot' in request.form
        prioritize_grouped_patterns = 'prioritize_grouped_patterns' in request.form
        prioritize_special_patterns = 'prioritize_special_patterns' in request.form
        prioritize_consecutive_patterns = 'prioritize_consecutive_patterns' in request.form
        
        force_specific_pattern_input = request.form.get('force_specific_pattern', '')
        force_specific_pattern = []
        if force_specific_pattern_input:
            force_specific_pattern = sorted([int(num.strip()) for num in force_specific_pattern_input.split(',') if num.strip().isdigit()])
            if not (2 <= len(force_specific_pattern) <= 3):
                flash("Forced specific pattern must contain 2 or 3 numbers.", 'error')
                return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)
            # Check if forced numbers are within range and not excluded
            for num in force_specific_pattern:
                if not (GLOBAL_WHITE_BALL_RANGE[0] <= num <= GLOBAL_WHITE_BALL_RANGE[1]):
                    flash(f"Forced number {num} is outside the valid white ball range (1-69).", 'error')
                    return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)
                if num in excluded_numbers_local:
                    flash(f"Forced number {num} is also in the excluded numbers list. Please remove it from excluded.", 'error')
                    return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)
            if len(set(force_specific_pattern)) != len(force_specific_pattern):
                flash("Forced specific pattern numbers must be unique.", 'error')
                return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)


        generated_sets = generate_smart_picks(
            df_source=df,
            num_sets=num_sets_to_generate,
            excluded_numbers=excluded_numbers_local,
            num_from_group_a=num_from_group_a,
            odd_even_choice=odd_even_choice,
            sum_range_tuple=selected_sum_range_tuple,
            prioritize_monthly_hot=prioritize_monthly_hot,
            prioritize_grouped_patterns=prioritize_grouped_patterns,
            prioritize_special_patterns=prioritize_special_patterns,
            prioritize_consecutive_patterns=prioritize_consecutive_patterns,
            force_specific_pattern=force_specific_pattern
        )
        
        # For display, get last draw dates for the *last* generated set
        last_draw_dates = {}
        if generated_sets:
            last_draw_dates = find_last_draw_dates_for_numbers(df, generated_sets[-1]['white_balls'], generated_sets[-1]['powerball'])

        return render_template('smart_pick_generator.html', 
                               generated_sets=generated_sets, 
                               last_draw_dates=last_draw_dates,
                               sum_ranges=SUM_RANGES,
                               group_a=group_a,
                               # Pass back selected values to re-populate form
                               num_sets_to_generate=num_sets_to_generate,
                               excluded_numbers=excluded_numbers_input,
                               num_from_group_a=num_from_group_a,
                               odd_even_choice=odd_even_choice,
                               selected_sum_range=selected_sum_range_label,
                               prioritize_monthly_hot=prioritize_monthly_hot,
                               prioritize_grouped_patterns=prioritize_grouped_patterns,
                               prioritize_special_patterns=prioritize_special_patterns,
                               prioritize_consecutive_patterns=prioritize_consecutive_patterns,
                               force_specific_pattern_input=force_specific_pattern_input
                               )

    except ValueError as e:
        flash(str(e), 'error')
        return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)
    except Exception as e:
        traceback.print_exc()
        flash(f"An unexpected error occurred: {e}", 'error')
        return render_template('smart_pick_generator.html', sum_ranges=SUM_RANGES, group_a=group_a)

