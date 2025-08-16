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

CACHE_EXPIRATION_SECONDS = 3600 # 1 hour

group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]
GLOBAL_WHITE_BALL_RANGE = (1, 69)
GLOBAL_POWERBALL_RANGE = (1, 26)

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


# --- Core Utility Functions ---

def _get_supabase_headers(is_service_key=False):
    key = SUPABASE_SERVICE_KEY if is_service_key else SUPABASE_ANON_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

def load_historical_data_from_supabase():
    """Fetches historical Powerball draw data from Supabase."""
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

def get_last_draw(df_source):
    """Retrieves the most recent draw from the DataFrame."""
    if df_source.empty:
        return pd.Series({
            'Draw Date': 'N/A', 'Number 1': 'N/A', 'Number 2': 'N/A',
            'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A',
            'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A'] 
        }, dtype='object')
    
    last_row = df_source.iloc[-1].copy() 
    
    if 'Number 1' in last_row and pd.notna(last_row['Number 1']): 
        last_row['Numbers'] = [
            int(last_row['Number 1']), int(last_row['Number 2']), int(last_row['Number 3']), 
            int(last_row['Number 4']), int(last_row['Number 5'])
        ]
    else:
        last_row['Numbers'] = ['N/A'] * 5 

    if 'Draw Date_dt' in last_row and pd.notna(last_row['Draw Date_dt']):
        last_row['Draw Date'] = last_row['Draw Date_dt'].strftime('%Y-%m-%d')
    elif 'Draw Date' not in last_row: 
        last_row['Draw Date'] = 'N/A'
    
    return last_row

def check_exact_match(white_balls):
    """Checks if a given set of white balls exactly matches any historical draw."""
    global historical_white_ball_sets
    return frozenset(white_balls) in historical_white_ball_sets

def generate_powerball_numbers(df_source, group_a_list, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None, selected_sum_range_tuple=None, is_simulation=False):
    """Generates a single Powerball combination based on various criteria."""
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 5000 
    attempts = 0
    
    base_available_white_balls = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
    if len(base_available_white_balls) < 5:
        raise ValueError("Not enough available white balls after exclusions and range constraints.")

    while attempts < max_attempts:
        
        white_balls_candidate = sorted(random.sample(base_available_white_balls, 5))

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

        if selected_sum_range_tuple:
            current_sum = sum(white_balls_candidate)
            if not (selected_sum_range_tuple[0] <= current_sum <= selected_sum_range_tuple[1]):
                attempts += 1
                continue 

        group_a_numbers = [num for num in white_balls_candidate if num in group_a_list]
        if len(group_a_numbers) < 2: 
            attempts += 1
            continue
            
        if high_low_balance is not None:
            low_numbers_count = sum(1 for num in white_balls_candidate if num <= LOW_NUMBER_MAX)
            high_numbers_count = sum(1 for num in white_balls_candidate if num >= HIGH_NUMBER_MIN)
            if low_numbers_count != high_low_balance[0] or high_numbers_count != high_low_balance[1]:
                attempts += 1
                continue
        
        if not is_simulation:
            last_draw_data = get_last_draw(df_source)
            if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
                last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
                if set(white_balls_candidate) == set(last_white_balls): 
                    attempts += 1
                    continue

            if check_exact_match(white_balls_candidate): 
                attempts += 1
                continue

        powerball = random.randint(powerball_range[0], powerball_range[1])
        return white_balls_candidate, powerball

    raise ValueError("Could not generate a unique combination meeting all criteria after many attempts. Try adjusting filters or increasing max_attempts.")


def generate_from_group_a(df_source, num_from_group_a, white_ball_range, powerball_range, excluded_numbers, selected_sum_range_tuple=None):
    """Generates a Powerball combination ensuring a certain number of Group A numbers."""
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
    """Generates a Powerball combination starting with two user-provided white balls."""
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

def frequency_analysis(df_source):
    """Calculates and returns frequency of white balls and powerballs."""
    if df_source.empty: 
        return [], []
    white_balls = df_source[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)
    powerball_freq = df_source['Powerball'].astype(int).value_counts().reindex(range(1, 27), fill_value=0)
    
    white_ball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.items()]
    powerball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in powerball_freq.items()]

    return white_ball_freq_list, powerball_freq_list

def hot_cold_numbers(df_source, last_draw_date_str):
    """Identifies hot and cold numbers based on recent draws."""
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
    """
    Analyzes monthly white ball and powerball draws, and streak numbers.
    This function is intended for the /monthly_white_ball_analysis route.
    """
    if dataframe.empty:
        return {'monthly_data': [], 'streak_numbers': {'3_month_streaks': [], '4_month_streaks': [], '5_month_streaks': []}}

    df_sorted = dataframe.sort_values(by='Draw Date_dt', ascending=False).copy()
    df_sorted['YearMonth'] = df_sorted['Draw Date_dt'].dt.to_period('M')
    unique_months_periods = sorted(df_sorted['YearMonth'].unique(), reverse=True)

    monthly_display_data = [] 
    
    for period in unique_months_periods[:num_months_for_top_display]: # Limiting to N months for display
        month_df = df_sorted[df_sorted['YearMonth'] == period]
        if month_df.empty:
            continue

        is_current_month_flag = (period == pd.Period(datetime.now(), freq='M'))

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
    
    monthly_display_data.sort(key=lambda x: datetime.strptime(x['month'], '%B %Y'))

    # Streaks calculation (needs all completed months)
    numbers_per_completed_month = defaultdict(set)
    for period in unique_months_periods:
        current_period_dt = pd.Period(datetime.now(), freq='M')
        if period == current_period_dt: 
            continue 
        month_df = df_sorted[df_sorted['YearMonth'] == period]
        if not month_df.empty:
            for _, row in month_df.iterrows():
                for i in range(1, 6):
                    numbers_per_completed_month[period].add(int(row[f'Number {i}']))
                numbers_per_completed_month[period].add(int(row['Powerball']))
    
    completed_months_sorted = sorted([p for p in unique_months_periods if p != pd.Period(datetime.now(), freq='M')])

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
    """Calculates sum of main balls, their frequency, and min/max/avg sums."""
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
    """Finds historical draws matching a specific sum."""
    if df_source.empty: return pd.DataFrame()
    temp_df = df_source.copy()
    for col in ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']: 
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)

    if 'Sum' not in temp_df.columns:
        temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    results = temp_df[temp_df['Sum'] == target_sum]
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum', 'Draw Date_dt']]

def simulate_multiple_draws(df_source, group_a_list, odd_even_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
    """Simulates multiple Powerball draws based on criteria and returns frequencies."""
    if df_source.empty: 
        return {'white_ball_freq': [], 'powerball_freq': []}
    
    white_ball_results = defaultdict(int)
    powerball_results = defaultdict(int)

    for _ in range(num_draws):
        try:
            white_balls, powerball = generate_powerball_numbers(
                df_source, group_a_list, odd_even_choice, "No Combo", 
                white_ball_range, powerball_range, excluded_numbers, 
                high_low_balance=None, selected_sum_range_tuple=None, is_simulation=True
            )
            
            for wb in white_balls:
                white_ball_results[wb] += 1
            powerball_results[powerball] += 1

        except ValueError: 
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
    """Calculates combinations (nCk)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

def winning_probability(white_ball_range_tuple, powerball_range_tuple):
    """Calculates the odds of winning Powerball given specified ranges."""
    total_white_balls_in_range = white_ball_range_tuple[1] - white_ball_range_tuple[0] + 1
    white_ball_combinations = calculate_combinations(total_white_balls_in_range, 5)

    total_powerballs_in_range = powerball_range_tuple[1] - powerball_range_tuple[0] + 1

    total_combinations = white_ball_combinations * total_powerballs_in_range

    probability_1_in_x = f"1 in {total_combinations:,}" if total_combinations > 0 else "N/A"
    probability_percentage = f"{1 / total_combinations * 100:.10f}%" if total_combinations > 0 else "N/A"

    return probability_1_in_x, probability_percentage

def partial_match_probabilities(white_ball_range_tuple, powerball_range_tuple):
    """Calculates probabilities for partial Powerball matches."""
    total_white_balls_in_range = white_ball_range_tuple[1] - white_ball_range_tuple[0] + 1
    total_powerballs_in_range = powerball_range_tuple[1] - powerball_range_tuple[0] + 1

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

def find_last_draw_dates_for_numbers(df_source, white_balls, powerball):
    """Finds the last draw date for each given number."""
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
    """Helper to get the last drawn date for a single number (white or powerball)."""
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
    """Finds the last co-occurrence date for a given pattern of numbers."""
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
    """Calculates the 'age' (miss streak) for each number."""
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
            detailed_ages.append({'number': int(i), 'type': 'Powerball', 'age': len(all_draw_dates), 'last_drawn_date': last_appearance_date_str}) 

    all_miss_streaks_only = [item['age'] for item in detailed_ages]
    age_counts = pd.Series(all_miss_streaks_only).value_counts().sort_index()
    age_counts_list = [{'age': int(age), 'count': int(count)} for age, count in age_counts.items()]
    
    return age_counts_list, detailed_ages

def get_co_occurrence_matrix(df_source):
    """Calculates the co-occurrence frequency of all white ball pairs."""
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
    """Identifies consecutive sequences in a list of numbers."""
    sequences = []
    if not numbers_list:
        return sequences

    sorted_nums = sorted(list(set(numbers_list))) 
    if not sorted_nums:
        return sequences

    current_sequence = [sorted_nums[0]]
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] == current_sequence[-1] + 1:
            current_sequence.append(sorted_nums[i])
        else:
            if len(current_sequence) >= 2: 
                sequences.append(current_sequence)
            current_sequence = [sorted_nums[i]]
    
    if len(current_sequence) >= 2:
        sequences.append(current_sequence)
        
    return sequences

def get_consecutive_numbers_trends(df_source, last_draw_date_str):
    """Analyzes recent draws for presence of consecutive numbers."""
    if df_source.empty or last_draw_date_str == 'N/A':
        return []

    try:
        last_draw_date = pd.to_datetime(last_draw_date_str)
    except Exception as e:
        return []

    one_year_ago = last_draw_date - pd.DateOffset(years=1) 

    if 'Draw Date_dt' not in df_source.columns or not pd.api.types.is_datetime64_any_dtype(df_source['Draw Date_dt']):
        return []

    recent_data = df_source[df_source['Draw Date_dt'] >= one_year_ago].copy() 
    recent_data = recent_data.sort_values(by='Draw Date_dt', ascending=False)
    if recent_data.empty:
        return []

    trend_data = []
    for idx, row in recent_data.iterrows():
        white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        consecutive_sequences = _find_consecutive_sequences(white_balls)
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            'consecutive_present': "Yes" if consecutive_sequences else "No",
            'consecutive_sequences': consecutive_sequences
        })
    
    return trend_data

def get_most_frequent_triplets(df_source): 
    """Finds the most frequent triplets of white balls."""
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
    """Analyzes odd/even splits, sum, and Group A numbers in recent draws."""
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
    """Calculates Powerball frequency per year."""
    if df_source.empty:
        return [], []

    current_year = datetime.now().year
    start_year = max(2017, current_year - 9) 
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
    """Fetches generated picks from the database for a specific date."""
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
                'time': datetime.fromisoformat(record['generated_date'].replace('Z', '+00:00')).strftime('%I:%M %p'), 
                'white_balls': white_balls,
                'powerball': int(record['powerball'])
            })
        return formatted_picks
    except requests.exceptions.RequestException as e:
        return []
    except Exception as e:
        return []

def _get_official_draw_for_date_from_db(date_str):
    """Fetches an official draw from the database for a specific date."""
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
    """Compares a batch of generated picks against an official draw."""
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
    """Saves a manually entered official draw to Supabase."""
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
    """Saves a generated Powerball combination to Supabase."""
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
    """Fetches the history of generated numbers from Supabase."""
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
    """Compares a generated pick against historical official draws."""
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
    """Analyzes grouped patterns (pairs and triplets) within defined ranges across years."""
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
    """Analyzes sum trends and identifies missing sums."""
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
    """Analyzes Powerball draw trends by weekday."""
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
        even_count = 5 - odd_count 

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
    """Analyzes grouped patterns (pairs and triplets) within a specific number range, grouped by year."""
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
        
        formatted_pairs = [{'pattern': list(p), 'count': c} for p, c in year_pairs.items()]
        formatted_triplets = [{'pattern': list(t), 'count': c} for t, c in year_triplets.items()]

        yearly_data_structured.append({
            "year": int(year),
            "pairs": formatted_pairs,
            "triplets": formatted_triplets,
            "total_unique_patterns": len(formatted_pairs) + len(formatted_triplets)
        })

    yearly_data_structured.sort(key=lambda x: x['year'])
    
    return yearly_data_structured

def get_boundary_crossing_pairs_trends(df_source, selected_pair_tuple=None):
    """Analyzes trends of numbers crossing decade boundaries."""
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
    start_year = max(2017, current_year - 9) 
    all_years = sorted([y for y in df_copy['Draw Date_dt'].dt.year.unique().tolist() if y >= start_year])


    yearly_boundary_pair_counts = defaultdict(lambda: defaultdict(int)) 
    overall_boundary_pair_counts = defaultdict(int) 

    for _, row in df_copy.iterrows():
        year = row['Draw Date_dt'].year
        if year < start_year: 
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
    """Analyzes historical draws for special number patterns (tens-apart, same last digit, repeating digit)."""
    if df_source.empty:
        return {
            'yearly_data': [], 
            'recent_trends': [],
            'tens_apart_patterns_overall': [],
            'same_last_digit_patterns_overall': [],
            'repeating_digit_patterns_overall': []
        }

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])

    if df_copy.empty:
         return {
            'yearly_data': [], 
            'recent_trends': [],
            'tens_apart_patterns_overall': [],
            'same_last_digit_patterns_overall': [],
            'repeating_digit_patterns_overall': []
        }

    current_year = datetime.now().year
    years_in_data = sorted([y for y in df_copy['Draw Date_dt'].dt.year.unique() if y >= 2017 and y <= current_year])

    all_yearly_patterns_data = []
    recent_special_trends = []

    all_tens_apart_pairs_set = set() 
    for n1 in range(1, 60):
        for diff in [10, 20, 30, 40, 50]:
            n2 = n1 + diff
            if n2 <= 69:
                all_tens_apart_pairs_set.add(tuple(sorted((n1, n2))))

    same_last_digit_groups_full = defaultdict(list)
    for i in range(1, 70):
        last_digit = i % 10
        same_last_digit_groups_full[last_digit].append(i)
    
    repeating_digit_numbers = [11, 22, 33, 44, 55, 66]

    overall_tens_apart_counts = defaultdict(int)
    overall_same_last_digit_counts = defaultdict(int)
    overall_repeating_digit_counts = defaultdict(int)

    for idx, row in df_copy.iterrows(): 
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])] )
        white_ball_set = set(white_balls)
        
        for pair in combinations(white_balls, 2):
            sorted_pair = tuple(sorted(pair))
            if sorted_pair in all_tens_apart_pairs_set:
                overall_tens_apart_counts[sorted_pair] += 1

        for last_digit, full_group_numbers in same_last_digit_groups_full.items():
            intersection_with_draw = white_ball_set.intersection(set(full_group_numbers))
            if len(intersection_with_draw) >= 2:
                for r in range(2, len(intersection_with_draw) + 1):
                    for pattern_combo in combinations(sorted(list(intersection_with_draw)), r):
                        overall_same_last_digit_counts[pattern_combo] += 1

        drawn_repeating_digits = [n for n in repeating_digit_numbers if n in white_ball_set]
        if len(drawn_repeating_digits) >= 2:
            for r in range(2, len(drawn_repeating_digits) + 1):
                for pattern_combo in combinations(sorted(drawn_repeating_digits), r):
                    overall_repeating_digit_counts[pattern_combo] += 1

    formatted_tens_apart_overall = [{'pattern': list(p), 'count': c} for p, c in overall_tens_apart_counts.items()]
    formatted_tens_apart_overall.sort(key=lambda x: (-x['count'], str(x['pattern'])))

    formatted_same_last_digit_overall = [{'pattern': list(p), 'count': c} for p, c in overall_same_last_digit_counts.items()]
    formatted_same_last_digit_overall.sort(key=lambda x: (-x['count'], str(x['pattern'])))

    formatted_repeating_digit_overall = [{'pattern': list(p), 'count': c} for p, c in overall_repeating_digit_counts.items()]
    formatted_repeating_digit_overall.sort(key=lambda x: (-x['count'], str(x['pattern'])))

    for year in years_in_data:
        yearly_df = df_copy[df_copy['Draw Date_dt'].dt.year == year].copy()
        
        if yearly_df.empty:
            all_yearly_patterns_data.append({
                'year': int(year),
                'total_draws': 0,
                'tens_apart_patterns': [],
                'same_last_digit_patterns': [],
                'repeating_digit_patterns': []
            })
            continue

        tens_apart_counts_year = defaultdict(int)
        same_last_digit_counts_year = defaultdict(int)
        repeating_digit_counts_year = defaultdict(int)
        
        total_draws_in_year = len(yearly_df)

        for idx, row in yearly_df.iterrows():
            white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])] )
            white_ball_set = set(white_balls)
            
            for pair in combinations(white_balls, 2):
                sorted_pair = tuple(sorted(pair))
                if sorted_pair in all_tens_apart_pairs_set:
                    tens_apart_counts_year[sorted_pair] += 1

            for last_digit, full_group_numbers in same_last_digit_groups_full.items():
                intersection_with_draw = white_ball_set.intersection(set(full_group_numbers))
                if len(intersection_with_draw) >= 2:
                    for r in range(2, len(intersection_with_draw) + 1):
                        for pattern_combo in combinations(sorted(list(intersection_with_draw)), r):
                            same_last_digit_counts_year[pattern_combo] += 1

            drawn_repeating_digits = [n for n in repeating_digit_numbers if n in white_ball_set]
            if len(drawn_repeating_digits) >= 2:
                for r in range(2, len(drawn_repeating_digits) + 1):
                    for pattern_combo in combinations(sorted(drawn_repeating_digits), r):
                        repeating_digit_counts_year[pattern_combo] += 1

        formatted_tens_apart_year = [{'pattern': list(p), 'count': c} for p, c in tens_apart_counts_year.items()]
        formatted_tens_apart_year.sort(key=lambda x: (-x['count'], str(x['pattern'])))

        formatted_same_last_digit_year = [{'pattern': list(p), 'count': c} for p, c in same_last_digit_counts_year.items()]
        formatted_same_last_digit_year.sort(key=lambda x: (-x['count'], str(x['pattern'])))

        formatted_repeating_digit_year = [{'pattern': list(p), 'count': c} for p, c in repeating_digit_counts_year.items()]
        formatted_repeating_digit_year.sort(key=lambda x: (-x['count'], str(x['pattern'])))

        all_yearly_patterns_data.append({
            'year': int(year),
            'total_draws': total_draws_in_year,
            'tens_apart_patterns': formatted_tens_apart_year,
            'same_last_digit_patterns': formatted_same_last_digit_year,
            'repeating_digit_patterns': formatted_repeating_digit_year
        })
    
    all_yearly_patterns_data.sort(key=lambda x: x['year'], reverse=True)

    one_year_ago = datetime.now() - timedelta(days=365)
    recent_data_df = df_copy[df_copy['Draw Date_dt'] >= one_year_ago].copy()
    recent_data_df = recent_data_df.sort_values(by='Draw Date_dt', ascending=False) 

    for idx, row in recent_data_df.iterrows():
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])] )
        white_ball_set = set(white_balls)
        draw_date_str = row['Draw Date_dt'].strftime('%Y-%m-%d')

        tens_apart_present = "No"
        for pair in combinations(white_balls, 2):
            if tuple(sorted(pair)) in all_tens_apart_pairs_set: 
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
            'repeating_digit': repeating_digit_present,
            'white_balls': white_balls
        })

    return {
        'yearly_data': all_yearly_patterns_data,
        'recent_trends': recent_special_trends,
        'tens_apart_patterns_overall': formatted_tens_apart_overall,
        'same_last_digit_patterns_overall': formatted_same_last_digit_overall,
        'repeating_digit_patterns_overall': formatted_repeating_digit_overall
    }

def get_white_ball_frequency_by_period(df_source, period_type='year'):
    """Calculates the frequency of each white ball per specified period."""
    if df_source.empty:
        return {}, []

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])

    if df_copy.empty:
        return {}, []

    current_year = datetime.now().year
    start_year = max(2017, current_year - 9) 
    years_to_analyze = range(start_year, current_year + 1)

    df_filtered_years = df_copy[df_copy['Draw Date_dt'].dt.year.isin(years_to_analyze)].copy()

    if period_type == 'year':
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str)
    elif period_type == 'half_year':
        df_filtered_years['half'] = (df_filtered_years['Draw Date_dt'].dt.month - 1) // 6 + 1
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str) + ' H' + df_filtered_years['half'].astype(str)
    elif period_type == 'quarter':
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str) + ' Q' + df_filtered_years['Draw Date_dt'].dt.quarter.astype(str)
    else:
        df_filtered_years['period_label'] = df_filtered_years['Draw Date_dt'].dt.year.astype(str)
        period_type = 'year' 

    all_period_labels = sorted(df_filtered_years['period_label'].unique().tolist())

    period_freq_data = {wb: {label: 0 for label in all_period_labels} for wb in range(1, 70)}

    for _, row in df_filtered_years.iterrows():
        period_label = row['period_label']
        white_balls = [int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])]
        for wb in white_balls:
            if 1 <= wb <= 69: 
                period_freq_data[wb][period_label] += 1
    
    formatted_data = {}
    for wb_num, period_counts in period_freq_data.items():
        formatted_data[wb_num] = sorted([
            {'period_label': label, 'frequency': count} 
            for label, count in period_counts.items()
        ], key=lambda x: x['period_label'])
        
    return formatted_data, all_period_labels 

def get_consecutive_numbers_yearly_trends(df_source):
    """Calculates the percentage of draws containing consecutive numbers for each year."""
    if df_source.empty:
        return {'yearly_data': [], 'years': [], 'all_consecutive_pairs_flat': []}

    df_copy = df_source.copy()
    if 'Draw Date_dt' not in df_copy.columns:
        df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])
    
    if df_copy.empty:
        return {'yearly_data': [], 'years': [], 'all_consecutive_pairs_flat': []}

    current_year = datetime.now().year
    start_year = max(2017, current_year - 9) 
    years_to_analyze = range(start_year, current_year + 1)

    yearly_trends = []
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
                    for sequence in current_draw_consecutive_sequences: 
                        sequence_tuple = tuple(sequence) 
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

    flat_consecutive_sequences_list = []
    for sequence_tuple, data in all_consecutive_sequences_aggregated.items(): 
        flat_consecutive_sequences_list.append({
            'sequence': list(sequence_tuple), 
            'count': data['count'],
            'dates': sorted(list(set(data['dates'])), reverse=True) 
        })
    
    flat_consecutive_sequences_list.sort(key=lambda x: (-x['count'], x['sequence'])) 
    yearly_trends.sort(key=lambda x: x['year'])
    
    return {
        'yearly_data': yearly_trends, 
        'years': list(years_to_analyze), 
        'all_consecutive_pairs_flat': flat_consecutive_sequences_list 
    }

POSITIONAL_ANALYSIS_CONFIG = {
    "1-10": {"range": (1, 10), "positions": [1]},
    "11-20": {"range": (11, 20), "positions": [2, 3]},
    "21-30": {"range": (21, 30), "positions": [1, 2, 3, 4, 5]},
    "31-40": {"range": (31, 40), "positions": [1, 2, 3, 4, 5]},
    "41-50": {"range": (41, 50), "positions": [1, 2, 3, 4, 5]},
    "51-60": {"range": (51, 60), "positions": [3, 4, 5]},
    "61-69": {"range": (61, 69), "positions": [3, 4, 5]},
}

def get_positional_range_frequency_analysis(df_source):
    """Analyzes frequency and percentage of white balls appearing at specific sorted positions within predefined ranges, by year."""
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

    current_year = datetime.now().year
    years_to_analyze = sorted([y for y in df_copy['Draw Date_dt'].dt.year.unique() if 2017 <= y <= current_year])

    yearly_analysis_results = []

    for year in years_to_analyze:
        yearly_df = df_copy[df_copy['Draw Date_dt'].dt.year == year].copy()
        total_draws_in_year = len(yearly_df)

        if total_draws_in_year == 0:
            yearly_analysis_results.append({
                'year': int(year),
                'total_draws': 0,
                'data': []
            })
            continue

        counts = defaultdict(int)

        for idx, row in yearly_df.iterrows():
            white_balls_sorted = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])])

            for range_label, config in POSITIONAL_ANALYSIS_CONFIG.items():
                min_val, max_val = config["range"]
                target_positions = config["positions"]

                for pos_idx in range(len(white_balls_sorted)):
                    current_position = pos_idx + 1
                    current_ball_value = white_balls_sorted[pos_idx]

                    if min_val <= current_ball_value <= max_val and current_position in target_positions:
                        counts[(range_label, current_position)] += 1

        year_data = []
        for (range_label, position), count in counts.items():
            percentage = round((count / total_draws_in_year) * 100, 2) if total_draws_in_year > 0 else 0.0
            year_data.append({
                'range_label': range_label,
                'position': position,
                'count': count,
                'percentage': percentage
            })
        
        year_data.sort(key=lambda x: (x['range_label'], x['position']))

        yearly_analysis_results.append({
            'year': int(year),
            'total_draws': total_draws_in_year,
            'data': year_data
        })
    
    yearly_analysis_results.sort(key=lambda x: x['year'], reverse=True)

    return yearly_analysis_results
    
def get_powerball_position_frequency(df_source):
    """Calculates the frequency of each Powerball number."""
    if df_source.empty:
        return {}
    
    position_freq = defaultdict(lambda: defaultdict(int))

    for _, row in df_source.iterrows():
        powerball = int(row['Powerball'])
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6)])
        
        if white_balls and powerball < white_balls[0]:
            position_freq[powerball]['Lower than all WB'] += 1
        elif white_balls and powerball > white_balls[-1]:
            position_freq[powerball]['Higher than all WB'] += 1
        else:
            position_freq[powerball]['Within WB Range'] += 1 
        
        position_freq[powerball]['Total Draws'] += 1

    formatted_data = []
    for pb_num in sorted(position_freq.keys()):
        total_draws = position_freq[pb_num]['Total Draws']
        formatted_data.append({
            'Powerball': int(pb_num),
            'Total Draws': int(total_draws)
        })
    return formatted_data

def initialize_core_data():
    """Initializes global DataFrame, last draw, and historical sets from Supabase."""
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
            return True 
        else:
            print("Core historical data is empty after loading. df remains empty.")
            last_draw = pd.Series({
                'Draw Date': 'N/A', 'Number 1': 'N/A', 'Number 2': 'N/A',
                'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A',
                'Numbers': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A']
            }, dtype='object')
            return False 
    except Exception as e:
        print(f"An error occurred during initial core data loading: {e}")
        traceback.print_exc()
        return False 

def get_cached_analysis(key, compute_function, *args, **kwargs):
    """Retrieves cached analysis results or computes and caches them."""
    global analysis_cache, last_analysis_cache_update
    
    serializable_args = [arg for arg in args if not isinstance(arg, pd.DataFrame)]
    serializable_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, pd.DataFrame)}

    cache_key_full = f"{key}_{json.dumps(serializable_args)}_{json.dumps(serializable_kwargs)}"

    if cache_key_full in analysis_cache and (datetime.now() - last_analysis_cache_update).total_seconds() < CACHE_EXPIRATION_SECONDS:
        print(f"Serving '{cache_key_full}' from cache.")
        return analysis_cache[cache_key_full]
    
    print(f"Computing and caching '{cache_key_full}'.")
    computed_data = compute_function(*args, **kwargs) 
    
    analysis_cache[cache_key_full] = computed_data
    last_analysis_cache_update = datetime.now()
    return computed_data

def invalidate_analysis_cache():
    """Invalidates the analysis cache."""
    global analysis_cache, last_analysis_cache_update
    analysis_cache = {}
    last_analysis_cache_update = datetime.min
    print("Analysis cache invalidated.")

def get_consecutive_trends_for_df(df_to_analyze):
    """Helper to get consecutive trends for a specific DataFrame."""
    if df_to_analyze.empty:
        return []

    trend_data = []
    for idx, row in df_to_analyze.iterrows():
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])])
        consecutive_sequences = _find_consecutive_sequences(white_balls)
        
        trend_data.append({
            'draw_date': row['Draw Date_dt'].strftime('%Y-%m-%d'),
            'consecutive_present': "Yes" if consecutive_sequences else "No", 
            'consecutive_sequences': consecutive_sequences 
        })
    return trend_data

# --- New Helper Functions for Custom Combinations Page ---

def _get_draws_for_month(year, month):
    """
    Fetches all Powerball draws for a given year and month from the global DataFrame `df`.
    Ensures `df` is loaded if empty.
    """
    global df
    if df.empty:
        initialize_core_data() 
        if df.empty:
            return pd.DataFrame() 

    if 'Draw Date_dt' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Draw Date_dt']):
        df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'], errors='coerce')
        df.dropna(subset=['Draw Date_dt'], inplace=True)
        if df.empty: return pd.DataFrame()

    monthly_draws = df[(df['Draw Date_dt'].dt.year == year) & (df['Draw Date_dt'].dt.month == month)]
    return monthly_draws

def _compute_two_months_unpicked_and_most_picked(df_source_placeholder, year, month):
    """
    Actual computation for unpicked and most picked numbers for a month.
    Designed to be called by _get_two_months_unpicked_and_most_picked and cached.
    `df_source_placeholder` is a placeholder because `get_cached_analysis` passes `df` here,
    but this function *must* use the global `df` or explicitly fetch to avoid issues with
    DataFrame being passed as a cache key.
    """
    all_possible_white_balls = set(range(1, 70))
    
    monthly_draws = _get_draws_for_month(year, month) 
    
    if monthly_draws.empty:
        return sorted(list(all_possible_white_balls)), [] 

    picked_counts = defaultdict(int)
    for _, row in monthly_draws.iterrows():
        white_balls = [int(row[f'Number {i}']) for i in range(1, 6) if pd.notna(row[f'Number {i}'])]
        for num in white_balls:
            picked_counts[num] += 1
    
    picked_numbers_in_month = set(picked_counts.keys())
    
    unpicked_numbers = sorted(list(all_possible_white_balls - picked_numbers_in_month))
    most_picked_numbers = sorted([num for num, count in picked_counts.items() if count > 1])

    return unpicked_numbers, most_picked_numbers

def _get_two_months_unpicked_and_most_picked(year, month):
    """
    Wrapper function to get cached unpicked and most picked numbers for a specific month
    for the custom combinations page.
    """
    cache_key_prefix = f"monthly_trends_custom_{year}_{month}"
    
    # Pass df as a parameter to get_cached_analysis so it's available to _compute_two_months_unpicked_and_most_picked
    # even though get_cached_analysis will filter it out for the cache key.
    return get_cached_analysis(cache_key_prefix, _compute_two_months_unpicked_and_most_picked, df, year, month)

def _get_current_month_hot_numbers(df_source):
    """Identifies numbers appearing more than once in the current (incomplete) month."""
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
        monthly_counts[int(row['Powerball'])] += 1 

    hot_numbers = {num for num, count in monthly_counts.items() if count > 1}
    return hot_numbers

def _score_pick_for_patterns(white_balls, criteria_data):
    """Scores a generated white ball pick based on how well it aligns with pattern preferences."""
    score = 0
    wb_set = set(white_balls)
    
    if criteria_data['prioritize_grouped_patterns'] and criteria_data['most_frequent_grouped_patterns']:
        for pattern_info in criteria_data['most_frequent_grouped_patterns']:
            pattern_set = set(pattern_info['pattern'])
            if pattern_set.issubset(wb_set):
                score += int(pattern_info['count']) * 0.1 

    if criteria_data['prioritize_special_patterns']:
        all_special_patterns_for_scoring = []
        all_special_patterns_for_scoring.extend(criteria_data['most_frequent_special_patterns'].get('tens_apart_patterns_overall', []))
        all_special_patterns_for_scoring.extend(criteria_data['most_frequent_special_patterns'].get('same_last_digit_patterns_overall', []))
        all_special_patterns_for_scoring.extend(criteria_data['most_frequent_special_patterns'].get('repeating_digit_patterns_overall', []))
        
        for pattern_info in all_special_patterns_for_scoring:
            pattern_set = set(pattern_info['pattern'])
            if pattern_set.issubset(wb_set):
                score += int(pattern_info['count']) * 0.05 

    if criteria_data['prioritize_consecutive_patterns']:
        consecutive_sequences = _find_consecutive_sequences(white_balls)
        score += len(consecutive_sequences) * 5 
        if any(len(s) >= 3 for s in consecutive_sequences):
            score += 10 

    if criteria_data['prioritize_monthly_hot'] and criteria_data['current_month_hot_numbers']:
        hot_count = len(wb_set.intersection(criteria_data['current_month_hot_numbers']))
        score += hot_count * 2 

    return score

def generate_smart_picks(df_source, num_sets, excluded_numbers, num_from_group_a, odd_even_choice, sum_range_tuple, prioritize_monthly_hot, prioritize_grouped_patterns, prioritize_special_patterns, prioritize_consecutive_patterns, force_specific_pattern):
    """Generates Powerball picks based on a combination of hard and soft criteria."""
    if df_source.empty:
        raise ValueError("Historical data is empty. Cannot generate smart picks.")

    generated_sets = []
    max_overall_attempts = 5000 * num_sets 

    all_grouped_patterns_data = get_cached_analysis('grouped_patterns', get_grouped_patterns_over_years, df_source)
    all_special_patterns_data = get_cached_analysis('special_patterns_analysis', get_special_patterns_analysis, df_source)
    
    most_frequent_grouped_patterns = sorted([p for p in all_grouped_patterns_data if 'count' in p], 
                                            key=lambda x: x['count'], reverse=True)[:50]
    
    most_frequent_special_patterns = {
        'tens_apart_patterns_overall': all_special_patterns_data.get('tens_apart_patterns_overall', []),
        'same_last_digit_patterns_overall': all_special_patterns_data.get('same_last_digit_patterns_overall', []),
        'repeating_digit_patterns_overall': all_special_patterns_data.get('repeating_digit_patterns_overall', [])
    }

    current_month_hot_numbers = set()
    if prioritize_monthly_hot: 
        current_month_hot_numbers = _get_current_month_hot_numbers(df_source)

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
        max_attempts_per_set = max_overall_attempts // num_sets 

        while current_set_attempts < max_attempts_per_set:
            current_set_attempts += 1
            
            candidate_white_balls = []
            candidate_powerball = random.randint(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1])
            
            remaining_to_pick = 5
            temp_excluded = set(excluded_numbers)
            
            if force_specific_pattern: 
                for num in force_specific_pattern:
                    if not (GLOBAL_WHITE_BALL_RANGE[0] <= num <= GLOBAL_WHITE_BALL_RANGE[1]) or num in temp_excluded:
                        continue 
                candidate_white_balls.extend(force_specific_pattern)
                temp_excluded.update(force_specific_pattern)
                remaining_to_pick -= len(force_specific_pattern)
                
            if remaining_to_pick < 0: 
                continue
            
            available_pool = [n for n in range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1)
                              if n not in temp_excluded and n not in candidate_white_balls]

            if len(available_pool) < remaining_to_pick:
                continue 

            current_group_a_count = sum(1 for num in candidate_white_balls if num in group_a)
            needed_from_group_a = num_from_group_a - current_group_a_count

            temp_available_pool = list(available_pool) 
            
            if needed_from_group_a > 0:
                possible_group_a_from_pool = [n for n in temp_available_pool if n in group_a]
                if len(possible_group_a_from_pool) < needed_from_group_a:
                    continue 
                
                try:
                    selected_group_a = random.sample(possible_group_a_from_pool, needed_from_group_a)
                    candidate_white_balls.extend(selected_group_a)
                    temp_excluded.update(selected_group_a)
                    remaining_to_pick -= needed_from_group_a
                    
                    available_pool = [n for n in available_pool if n not in selected_group_a]
                except ValueError: 
                    continue
            elif needed_from_group_a < 0: 
                continue

            if remaining_to_pick > 0:
                if len(available_pool) < remaining_to_pick:
                    continue 
                try:
                    random_fill = random.sample(available_pool, remaining_to_pick)
                    candidate_white_balls.extend(random_fill)
                except ValueError: 
                    continue
            
            if len(set(candidate_white_balls)) != 5:
                continue
            
            candidate_white_balls = sorted(candidate_white_balls)

            even_count = sum(1 for num in candidate_white_balls if num % 2 == 0)
            odd_count = 5 - even_count
            
            expected_odd_even_split = odd_even_choice
            
            if expected_odd_even_split != "Any":
                current_split_str = f"{odd_count} Odd / {even_count} Even"
                if expected_odd_even_split == "All Even" and even_count != 5: continue
                if expected_odd_even_split == "All Odd" and odd_count != 5: continue
                if expected_odd_even_split not in ["All Even", "All Odd"] and current_split_str != expected_odd_even_split: continue

            current_sum = sum(candidate_white_balls)
            if sum_range_tuple and not (sum_range_tuple[0] <= current_sum <= sum_range_tuple[1]):
                continue

            if check_exact_match(candidate_white_balls): 
                continue
            
            last_draw_data = get_last_draw(df_source)
            if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
                last_white_balls_list = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
                if set(candidate_white_balls) == set(last_white_balls_list) and candidate_powerball == int(last_draw_data['Powerball']):
                    continue

            current_score = _score_pick_for_patterns(candidate_white_balls, criteria_for_scoring)

            if current_score > highest_score:
                highest_score = current_score
                best_pick_white_balls = candidate_white_balls
                best_pick_powerball = candidate_powerball
            
            if highest_score > 0 and current_set_attempts > max_attempts_per_set / 2: 
                 break
        
        if best_pick_white_balls:
            generated_sets.append({'white_balls': best_pick_white_balls, 'powerball': best_pick_powerball})
        else:
            raise ValueError(f"Could not generate a smart pick meeting all criteria after {max_attempts_per_set} attempts. Try adjusting filters or reducing strictness.")
            
    return generated_sets
    
# Initialize core data on app startup
initialize_core_data() 

# --- Flask Routes ---

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
    excluded_numbers_local = [int(num.strip()) for num in request.form.get('excluded_numbers_pair', '').split(",") if num.strip().isdigit()] if request.form.get('excluded_numbers_pair') else []
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
            excluded_numbers_local, df, selected_sum_range_tuple 
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
    
    excluded_numbers_local = [int(num.strip()) for num in request.form.get('excluded_numbers_group_a', '').split(",") if num.strip().isdigit()] if request.form.get('excluded_numbers_group_a') else []
    
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
    if df.empty:
        flash("Cannot perform monthly trends analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    monthly_trends_data = get_cached_analysis(
        'monthly_trends_and_streaks', 
        get_monthly_white_ball_analysis_data, 
        df, 
        num_top_wb=69, 
        num_top_pb=3,
        num_months_for_top_display=6 
    )
    
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_data=monthly_trends_data['monthly_data'], 
                           streak_numbers=monthly_trends_data['streak_numbers'])

@app.route('/sum_of_main_balls_analysis')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot display Sum of Main Balls Analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
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
                [], num_draws # Excluded numbers empty list for simulation
            )
            
            simulated_white_ball_freq_list = sim_results['white_ball_freq']
            simulated_powerball_freq_list = sim_results['powerball_freq']
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": "Please enter a valid number for Number of Simulations."}), 400
            else:
                flash("Please enter a valid number for Number of Simulations.", 'error')


        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                "simulated_white_ball_freq": simulated_white_ball_freq_list,
                "simulated_powerball_freq": simulated_powerball_freq_list,
                "num_simulations": num_draws_display
            })
        else:
            return render_template('simulate_multiple_draws.html', 
                                simulated_white_ball_freq=simulated_white_ball_freq_list, 
                                simulated_powerball_freq=simulated_powerball_freq_list,   
                                num_simulations=num_draws_display,
                                selected_odd_even_choice=odd_even_choice_display)

    return render_template('simulate_multiple_draws.html', 
                           simulated_white_ball_freq=[], 
                           simulated_powerball_freq=[],   
                           num_simulations=100, 
                           selected_odd_even_choice="Any")

@app.route('/api/generate_single_draw', methods=['GET'])
def generate_single_draw_api():
    try:
        white_balls = sorted(random.sample(range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1), 5))
        powerball = random.randint(GLOBAL_POWERBALL_RANGE[0], GLOBAL_POWERBALL_RANGE[1])
        return jsonify({'success': True, 'white_balls': white_balls, 'powerball': powerball})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/number_age_distribution')
def number_age_distribution_route():
    number_age_counts, detailed_number_ages = get_cached_analysis('number_age_distribution', get_number_age_distribution, df)
    return render_template('number_age_distribution.html',
                           number_age_data=number_age_counts,
                           detailed_number_ages=detailed_number_ages)

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

@app.route('/powerball_frequency_by_year')
def powerball_frequency_by_year_route():
    yearly_pb_freq_data, years = get_cached_analysis('yearly_pb_freq', get_powerball_frequency_by_year, df)
    return render_template('powerball_frequency_by_year.html',
                           yearly_pb_freq_data=yearly_pb_freq_data,
                           years=years)

@app.route('/odd_even_trends')
def odd_even_trends_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    odd_even_trends = get_cached_analysis('odd_even_trends', get_odd_even_split_trends, df, last_draw_date_str_for_cache)
    return render_template('odd_even_trends.html',
                           odd_even_trends=odd_even_trends)

@app.route('/consecutive_trends')
def consecutive_trends_route():
    if df.empty:
        flash("Cannot display Consecutive Trends: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    consecutive_trends = get_cached_analysis('consecutive_trends', get_consecutive_numbers_trends, df, last_draw_date_str_for_cache)
    
    yearly_consecutive_data_full = get_cached_analysis('consecutive_yearly_trends', get_consecutive_numbers_yearly_trends, df)
    
    yearly_consecutive_percentage_data = yearly_consecutive_data_full['yearly_data']
    years_for_dropdown = yearly_consecutive_data_full['years']
    all_consecutive_pairs_flat = yearly_consecutive_data_full['all_consecutive_pairs_flat']

    return render_template('consecutive_trends.html',
                           consecutive_trends=consecutive_trends,
                           yearly_consecutive_percentage_data=yearly_consecutive_percentage_data,
                           years_for_dropdown=years_for_dropdown,
                           all_consecutive_pairs_flat=all_consecutive_pairs_flat)

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

@app.route('/consecutive_trends_by_year/<int:year>')
def consecutive_trends_by_year(year):
    df_year = df[(df['Draw Date_dt'].dt.year == year)].copy()
    trends = get_consecutive_trends_for_df(df_year) 
    return jsonify(trends)

@app.route('/triplets_analysis')
def triplets_analysis_route():
    triplets_data = get_cached_analysis('triplets_analysis', get_most_frequent_triplets, df) 
    return render_template('triplets_analysis.html',
                           triplets_data=triplets_data)

@app.route('/grouped_patterns_analysis')
def grouped_patterns_analysis_route():
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
    
    special_patterns_data = get_cached_analysis('special_patterns_analysis', get_special_patterns_analysis, df)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(special_patterns_data)
    else:
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
    
    weekday_data = get_cached_analysis('weekday_all_trends', get_weekday_draw_trends, df, group_a_numbers_def=group_a)
    
    return render_template('weekday_trends.html', 
                           weekday_trends=weekday_data)

@app.route('/yearly_white_ball_trends')
def yearly_white_ball_trends_route():
    if df.empty:
        flash("Cannot display Yearly White Ball Trends: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    current_year = datetime.now().year
    start_year_for_display = max(2017, current_year - 9) 
    years_for_display = list(range(start_year_for_display, current_year + 1))

    return render_template('yearly_white_ball_trends.html',
                           years=years_for_display)

@app.route('/api/white_ball_trends')
def api_white_ball_trends_route():
    if df.empty:
        return jsonify({"error": "Historical data not loaded or is empty."}), 500
    
    period_type = request.args.get('period', 'year') 

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
    
@app.route('/positional_analysis')
def positional_analysis_route():
    if df.empty:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"error": "Historical data not loaded or is empty."}), 500
        else:
            flash("Cannot display Positional Analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
            return redirect(url_for('index'))

    positional_data = get_cached_analysis('positional_range_frequency', get_positional_range_frequency_analysis, df)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(positional_data)
    else:
        return render_template('positional_analysis.html', positional_data=positional_data)

@app.route('/ai_assistant')
def ai_assistant_route():
    try:
        return render_template('ai_assistant.html')
    except Exception as e:
        traceback.print_exc()
        flash("An error occurred loading the AI Assistant page. Please try again.", 'error')
        return redirect(url_for('index'))
        
@app.route('/smart_pick_generator')
def smart_pick_generator_route():
    global df, last_draw

    if df.empty or last_draw.empty:
        success = initialize_core_data() 
        if not success: 
            if df.empty:
                flash("Failed to load historical data. Please try again later.", 'error')
                return redirect(url_for('index'))
    
    generated_sets = []
    last_draw_dates = {}

    return render_template('smart_pick_generator.html', 
                           sum_ranges=SUM_RANGES,
                           group_a=group_a,
                           generated_sets=generated_sets, 
                           last_draw_dates=last_draw_dates, 
                           num_sets_to_generate=1, 
                           excluded_numbers='',     
                           num_from_group_a=2,      
                           odd_even_choice="Any",   
                           selected_sum_range="Any" 
                          )

@app.route('/generate_smart_picks_route', methods=['POST'])
def generate_smart_picks_route():
    if df.empty:
        return jsonify({'success': False, 'error': "Historical data not loaded or is empty. Please check Supabase connection."}), 500

    try:
        data = request.json 
        num_sets_to_generate = int(data.get('num_sets_to_generate', 1))
        excluded_numbers_input = data.get('excluded_numbers', '')
        excluded_numbers_local = [int(num.strip()) for num in excluded_numbers_input.split(',') if num.strip().isdigit()] if excluded_numbers_input else []
        
        num_from_group_a = int(data.get('num_from_group_a', 0))
        odd_even_choice = data.get('odd_even_choice', 'Any')
        
        selected_sum_range_label = data.get('sum_range_filter', 'Any')
        selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)

        prioritize_monthly_hot = data.get('prioritize_monthly_hot', False) 
        prioritize_grouped_patterns = data.get('prioritize_grouped_patterns', False)
        prioritize_special_patterns = data.get('prioritize_special_patterns', False)
        prioritize_consecutive_patterns = data.get('prioritize_consecutive_patterns', False)
        
        force_specific_pattern_input = data.get('force_specific_pattern', '') 
        force_specific_pattern = []
        if force_specific_pattern_input:
            force_specific_pattern = sorted([int(num.strip()) for num in force_specific_pattern_input.split(',') if num.strip().isdigit()])
            if not (2 <= len(force_specific_pattern) <= 3):
                raise ValueError("Forced specific pattern must contain 2 or 3 numbers.")
            for num in force_specific_pattern:
                if not (GLOBAL_WHITE_BALL_RANGE[0] <= num <= GLOBAL_WHITE_BALL_RANGE[1]):
                    raise ValueError(f"Forced number {num} is outside the valid white ball range (1-69).")
                if num in excluded_numbers_local:
                    raise ValueError(f"Forced number {num} is also in the excluded numbers list. Please remove it from excluded.")
            if len(set(force_specific_pattern)) != len(force_specific_pattern):
                raise ValueError("Forced specific pattern numbers must be unique.")

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
        
        last_draw_dates = {}
        if generated_sets:
            last_draw_dates = find_last_draw_dates_for_numbers(df, generated_sets[-1]['white_balls'], generated_sets[-1]['powerball'])

        return jsonify({
            'success': True,
            'generated_sets': generated_sets,
            'last_draw_dates': last_draw_dates
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"An unexpected error occurred: {e}"}), 500

