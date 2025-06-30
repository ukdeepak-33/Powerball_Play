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
import traceback # Ensure this is imported for logging errors

# --- Supabase Configuration ---
# IMPORTANT: These values should ALWAYS be set as environment variables in your deployment platform (e.g., Render, Vercel).\
# The 'default' values provided here are placeholders, ONLY used if the environment variable is NOT found.\
# If running locally for testing, you can set them in your shell or a .env file.
SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "YOUR_ACTUAL_SUPABASE_ANON_KEY_GOES_HERE") # REPLACE THIS IN YOUR ENV VARS
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "YOUR_ACTUAL_SUPABASE_SERVICE_ROLE_KEY_GOES_HERE") # REPLACE THIS IN YOUR ENV VARS

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

# Define number ranges for grouped patterns analysis and the new "starting pair" logic
NUMBER_RANGES = {
    "1-9": (1, 9),
    "10s": (10, 19),
    "20s": (20, 29),
    "30s": (30, 39),
    "40s": (40, 49),
    "50s": (50, 59),
    "60s": (60, 69)
}

# Specific ranges for the "Generate from Your Starting Pair" ascending logic
ASCENDING_GEN_RANGES = [
    (10, 19), # 10s
    (20, 29), # 20s
    (30, 39), # 30s
    (40, 49), # 40s
    (50, 59), # 50s
    (60, 69)  # 60s
]

# NEW: Sum Ranges for Grouped Sum Analysis (Corrected ranges based on user input)
SUM_RANGES = {
    "Any": None, # Special value for no sum filtering
    "Zone A (60-99)": (60, 99),
    "Zone B (100-129)": (100, 129),
    "Zone C (130-159)": (130, 159), 
    "Zone D (160-189)": (160, 189),
    "Zone E (190-220)": (190, 220),
    "Zone F (221-249)": (221, 249), 
    "Zone G (250-300)": (250, 300)  
}

# NEW: Low/High split definition for white balls
LOW_NUMBER_MAX = 34 # Numbers 1-34 are considered 'low'
HIGH_NUMBER_MIN = 35 # Numbers 35-69 are considered 'high'

# Days of the week Powerball is drawn
POWERBALL_DRAW_DAYS = ['Monday', 'Wednesday', 'Saturday']

# --- Core Utility Functions (All helpers defined here) ---

def _get_supabase_headers(is_service_key=False):
    key = SUPABASE_SERVICE_KEY if is_service_key else SUPABASE_ANON_KEY
    
    # --- TEMPORARY DEBUG PRINT: Remove these lines after verifying your keys are loaded ---
    # print(f"[DEBUG] Supabase URL used: {SUPABASE_PROJECT_URL}")
    # print(f"[DEBUG] Using {'Service Key' if is_service_key else 'Anon Key'}: {key[:5]}...{key[-5:]}") # Print only first/last 5 chars for security
    # --- END TEMPORARY DEBUG PRINT ---

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

def generate_powerball_numbers(df_source, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None, selected_sum_range_tuple=None, is_simulation=False):
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 2000 # Increased max attempts due to added sum constraint
    attempts = 0
    while attempts < max_attempts:
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        
        if len(available_numbers) < 5:
            raise ValueError("Not enough available numbers for white balls after exclusions and range constraints.")
            
        white_balls = sorted(random.sample(available_numbers, 5))

        # --- New: Sum Range Check ---
        if selected_sum_range_tuple:
            current_sum = sum(white_balls)
            if not (selected_sum_range_tuple[0] <= current_sum <= selected_sum_range_tuple[1]):
                attempts += 1
                continue # Retry if sum is not in the desired range
        # --- End New: Sum Range Check ---

        group_a_numbers = [num for num in white_balls if num in group_a]
        if len(group_a_numbers) < 2:
            attempts += 1
            continue

        powerball = random.randint(powerball_range[0], powerball_range[1])

        # Skip these checks if it's a simulation
        if not is_simulation:
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

        # Apply odd/even choice for simulation if specified
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


        if high_low_balance is not None:
            low_numbers_count = sum(1 for num in white_balls if num <= 34)
            high_numbers_count = sum(1 for num in white_balls if num >= 35)
            if low_numbers_count != high_low_balance[0] or high_numbers_count != high_low_balance[1]:
                attempts += 1
                continue
        
        break
    else:
        raise ValueError("Could not generate a unique combination meeting all criteria after many attempts. Try adjusting filters.")

    return white_balls, powerball

def generate_from_group_a(df_source, num_from_group_a, white_ball_range, powerball_range, excluded_numbers, selected_sum_range_tuple=None):
    if df_source.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 2000 # Increased max attempts due to added sum constraint
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
            
            # --- New: Sum Range Check ---
            if selected_sum_range_tuple:
                current_sum = sum(white_balls)
                if not (selected_sum_range_tuple[0] <= current_sum <= selected_sum_range_tuple[1]):
                    attempts += 1
                    continue # Retry if sum is not in the desired range
            # --- End New: Sum Range Check ---

            powerball = random.randint(powerball_range[0], powerball_range[1])

            if check_exact_match(white_balls): 
                attempts += 1
                continue

            break
        except ValueError as e:
            # print(f"Attempt failed during group_a strategy: {e}. Retrying...") # For debugging
            attempts += 1
            continue
        except IndexError: # Catches errors if, for example, random.choice gets an empty sequence
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

    # Validate user-provided numbers
    if not (white_ball_range[0] <= num1 <= white_ball_range[1] and
            white_ball_range[0] <= num2 <= white_ball_range[1]):
        raise ValueError(f"Provided numbers ({num1}, {num2}) must be within the white ball range ({white_ball_range[0]}-{white_ball_range[1]}).")
    
    if num1 == num2:
        raise ValueError("The two provided white balls must be unique.")
    
    if num1 in excluded_numbers or num2 in excluded_numbers:
        raise ValueError(f"One or both provided numbers ({num1}, {num2}) are in the excluded list.")

    initial_white_balls = sorted([num1, num2]) # Ensure they are sorted
    
    max_attempts_overall = 2000 # Increased max attempts due to added sum constraint
    attempts_overall = 0

    while attempts_overall < max_attempts_overall:
        candidate_white_balls_generated = []
        temp_current_min = initial_white_balls[-1] + 1 # Start for the 3rd number, e.g., 14 -> 15, 25 -> 26
        
        try:
            # Generate the 3rd, 4th, and 5th white balls
            for i in range(3): # We need to generate 3 more numbers
                possible_nums_for_slot = []
                
                # Determine the eligible "tens" range to start from
                start_range_idx = -1
                for idx, (range_min, range_max) in enumerate(ASCENDING_GEN_RANGES):
                    # Find the first range that either contains temp_current_min OR starts after temp_current_min
                    if temp_current_min <= range_max and temp_current_min >= range_min: # temp_current_min is within this range
                        start_range_idx = idx
                        break
                    elif temp_current_min < range_min: # temp_current_min is before this range's start
                        start_range_idx = idx
                        break
                
                if start_range_idx == -1: # temp_current_min is already beyond the 60s range max (69)
                    # This implies we can't find 3 more numbers in ascending "tens" ranges.
                    # Instead of raising an error here, let the outer loop try again by continuing
                    raise ValueError("Not enough space in ascending ranges to complete combination.")

                # Build the pool of possible numbers for the current slot
                eligible_ranges = ASCENDING_GEN_RANGES[start_range_idx:]
                
                # Populate possible numbers respecting temp_current_min, ranges, uniqueness, and exclusions
                for range_min, range_max in eligible_ranges:
                    actual_start_val = max(temp_current_min, range_min) # Ensure ascending order and range constraint
                    
                    for num in range(actual_start_val, range_max + 1):
                        if num not in excluded_numbers and \
                           num not in initial_white_balls and \
                           num not in candidate_white_balls_generated:
                            possible_nums_for_slot.append(num)
                
                if not possible_nums_for_slot:
                    raise ValueError(f"No available numbers for slot {i+3}. Current min: {temp_current_min}, initial: {initial_white_balls}, generated: {candidate_white_balls_generated}")

                # Pick one number for the current slot
                picked_num = random.choice(possible_nums_for_slot)
                candidate_white_balls_generated.append(picked_num)
                temp_current_min = picked_num + 1 # Next number must be strictly greater
            
            # Combine user's numbers with generated numbers and sort them
            final_white_balls = sorted(initial_white_balls + candidate_white_balls_generated)
            
            # --- New: Sum Range Check ---
            if selected_sum_range_tuple:
                current_sum = sum(final_white_balls)
                if not (selected_sum_range_tuple[0] <= current_sum <= selected_sum_range_tuple[1]):
                    attempts_overall += 1
                    continue # Retry if sum is not in the desired range
            # --- End New: Sum Range Check ---

            # Generate Powerball
            powerball = random.randint(powerball_range[0], powerball_range[1])

            # Check for exact historical match of the 5 white balls
            if check_exact_match(final_white_balls):
                attempts_overall += 1
                continue # Retry if combination already exists historically

            # Check against last draw (existing logic from previous versions)
            last_draw_data = get_last_draw(df_source)
            if not last_draw_data.empty and last_draw_data.get('Draw Date') != 'N/A':
                last_white_balls = [int(last_draw_data['Number 1']), int(last_draw_data['Number 2']), int(last_draw_data['Number 3']), int(last_draw_data['Number 4']), int(last_draw_data['Number 5'])]
                if set(final_white_balls) == set(last_white_balls) and powerball == int(last_draw_data['Powerball']):
                    attempts_overall += 1
                    continue

            # If all checks pass, return the valid combination
            return final_white_balls, powerball

        except ValueError as e:
            # print(f"Attempt failed during user-pair strategy: {e}. Retrying... (Overall attempt {attempts_overall+1})") # For debugging
            attempts_overall += 1
            continue
        except IndexError: # Catches errors if, for example, random.choice gets an empty sequence
            attempts_overall += 1
            continue
    else:
        raise ValueError("Could not generate a unique combination with the provided pair and ascending range constraint meeting all criteria after many attempts. Try adjusting filters.")


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

def get_monthly_white_ball_analysis_data(dataframe, num_top_wb=69, num_top_pb=3, num_months_for_top_display=6):
    """
    Analyzes monthly trends for white balls and Powerballs, including all numbers drawn per month
    for the last N months, and numbers on consecutive monthly streaks (for completed months).
    """
    if dataframe.empty:
        return {'monthly_data': [], 'streak_numbers': {'3_month_streaks': [], '4_month_streaks': [], '5_month_streaks': []}}

    df_sorted = dataframe.sort_values(by='Draw Date_dt', ascending=False).copy()
    df_sorted['YearMonth'] = df_sorted['Draw Date_dt'].dt.to_period('M')
    unique_months_periods = sorted(df_sorted['YearMonth'].unique(), reverse=True)

    monthly_display_data = [] # Renamed for clarity on what's being displayed
    current_period = pd.Period(datetime.now(), freq='M')
    
    processed_months_count = 0
    # Iterate through unique months, including the current month for display up to `num_months_for_top_display`
    for period in unique_months_periods:
        # If we've processed enough months (including current one if applicable), break
        if processed_months_count >= num_months_for_top_display:
            # We want exactly num_months_for_top_display entries.
            # If the current month pushed us over, and we already added it, stop.
            # If the current month is the last one to be added, break after adding.
            # This logic ensures we get exactly the last N months, including current if present.
            if not (period == current_period and processed_months_count < num_months_for_top_display):
                break


        month_df = df_sorted[df_sorted['YearMonth'] == period]
        if month_df.empty:
            continue

        is_current_month_flag = (period == current_period)

        # White Balls Drawn
        drawn_white_balls_set = set()
        wb_monthly_counts = defaultdict(int) 
        for _, row in month_df.iterrows():
            for i in range(1, 6):
                num = int(row[f'Number {i}'])
                drawn_white_balls_set.add(num)
                wb_monthly_counts[num] += 1
        
        # Prepare drawn white balls with counts (sorted by number for display)
        drawn_wb_with_counts = sorted([{'number': n, 'count': wb_monthly_counts[n]} for n in drawn_white_balls_set], key=lambda x: x['number'])

        # Calculate not picked white balls
        all_possible_white_balls = set(range(GLOBAL_WHITE_BALL_RANGE[0], GLOBAL_WHITE_BALL_RANGE[1] + 1))
        not_picked_white_balls = sorted(list(all_possible_white_balls - drawn_white_balls_set))

        # Powerballs Drawn
        drawn_powerballs_set = set()
        pb_monthly_counts = defaultdict(int) 
        for _, row in month_df.iterrows():
            pb_num = int(row['Powerball'])
            drawn_powerballs_set.add(pb_num)
            pb_monthly_counts[pb_num] += 1

        # Prepare top Powerballs (sorted by frequency as before, then by number for tie-breaking)
        sorted_pb_freq = sorted(pb_monthly_counts.items(), key=lambda item: (-item[1], item[0])) # Sort by count DESC, then number ASC
        top_pb = [{'number': int(n), 'count': int(c)} for n, c in sorted_pb_freq[:num_top_pb]]

        # Calculate not picked powerballs
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
    
    # Sort monthly data from oldest to newest for display
    monthly_display_data.sort(key=lambda x: datetime.strptime(x['month'], '%B %Y'))

    # --- Part 2: Monthly Streaks (ONLY for completed months) ---
    # This logic remains unchanged, as streaks should be based on full data.
    numbers_per_completed_month = defaultdict(set)
    for period in unique_months_periods:
        if period == current_period: # Exclude current month for streak calculation
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
        
        for i in range(len(completed_months_sorted) - 1, -1, -1): # Iterate backwards from most recent completed month
            month_period = completed_months_sorted[i]
            if num in numbers_per_completed_month[month_period]:
                current_streak_length += 1
            else:
                break # Streak broken
        
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
        'monthly_data': monthly_display_data, # Return the renamed variable
        'streak_numbers': streak_numbers
    }


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

    return temp_df[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum', 'Draw Date_dt']], sum_freq_list, min_sum, max_sum, avg_sum

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
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum', 'Draw Date_dt']]

def simulate_multiple_draws(df_source, group_a, odd_even_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
    if df_source.empty: 
        # Return empty lists for white ball and powerball frequencies
        return {'white_ball_freq': [], 'powerball_freq': []}
    
    white_ball_results = defaultdict(int)
    powerball_results = defaultdict(int)

    for _ in range(num_draws):
        try:
            # Pass is_simulation=True and the odd_even_choice
            white_balls, powerball = generate_powerball_numbers(
                df_source, group_a, odd_even_choice, "No Combo", # Combo choice not relevant for this simulation
                white_ball_range, powerball_range, excluded_numbers, is_simulation=True
            )
            
            for wb in white_balls:
                white_ball_results[wb] += 1
            powerball_results[powerball] += 1

        except ValueError:
            pass # Silently skip failed generations in simulation
    
    # Convert defaultdicts to sorted lists of dictionaries
    # Ensure all possible numbers from the full range are included, even if frequency is 0
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

def get_most_frequent_triplets(df_source): # Removed top_n parameter
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
    for triplet, count in sorted_triplets: # Iterate through all sorted triplets
        formatted_triplets.append({
            'triplet': list(triplet),
            'count': int(count)
        })
    
    print(f"[DEBUG-Triplets] Found {len(triplet_counts)} unique triplets. Returning all {len(formatted_triplets)}.")
    return formatted_triplets


# MODIFIED FUNCTION SIGNATURE AND LOGIC
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

        # Identify group_a numbers present in the draw - now using the global group_a directly
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
        return []

def analyze_generated_batch_against_official_draw(generated_picks_list, official_draw):
    """
    Analyzes a batch of generated picks against a single official draw result.
    Returns a summary of matches.
    """
    summary = {
        "Match 5 White Balls + Powerball": {"count": 0, "draws": []}, # Added 'draws' list for consistency
        "Match 5 White Balls Only": {"count": 0, "draws": []},
        "Match 4 White Balls + Powerball": {"count": 0, "draws": []},
        "Match 4 White Balls Only": {"count": 0, "draws": []},
        "Match 3 White Balls + Powerball": {"count": 0, "draws": []},
        "Match 3 White Balls Only": {"count": 0, "draws": []},
        "Match 2 White Balls + Powerball": {"count": 0, "draws": []},
        "Match 1 White Ball + Powerball": {"count": 0, "draws": []},
        "Match Powerball Only": {"count": 0, "draws": []},
        "No Match": {"count": 0, "draws": []} # Added 'draws' list for consistency
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
        # For batch analysis, we don't necessarily need to store all individual matching draws for each category
        # However, for consistency with check_generated_against_history, we will add dummy draws
        # if matchInfo.draws is expected to be present. Otherwise, simply remove the .append line
        summary[category]["draws"].append({
            "date": official_draw['Draw Date'], # The official draw date itself
            "white_balls": official_white_balls,
            "powerball": official_powerball
        })

    # The 'draws' list will contain duplicates if multiple generated picks match the *same* official draw.
    # If uniqueness is desired here, you would need to process this list further.
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
# Utility function for Supabase search (used by strict_positional_search_route)
def supabase_search_draws(query_params):
    url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
    headers = _get_supabase_headers(is_service_key=False) # Use anon key for reads

    try:
        response = requests.get(url, headers=headers, params=query_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during Supabase search: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Supabase error response: {e.response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in supabase_search_draws: {e}")
        traceback.print_exc()
        return []

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
    # Initialize the results dictionary correctly
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

# MODIFIED FUNCTION: get_sum_trends_and_gaps_data
def get_sum_trends_and_gaps_data(df_source):
    """
    Analyzes sums of white balls, identifies their last appearance dates,
    finds missing sums, and groups sums into predefined ranges, with enhanced details.
    """
    if df_source.empty:
        return {
            'min_possible_sum': 15,
            'max_possible_sum': 335,
            'appeared_sums_details': [],
            'missing_sums': [],
            'grouped_sums_analysis': {}
        }

    df_copy = df_source.copy()
    
    # Ensure Draw Date is datetime and sum columns are numeric
    df_copy['Draw Date_dt'] = pd.to_datetime(df_copy['Draw Date'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Draw Date_dt'])
    for i in range(1, 6):
        col = f'Number {i}'
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)

    # Calculate sum for each row
    df_copy['Sum'] = df_copy[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)

    # 1. Find last appearance date for each unique sum
    last_appearance_by_sum_df = df_copy.groupby('Sum')['Draw Date_dt'].max().reset_index()
    last_appearance_by_sum_df['last_drawn_date'] = last_appearance_by_sum_df['Draw Date_dt'].dt.strftime('%Y-%m-%d')
    last_drawn_dates_map = last_appearance_by_sum_df.set_index('Sum')['last_drawn_date'].to_dict()

    # Get overall sum frequencies
    sum_freq_series = df_copy['Sum'].value_counts()
    sum_counts_map = sum_freq_series.to_dict()

    appeared_sums_details = sorted([
        {'sum': int(s), 'last_drawn_date': last_drawn_dates_map.get(s, 'N/A'), 'count': sum_counts_map.get(s, 0)}
        for s in sum_freq_series.index
    ], key=lambda x: x['sum'])

    # 2. Determine min/max possible sums (theoretical)
    min_possible_sum = 1 + 2 + 3 + 4 + 5 # Smallest possible sum for 5 distinct numbers from 1-69
    max_possible_sum = 69 + 68 + 67 + 66 + 65 # Largest possible sum for 5 distinct numbers from 1-69

    # 3. Identify missing sums (theoretical range vs. actual appeared)
    all_possible_sums = set(range(min_possible_sum, max_possible_sum + 1))
    actual_appeared_sums = set(df_copy['Sum'].unique())
    missing_sums = sorted(list(all_possible_sums - actual_appeared_sums))

    # 4. Grouped Sum Analysis Enhancement
    grouped_sums_analysis = {}
    # FIX: Iterate over items and explicitly check for None before unpacking
    for range_name, range_tuple in SUM_RANGES.items():
        if range_tuple is None: # Skip "Any" or any other None entry
            continue
        
        range_min, range_max = range_tuple # Now safe to unpack
        
        # Identify all sums within this range (both appeared and missing)
        sums_in_current_range = sorted(list(set(range(range_min, range_max + 1)).intersection(all_possible_sums)))

        # Filter appeared sums for this range
        appeared_in_range_details = [
            {'sum': s_data['sum'], 'last_drawn_date': s_data['last_drawn_date'], 'count': s_data['count']}
            for s_data in appeared_sums_details if range_min <= s_data['sum'] <= range_max
        ]
        
        # Sort by count descending for most frequent, then by sum ascending for tie-breaking
        most_frequent_sums = sorted(appeared_in_range_details, key=lambda x: (-x['count'], x['sum']))[:5]
        
        # Sort by count ascending for least frequent, then by sum ascending for tie-breaking
        # Exclude sums with 0 count (already handled by missing_sums)
        least_frequent_sums = sorted([s for s in appeared_in_range_details if s['count'] > 0], key=lambda x: (x['count'], x['sum']))[:5]

        # Calculate average frequency for sums that *have* appeared in this range
        total_freq_in_range = sum(s['count'] for s in appeared_in_range_details)
        if appeared_in_range_details:
            avg_freq_in_range = round(total_freq_in_range / len(appeared_in_range_details), 2)
        else:
            avg_freq_in_range = 0.0

        # Find the latest draw date for any sum within this range
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
            'all_appeared_sums_in_range': appeared_in_range_details # For comprehensive display
        }
    
    return {
        'min_possible_sum': min_possible_sum,
        'max_possible_sum': max_possible_sum,
        'appeared_sums_details': appeared_sums_details,
        'missing_sums': missing_sums,
        'grouped_sums_analysis': grouped_sums_analysis
    }

# NEW/MODIFIED FUNCTION: get_weekday_draw_trends
def get_weekday_draw_trends(df_source, group_a_numbers_def=None): # Keep group_a_numbers_def as an optional parameter
    """
    Analyzes various drawing trends (Low/High, Odd/Even, Consecutive, Sum, Group A)
    based on the day of the week (Monday, Wednesday, Saturday).
    """
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
        
        white_balls = sorted([int(row[f'Number {i}']) for i in range(1, 6)]) # Ensure sorted for consecutive check
        
        # Low/High Counts
        low_count = sum(1 for num in white_balls if LOW_NUMBER_MAX >= num >= 1)
        high_count = sum(1 for num in white_balls if HIGH_NUMBER_MIN <= num <= GLOBAL_WHITE_BALL_RANGE[1])
        
        # Odd/Even Counts
        odd_count = sum(1 for num in white_balls if num % 2 != 0)
        even_count = sum(1 for num in white_balls if num % 2 == 0)

        # Sum of Balls
        current_sum = sum(white_balls)

        # Use the provided group_a_numbers_def if it's not None, otherwise use the global group_a
        group_a_to_use = group_a_numbers_def if group_a_numbers_def is not None else group_a
        current_group_a_count = sum(1 for num in white_balls if num in group_a_to_use)

        # Consecutive Numbers Presence
        consecutive_present = False
        for i in range(len(white_balls) - 1):
            if white_balls[i] + 1 == white_balls[i+1]:
                consecutive_present = True
                break

        # Aggregate data
        weekday_stats[day_name]['total_draws'] += 1
        weekday_stats[day_name]['total_low_balls'] += low_count
        weekday_stats[day_name]['total_high_balls'] += high_count
        weekday_stats[day_name]['total_odd_balls'] += odd_count
        weekday_stats[day_name]['total_even_balls'] += even_count
        weekday_stats[day_name]['total_sum'] += current_sum
        weekday_stats[day_name]['total_group_a_balls'] += current_group_a_count
        if consecutive_present:
            weekday_stats[day_name]['consecutive_draws_count'] += 1
        
        # Low/High Splits frequency
        low_high_split_key = f"{low_count} Low / {high_count} High"
        weekday_stats[day_name]['low_high_splits'][low_high_split_key] += 1

        # Odd/Even Splits frequency
        odd_even_split_key = f"{odd_count} Odd / {even_count} Even"
        weekday_stats[day_name]['odd_even_splits'][odd_even_split_key] += 1
    
    final_results = {}
    for day in POWERBALL_DRAW_DAYS: # Ensure results are in defined order
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
            final_results[day] = { # Ensure all days are present, even if no draws or data
                'total_draws': 0, 'avg_low_balls': 0.0, 'avg_high_balls': 0.0,
                'avg_odd_balls': 0.0, 'avg_even_balls': 0.0, 'avg_sum': 0.0,
                'avg_group_a_balls': 0.0, 'consecutive_present_percentage': 0.0,
                'low_high_splits': [], 'odd_even_splits': []
            }
            
    return final_results


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
            if not last_draw.empty and 'Draw Date_dt' in last_draw and pd.notna(last_draw['Draw Date_dt']):
                last_draw['Draw Date'] = last_draw['Draw Date_dt'].strftime('%Y-%m-%d')
            else:
                 last_draw['Draw Date'] = 'N/A' # Fallback for display if datetime conversion fails
            
            # Populate historical_white_ball_sets
            historical_white_ball_sets.clear() # Clear existing set before repopulating
            for _, row in df.iterrows():
                white_balls_tuple = tuple(sorted([
                    int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), 
                    int(row['Number 4']), int(row['Number 5'])
                ]))
                historical_white_ball_sets.add(frozenset(white_balls_tuple))

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
        traceback.print_exc()

initialize_core_data()


# --- Analysis Cache Management ---
def get_cached_analysis(key, compute_function, *args, **kwargs):
    global analysis_cache, last_analysis_cache_update
    
    if key in analysis_cache and (datetime.now() - last_analysis_cache_update).total_seconds() < CACHE_EXPIRATION_SECONDS:
        print(f"Serving '{key}' from cache.")
        return analysis_cache[key]
    
    print(f"Computing and caching '{key}'.")
    # Pass args and kwargs directly to the compute_function
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
    # Pass SUM_RANGES to the index template
    return render_template('index.html', last_draw=last_draw_dict, sum_ranges=SUM_RANGES)

# Generation Routes (referenced directly in index.html forms)
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

    # --- New: Get selected sum range ---
    selected_sum_range_label = request.form.get('sum_range_filter', 'Any')
    selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)
    # --- End New ---

    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        white_balls, powerball = generate_powerball_numbers(
            df, group_a, odd_even_choice, combo_choice, white_ball_range_local, powerball_range_local, 
            excluded_numbers_local, high_low_balance, selected_sum_range_tuple, is_simulation=False
        )
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)

    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='generated',
                           sum_ranges=SUM_RANGES, # Pass sum_ranges back
                           selected_sum_range=selected_sum_range_label) # Pass selected sum range back for persistence


@app.route('/generate_with_user_pair', methods=['POST'])
def generate_with_user_pair_route():
    if df.empty:
        flash("Cannot generate with provided pair: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)

    user_pair_str = request.form.get('user_pair')
    # --- New: Get selected sum range ---
    selected_sum_range_label = request.form.get('sum_range_filter_pair', 'Any') # Unique name for this form's select
    selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)
    # --- End New ---

    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        if not user_pair_str:
            raise ValueError("Please enter two numbers for the starting pair.")
        
        # Use a regex or more robust split if needed, but for 'X, Y' split(',') is fine
        pair_parts = [int(num.strip()) for num in user_pair_str.split(',') if num.strip().isdigit()]
        if len(pair_parts) != 2:
            raise ValueError("Please enter exactly two numbers for the pair, separated by a comma (e.g., '18, 19').")
        
        num1, num2 = pair_parts

        white_balls, powerball = generate_with_user_provided_pair(
            num1, num2, GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, 
            excluded_numbers, df, selected_sum_range_tuple
        )
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        return render_template('index.html', 
                               white_balls=white_balls, 
                               powerball=powerball, 
                               last_draw=last_draw.to_dict(), 
                               last_draw_dates=last_draw_dates,
                               generation_type='user_pair',
                               sum_ranges=SUM_RANGES, # Pass sum_ranges back
                               selected_sum_range=selected_sum_range_label) # Pass selected sum range back for persistence
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
    
    # --- New: Get selected sum range ---
    selected_sum_range_label = request.form.get('sum_range_filter_group_a', 'Any') # Unique name for this form's select
    selected_sum_range_tuple = SUM_RANGES.get(selected_sum_range_label)
    # --- End New ---

    white_balls = []
    powerball = None
    last_draw_dates = {}

    try:
        white_balls, powerball = generate_from_group_a(
            df, num_from_group_a, white_ball_range_local, powerball_range_local, 
            excluded_numbers_local, selected_sum_range_tuple
        )
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)
    except ValueError as e:
        flash(str(e), 'error')
        return render_template('index.html', last_draw=last_draw.to_dict(), sum_ranges=SUM_RANGES)

    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw.to_dict(), 
                           last_draw_dates=last_draw_dates,
                           generation_type='group_a_strategy',
                           sum_ranges=SUM_RANGES, # Pass sum_ranges back
                           selected_sum_range=selected_sum_range_label) # Pass selected sum range back for persistence

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
    if df.empty:
        flash("Cannot perform monthly trends analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Get last_draw_date_str correctly from the global df
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'

    # Call the enhanced function that calculates both monthly tops and streaks
    # num_top_wb set to 69 to get all white balls (max possible is 69)
    # num_months_for_top_display set to 6 as requested, will include current partial month.
    monthly_trends_data = get_cached_analysis(
        'monthly_trends_and_streaks', 
        get_monthly_white_ball_analysis_data, 
        df, 
        num_top_wb=69, # Get all white balls (passed for clarity, but logic handles all now)
        num_top_pb=3,  # Top 3 Powerballs
        num_months_for_top_display=6 # Display last 6 months for top numbers
    )
    
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_data=monthly_trends_data['monthly_data'], # Renamed variable here
                           streak_numbers=monthly_trends_data['streak_numbers'])

# New/Modified route for the overall Sum of Main Balls analysis (with chart and full table)
@app.route('/sum_of_main_balls_analysis')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot display Sum of Main Balls Analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Get all sum data and frequencies for the chart and full table
    sums_data_df, sum_freq_list, min_sum, max_sum, avg_sum = get_cached_analysis('sum_of_main_balls_data', sum_of_main_balls, df)
    
    # Convert DataFrame for template rendering for the table
    sums_data = sums_data_df.to_dict('records') 
    
    # Convert sum frequency data to JSON string for Chart.js
    sum_freq_json = json.dumps(sum_freq_list)

    return render_template('sum_of_main_balls.html', # This will be the new template
                           sums_data=sums_data,
                           sum_freq_json=sum_freq_json,
                           min_sum=min_sum,
                           max_sum=max_sum,
                           avg_sum=avg_sum)


# This route remains for the search-by-sum functionality
@app.route('/find_results_by_sum', methods=['GET', 'POST'])
def find_results_by_sum_route():
    if df.empty:
        flash("Cannot display Search by Sum: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    results = []
    target_sum_display = None
    selected_sort_by = request.args.get('sort_by', 'date_desc') # Default sort for GET requests or initial load

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
                    results_df_raw['WhiteBallsTuple'] = results_df_raw.apply( # Changed to results_df_raw here
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
    
    # Pass all necessary data to the template
    return render_template('find_results_by_sum.html', # This is the correct template for search by sum
                           results=results,
                           target_sum=target_sum_display,
                           selected_sort_by=selected_sort_by)

@app.route('/simulate_multiple_draws', methods=['GET', 'POST'])
def simulate_multiple_draws_route():
    if df.empty:
        flash("Cannot run simulation: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    simulated_white_ball_freq_list = []
    simulated_powerball_freq_list = []
    num_draws_display = None
    odd_even_choice_display = "Any" # Default value for display

    if request.method == 'POST':
        num_draws_str = request.form.get('num_draws')
        odd_even_choice_display = request.form.get('odd_even_choice', 'Any') # Get from form

        if num_draws_str and num_draws_str.isdigit():
            num_draws = int(num_draws_str)
            num_draws_display = num_draws
            
            # Pass the odd_even_choice to the simulation function
            sim_results = simulate_multiple_draws(
                df, group_a, odd_even_choice_display, # Pass odd_even_choice here
                GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE, 
                excluded_numbers, num_draws
            )
            
            simulated_white_ball_freq_list = sim_results['white_ball_freq']
            simulated_powerball_freq_list = sim_results['powerball_freq']
        else:
            flash("Please enter a valid number for Number of Simulations.", 'error')

    return render_template('simulate_multiple_draws.html', 
                           simulated_white_ball_freq=simulated_white_ball_freq_list, 
                           simulated_powerball_freq=simulated_powerball_freq_list,   
                           num_simulations=num_draws_display,
                           # Pass back the selected odd_even_choice so the dropdown remains selected
                           selected_odd_even_choice=odd_even_choice_display)


# Commented out as requested
# @app.route('/winning_probability', methods=['GET', 'POST'])
# def winning_probability_route():
#     current_white_ball_range = GLOBAL_WHITE_BALL_RANGE
#     current_powerball_range = GLOBAL_POWERBALL_RANGE
    
#     if request.method == 'POST':
#         try:
#             wb_min = int(request.form.get('white_ball_min', GLOBAL_WHITE_BALL_RANGE[0]))
#             wb_max = int(request.form.get('white_ball_max', GLOBAL_WHITE_BALL_RANGE[1]))
#             pb_min = int(request.form.get('powerball_min', GLOBAL_POWERBALL_RANGE[0]))
#             pb_max = int(request.form.get('powerball_max', GLOBAL_POWERBALL_RANGE[1]))

#             if not (1 <= wb_min <= wb_max <= 69 and 1 <= pb_min <= pb_max <= 26):
#                 flash("Invalid range inputs. White ball range must be 1-69 and Powerball 1-26, and min <= max.", 'error')
#                 wb_min, wb_max = GLOBAL_WHITE_BALL_RANGE
#                 pb_min, pb_max = GLOBAL_POWERBALL_RANGE
            
#             current_white_ball_range = (wb_min, wb_max)
#             current_powerball_range = (pb_min, pb_max)

#         except ValueError:
#             flash("Invalid number format for range inputs. Please enter integers.", 'error')
#             current_white_ball_range = GLOBAL_WHITE_BALL_RANGE
#             current_powerball_range = GLOBAL_POWERBALL_RANGE
    
#     cache_key = f"winning_probability_{current_white_ball_range[0]}-{current_white_ball_range[1]}_{current_powerball_range[0]}-{current_powerball_range[1]}"
    
#     probability_1_in_x, probability_percentage = get_cached_analysis(
#         cache_key, winning_probability, current_white_ball_range, current_powerball_range
#     )

#     return render_template('winning_probability.html', 
#                            probability_1_in_x=probability_1_in_x, 
#                            probability_percentage=probability_percentage,
#                            white_ball_range=current_white_ball_range,
#                            powerball_range=current_powerball_range)

# Commented out as requested
# @app.route('/partial_match_probabilities')
# def partial_match_probabilities_route():
#     probabilities = get_cached_analysis('partial_match_probabilities', partial_match_probabilities, GLOBAL_WHITE_BALL_RANGE, GLOBAL_POWERBALL_RANGE)
#     return render_template('partial_match_probabilities.html', 
#                            probabilities=probabilities)

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
    # Calling with num_years=7 as requested
    yearly_pb_freq_data, years = get_cached_analysis('yearly_pb_freq', get_powerball_frequency_by_year, df, num_years=7)
    return render_template('powerball_frequency_by_year.html',
                           yearly_pb_freq_data=yearly_pb_freq_data,
                           years=years)

@app.route('/odd_even_trends')
def odd_even_trends_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    # Pass last_draw_date_str_for_cache as the second argument, correctly mapped now
    odd_even_trends = get_cached_analysis('odd_even_trends', get_odd_even_split_trends, df, last_draw_date_str_for_cache)
    return render_template('odd_even_trends.html',
                           odd_even_trends=odd_even_trends)

@app.route('/consecutive_trends')
def consecutive_trends_route():
    last_draw_date_str_for_cache = last_draw['Draw Date'] if not last_draw.empty and 'Draw Date' in last_draw else 'N/A'
    consecutive_trends = get_cached_analysis('consecutive_trends', get_consecutive_numbers_trends, df, last_draw_date_str_for_cache)
    return render_template('consecutive_trends.html',
                           consecutive_trends=consecutive_trends)

@app.route('/triplets_analysis')
def triplets_analysis_route():
    triplets_data = get_cached_analysis('triplets_analysis', get_most_frequent_triplets, df) # No top_n passed, will return all
    return render_template('triplets_analysis.html',
                           triplets_data=triplets_data)

@app.route('/grouped_patterns_analysis')
def grouped_patterns_analysis_route():
    patterns_data = get_cached_analysis('grouped_patterns', get_grouped_patterns_over_years, df)
    return render_template('grouped_patterns_analysis.html', patterns_data=patterns_data)

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

@app.route('/generated_numbers_history')
def generated_numbers_history_route():
    generated_history = get_cached_analysis('generated_history', get_generated_numbers_history)
    
    # Get all official draw dates for the new dropdown
    official_draw_dates = []
    if not df.empty:
        # Sort dates in descending order (most recent first)
        official_draw_dates = sorted(df['Draw Date'].unique(), reverse=True)

    # Prepare last_draw to pass to template
    last_draw_for_template = last_draw.to_dict()

    return render_template('generated_numbers_history.html', 
                           generated_history=generated_history,
                           official_draw_dates=official_draw_dates,
                           last_official_draw=last_draw_for_template) # Pass the latest official draw


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
        # Corrected indexing here: random.sample returns a list, need to access elements properly
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
        
        print(f"Simulated new draw data: {new_draw_data}")

        if new_draw_data['Draw Date'] == last_db_draw_date:
            print(f"Draw for {new_draw_data['Draw Date']} already exists. No update needed.")
            return "No new draw data. Database is up-to-date.", 200
        
        url_insert = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        insert_response = requests.post(url_insert, headers=service_headers, data=json.dumps(new_draw_data))
        insert_response.raise_for_status()

        if insert_response.status_code == 201:
            print(f"Successfully inserted new draw: {new_draw_data}")
            
            initialize_core_data() 
            invalidate_analysis_cache()

            return f"Data updated successfully with draw for {simulated_draw_date}.", 200
        else:
            print(f"Failed to insert data. Status: {insert_response.status_code}, Response: {insert_response.text}")
            return f"Error updating data: {insert_response.status_code} - {insert_response.text}", 500

    except requests.exceptions.RequestException as e:
        print(f"Network or HTTP error during update_powerball_data: {e}")
        traceback.print_exc() # Print full traceback
        if hasattr(e, 'response') and e.response is not None:
            print(f"Supabase response content: {e.response.text}")
        return f"Network or HTTP error: {e}", 500
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in update_powerball_data: {e}")
        traceback.print_exc() # Print full traceback
        if 'insert_response' in locals() and insert_response is not None:
            print(f"Response content that failed JSON decode: {insert_response.text}")
        return f"JSON parsing error: {e}", 500
    except Exception as e:
        print(f"An unexpected error occurred during data update: {e}")
        traceback.print_exc() # Print full traceback
        return f"An internal error occurred: {e}", 500


@app.route('/export_analysis_results')
def export_analysis_results_route():
    if df.empty:
        flash("Cannot export results: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    export_analysis_results(df) 
    flash("Analysis results exported to analysis_results.csv (this file is temporary on Vercel's serverless environment).", 'info')
    return redirect(url_for('index'))


# Routes for backend API calls only (these do not render templates directly)
@app.route('/analyze_batch_vs_official', methods=['POST'])
def analyze_batch_vs_official_route():
    try:
        data = request.get_json()
        generated_date_str = data.get('generated_date')
        official_draw_date_str = data.get('official_draw_date')

        if not generated_date_str or not official_draw_date_str:
            return jsonify({"error": "Missing generated_date or official_draw_date"}), 400

        # Fetch the specific batch of generated numbers
        generated_picks = _get_generated_picks_for_date_from_db(generated_date_str)
        if not generated_picks:
            return jsonify({"error": f"No generated picks found for date: {generated_date_str}"}), 404

        # Fetch the specific official draw
        official_draw = _get_official_draw_for_date_from_db(official_draw_date_str)
        if not official_draw:
            return jsonify({"error": f"No official draw found for date: {official_draw_date_str}. Please ensure it is added to the database."}), 404

        # Perform the analysis
        analysis_summary = analyze_generated_batch_against_official_draw(generated_picks, official_draw)
        
        return jsonify({
            "success": True,
            "generated_date": generated_date_str,
            "official_draw_date": official_draw_date_str,
            "total_generated_picks_in_batch": len(generated_picks),
            "summary": analysis_summary
        })

    except Exception as e:
        print(f"Error during analyze_batch_vs_official_route: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/analyze_generated_historical_matches', methods=['POST'])
def analyze_generated_historical_matches_route():
    if df.empty:
        return jsonify({"success": False, "error": "Historical data not loaded or is empty."}), 500
    
    try:
        data = request.get_json() # Expecting JSON now
        generated_white_balls_str = data.get('generated_white_balls')
        generated_powerball_str = data.get('generated_powerball')

        if not generated_white_balls_str or not generated_powerball_str: 
            return jsonify({"success": False, "error": "Missing generated_white_balls or generated_powerball"}), 400

        generated_white_balls = sorted([int(x.strip()) for x in generated_white_balls_str.split(',') if x.strip().isdigit()])
        generated_powerball = int(generated_powerball_str)

        if len(generated_white_balls) != 5:
            return jsonify({"success": False, "error": "Invalid generated white balls format. Expected 5 numbers."}), 400

        # Pass the results dictionary directly to the function
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
        print(f"An error occurred during historical analysis: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

# --- NEW AI ASSISTANT ROUTES ---
@app.route('/ai_assistant')
def ai_assistant_route():
    return render_template('ai_assistant.html')

@app.route('/chat_with_ai', methods=['POST'])
def chat_with_ai_route():
    user_message = request.json.get('message')
    print(f"[AI-DEBUG] Received user message: '{user_message}'")

    if not user_message:
        print("[AI-DEBUG] No user message provided.")
        return jsonify({"response": "Please provide a message."}), 400

    chat_history = []
    
    initial_prompt_text = """
    You are a highly knowledgeable Powerball Lottery Analysis AI Assistant. 
    Your purpose is to help users understand Powerball data, trends, and probabilities.
    You should be helpful, informative, and concise.
    
    Here's some general knowledge about Powerball:
    - White Balls: 5 numbers are drawn from a pool of 1 to 69.
    - Powerball: 1 number is drawn from a separate pool of 1 to 26.
    - Odds of winning the jackpot (matching 5 White Balls + Powerball): 1 in 292,201,338.
    - Common analysis includes: number frequency, hot/cold numbers, sum of balls, odd/even splits, consecutive numbers, grouped patterns, and Powerball position frequency.

    When asked about specific data analysis (e.g., "hot numbers," "sum of balls," "monthly trends"), you should explain *what* that analysis means and *why* it might be relevant, without actually providing specific real-time data unless that functionality is explicitly added later through tool use. For now, focus on explanations and general insights.

    If a user asks to "generate numbers," or requests specific real-time data that you don't have access to (like "what are the current hot numbers?"), you should gently explain that you are an analytical assistant and cannot perform live data lookups or generate numbers, but can explain the *concepts* behind them.

    User's question: """ + user_message

    chat_history.append({
        "role": "user",
        "parts": [{"text": initial_prompt_text}]
    })
    
    # Debugging: Print the full chat history being sent
    print(f"[AI-DEBUG] Chat history payload to Gemini: {json.dumps(chat_history, indent=2)}")

    try:
        apiKey = "" # Leave this empty, Canvas runtime will inject the key.
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
        
        # Debugging: Print the full payload before sending
        print(f"[AI-DEBUG] Full API payload: {json.dumps(payload, indent=2)}")

        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        
        # Debugging: Print raw response info
        print(f"[AI-DEBUG] Gemini API Response Status Code: {response.status_code}")
        print(f"[AI-DEBUG] Gemini API Raw Response Text: {response.text}")

        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        result = response.json()
        # Debugging: Print parsed JSON result
        print(f"[AI-DEBUG] Gemini API Parsed JSON Result: {json.dumps(result, indent=2)}")

        if result and result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            ai_response = result['candidates'][0]['content']['parts'][0]['text']
            print(f"[AI-DEBUG] Successfully extracted AI response.")
            return jsonify({"response": ai_response})
        else:
            error_message = "I'm sorry, I couldn't generate a response. The AI model did not return valid content."
            print(f"[AI-DEBUG] Failed to extract AI response: {error_message}")
            return jsonify({"response": error_message}), 500

    except requests.exceptions.RequestException as e:
        print(f"[AI-ERROR] Error calling Gemini API: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({"response": f"Error communicating with AI: {e}"}), 500
    except json.JSONDecodeError as e:
        print(f"[AI-ERROR] JSON decoding error from Gemini API: {e}. Raw response was: {response.text}")
        traceback.print_exc() # Print full traceback
        return jsonify({"response": f"Error parsing AI response: {e}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred in chat_with_ai_route: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({"response": f"An internal error occurred: {e}"}), 500

# --- NEW MANUAL PICK ROUTES ---
@app.route('/my_jackpot_pick')
def my_jackpot_pick_route():
    try:
        return render_template('my_jackpot_pick.html')
    except Exception as e:
        print(f"Error rendering my_jackpot_pick.html: {e}")
        traceback.print_exc() # Print full traceback
        flash("An error occurred loading the Jackpot Pick page. Please try again.", 'error')
        return redirect(url_for('index')) # Redirect to home or show a fallback

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
        
        # Ensure numbers are integers and sorted for consistency
        white_balls = sorted([int(n) for n in white_balls])
        powerball = int(powerball)

        # Perform historical match analysis
        historical_match_results = check_generated_against_history(white_balls, powerball, df)
        
        # Get last drawn dates for each selected number
        last_drawn_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        return jsonify({
            "success": True,
            "generated_numbers": white_balls, # Renamed key for clarity for manual pick
            "generated_powerball": powerball,
            "match_summary": historical_match_results['summary'],
            "last_drawn_dates": last_drawn_dates
        })

    except ValueError:
        return jsonify({"error": "Invalid number format provided."}), 400
    except Exception as e:
        print(f"Error during manual pick analysis: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/save_manual_pick', methods=['POST']) # NEW ROUTE FOR SAVING MANUAL PICKS
def save_manual_pick_route():
    try:
        data = request.get_json()
        white_balls = data.get('white_balls')
        powerball = data.get('powerball')

        if not white_balls or len(white_balls) != 5 or powerball is None:
            return jsonify({"success": False, "error": "Invalid input. Please provide 5 white balls and 1 powerball."}), 400
        
        # Ensure numbers are integers and sorted for consistency
        white_balls = sorted([int(n) for n in white_balls])
        powerball = int(powerball)

        # Use the existing save_generated_numbers_to_db function
        success, message = save_generated_numbers_to_db(white_balls, powerball)

        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"success": False, "error": message}), 400

    except ValueError:
        return jsonify({"success": False, "error": "Invalid number format provided."}), 400
    except Exception as e:
        print(f"Error during save_manual_pick_route: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

# NEW ROUTE: Sum Trends and Gaps analysis
@app.route('/sum_trends_and_gaps')
def sum_trends_and_gaps_route():
    if df.empty:
        flash("Cannot display Sum Trends and Gaps: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    sum_data = get_cached_analysis('sum_trends_and_gaps', get_sum_trends_and_gaps_data, df)
    
    # Render the new template with the comprehensive sum data
    return render_template('sum_trends_and_gaps.html', 
                           min_possible_sum=sum_data['min_possible_sum'],
                           max_possible_sum=sum_data['max_possible_sum'],
                           appeared_sums_details=sum_data['appeared_sums_details'],
                           missing_sums=sum_data['missing_sums'],
                           grouped_sums_analysis=sum_data['grouped_sums_analysis'])

# NEW/MODIFIED ROUTE: Weekday Trends (Low/High) analysis
@app.route('/weekday_trends')
def weekday_trends_route():
    if df.empty:
        flash("Cannot display Weekday Trends: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
    
    # Pass df and group_a (if needed by get_weekday_draw_trends)
    # The get_weekday_draw_trends function already accepts group_a_numbers_def=None,
    # and it uses the global `group_a` if that parameter is not explicitly passed,
    # or if it's passed as None. So we can just pass `df`.
    weekday_data = get_cached_analysis('weekday_all_trends', get_weekday_draw_trends, df, group_a_numbers_def=group_a)
    
    return render_template('weekday_trends.html', 
                           weekday_trends=weekday_data)
    # NEW ROUTE: Strict Positional Search
@app.route('/strict_positional_search', methods=['GET', 'POST'])
def strict_positional_search_route():
    entered_numbers = {
        'white_ball_1': '', 'white_ball_2': '', 'white_ball_3': '', 
        'white_ball_4': '', 'white_ball_5': '', 'powerball_pos': ''
    }
    search_results = []
    total_results = 0
    
    if request.method == 'POST':
        # Retrieve form data
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

        # Build query parameters for Supabase
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

        # Perform the search using the utility function
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
