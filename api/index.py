import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
import random
from itertools import combinations
import math
import os
from collections import defaultdict
from datetime import datetime
import requests # Import requests for direct API calls
import json # For handling JSON data

# --- Flask App Initialization with Template Path ---
# Get the directory of the current script (api/index.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the templates directory, assuming it's one level up from 'api'
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')

# Initialize Flask app, specifying the template folder
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = 'supersecretkey'

# --- Supabase Configuration (IMPORTANT: Use Environment Variables for Production) ---
# For Vercel, set these as Environment Variables: SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY
# Using provided credentials for demonstration, but recommend env vars for actual deployment.
SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NzMwNjUsImV4cCI6MjA2NTM0OTA2NX0.AzUD7wjR7VbvtUH27NDqJ3AlvFW0nCWpiN9ADG8T_t4")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "YOUR_SUPABASE_SERVICE_ROLE_KEY") # <<< Set this in Vercel env vars for write access!

SUPABASE_TABLE_NAME = 'powerball_draws' # Your table name in Supabase

# --- Utility Functions (Modified for direct API calls) ---

def _get_supabase_headers(is_service_key=False):
    """Helper to get common Supabase API headers."""
    key = SUPABASE_SERVICE_KEY if is_service_key else SUPABASE_ANON_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

def load_historical_data_from_supabase():
    """
    Fetches all historical data from Supabase using requests with pagination
    and returns as a pandas DataFrame.
    """
    all_data = []
    offset = 0
    limit = 1000 # Supabase default limit, fetch in chunks

    try:
        url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        headers = _get_supabase_headers(is_service_key=False)
        
        while True:
            params = {
                'select': '*',
                'order': 'Draw Date.asc', # Use the exact column name 'Draw Date' for ordering
                'offset': offset,
                'limit': limit
            }

            print(f"Fetching from Supabase URL: {url} with params: {params}")

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            chunk = response.json()
            if not chunk:
                break # No more data to fetch

            all_data.extend(chunk)
            offset += limit
            print(f"Fetched {len(chunk)} records. Total so far: {len(all_data)}")

        if not all_data:
            print("No data fetched from Supabase after pagination attempts.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        
        # Convert 'Draw Date' (which is now directly accessed from Supabase response) to datetime objects
        # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
        df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'], errors='coerce')
        # Drop rows where 'Draw Date' could not be parsed to a valid date
        df = df.dropna(subset=['Draw Date_dt'])

        # Ensure numeric columns are integers. Handle potential errors by coercing to numeric, filling NaNs (e.g., from 'werb'), and then converting to int.
        numeric_cols = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Convert to numeric, non-numeric become NaN
                df[col] = df[col].fillna(0).astype(int) # Fill NaNs (e.g., 'werb' converted to NaN) with 0, then convert to int
            else:
                print(f"Warning: Column '{col}' not found in fetched data. Skipping conversion for this column.")

        # Reformat 'Draw Date' for display
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
        # Fixed: Explicitly specify dtype to silence FutureWarning
        return pd.Series({
            'Draw Date': 'N/A', 'Number 1': 'N/A', 'Number 2': 'N/A',
            'Number 3': 'N/A', 'Number 4': 'N/A', 'Number 5': 'N/A', 'Powerball': 'N/A'
        }, dtype='object')
    return df.iloc[-1]

def check_exact_match(df, white_balls):
    if df.empty: return False
    for _, row in df.iterrows():
        # Ensure numbers are treated as integers for comparison
        historical_white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
        if set(white_balls) == set(historical_white_balls):
            return True
    return False

def generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None):
    if df.empty:
        raise ValueError("Cannot generate numbers: Historical data is empty.")

    max_attempts = 1000 # Limit attempts to prevent infinite loops
    attempts = 0
    while attempts < max_attempts:
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        
        if len(available_numbers) < 5:
            raise ValueError("Not enough available numbers for white balls after exclusions and range constraints.")
            
        white_balls = sorted(random.sample(available_numbers, 5)) # Sort for consistency

        # Group A constraint
        group_a_numbers = [num for num in white_balls if num in group_a]
        # This original logic was "less than 2", meaning 0 or 1. If it needs to be *at least* 2, then >= 2.
        # Assuming the intent is "at least 2 numbers from Group A must be present"
        if len(group_a_numbers) < 2:
            attempts += 1
            continue

        powerball = random.randint(powerball_range[0], powerball_range[1])

        last_draw_data = df.iloc[-1]
        # Ensure 'Number 1' is not 'N/A' from initial empty series
        if not last_draw_data.empty and last_draw_data.get('Number 1') != 'N/A':
            # Convert to list of ints before set comparison
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
            # Assuming low <= 34, high >= 35 based on common Powerball splits
            low_numbers_count = sum(1 for num in white_balls if num <= 34)
            high_numbers_count = sum(1 for num in white_balls if num >= 35)
            if low_numbers_count != high_low_balance[0] or high_numbers_count != high_low_balance[1]:
                attempts += 1
                continue
        
        # If all conditions met, break loop
        break
    else: # This else block executes if the while loop completes without a 'break' (i.e., max_attempts reached)
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

    # Ensure consistent return types if no hot/cold numbers (re-checked and corrected if they were empty initially)
    if hot_numbers.empty:
        hot_numbers = pd.Series([], dtype=int)
    if cold_numbers.empty:
        cold_numbers = pd.Series([], dtype=int)

    return hot_numbers, cold_numbers

def monthly_white_ball_analysis(df, last_draw_date_str):
    if df.empty or last_draw_date_str == 'N/A': return {}
    last_draw_date = pd.to_datetime(last_draw_date_str)
    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    
    recent_data = df[df['Draw Date_dt'] >= six_months_ago].copy()
    if recent_data.empty: return {}

    recent_data['Month'] = recent_data['Draw Date_dt'].dt.to_period('M')
    
    monthly_balls = recent_data.groupby('Month')[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].apply(
        lambda x: sorted(list(set(x.values.flatten().astype(int)))) # Ensure integers and sorted
    ).to_dict()
    
    monthly_balls_str_keys = {str(k): v for k, v in monthly_balls.items()}

    return monthly_balls_str_keys

def sum_of_main_balls(df):
    if df.empty: return pd.DataFrame(), [], 0, 0, 0.0
    temp_df = df.copy()
    # Convert relevant columns to integers if they aren't already
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
    # Ensure relevant columns are integers before calculating sum
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
            # Pass all necessary arguments to generate_powerball_numbers
            white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers) # Fixed: using "Any" and "No Combo" defaults
            results.append(white_balls + [powerball])
        except ValueError:
            pass # Skip combinations that don't meet criteria
    
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

    total_winning_white_comb = calculate_combinations(total_white_balls_in_range, 5) # C(69, 5)

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
        # Combinations for matching white balls
        comb_matched_w = calculate_combinations(5, data["matched_w"]) # Choose 'matched_w' from the 5 winning balls
        
        # Combinations for unmatched white balls
        # (total_white_balls_in_range - 5) is the pool of non-winning white balls
        comb_unmatched_w = calculate_combinations(total_white_balls_in_range - 5, data["unmatched_w"])

        # Combinations for Powerball
        if data["matched_p"] == 1:
            comb_p = calculate_combinations(1, 1) # Choose the single winning Powerball
        else: # Powerball does not match
            comb_p = calculate_combinations(total_powerballs_in_range - 1, 1) # Choose 1 from (total - 1) non-winning Powerballs
            if total_powerballs_in_range == 1: # Edge case: if only 1 powerball possible, can't not match it
                comb_p = 0
        
        numerator = comb_matched_w * comb_unmatched_w * comb_p
        
        if numerator == 0:
            probabilities[scenario] = "N/A"
        else:
            # Total ways to choose 5 white balls and 1 powerball
            total_possible_combinations = total_winning_white_comb * total_powerballs_in_range
            
            # The odds are Total Possible Combinations / Favorable Combinations
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
            # Convert numbers in row to list of integers for proper comparison
            historical_white_balls = [int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])]
            if number in historical_white_balls:
                last_draw_dates[f"White Ball {number}"] = row['Draw Date']
                found = True
                break
        if not found:
            last_draw_dates[f"White Ball {number}"] = "N/A (Never Drawn)" # Add if not found

    found_pb = False
    for _, row in sorted_df.iterrows():
        if powerball == int(row['Powerball']):
            last_draw_dates[f"Powerball {powerball}"] = row['Draw Date']
            found_pb = True
            break
    if not found_pb:
        last_draw_dates[f"Powerball {powerball}"] = "N/A (Never Drawn)" # Add if not found

    return last_draw_dates

def modify_combination(df, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers):
    if df.empty:
        raise ValueError("Cannot modify combination: Historical data is empty.")

    white_balls = list(white_balls)
    
    # Ensure there are enough unique elements to sample from before modifying
    if len(white_balls) < 5: # Should always be 5, but as a safeguard
        raise ValueError("Initial white balls list is too short for modification.")

    indices_to_modify = random.sample(range(5), 3) # Select 3 random indices to modify
    
    for i in indices_to_modify:
        attempts = 0
        max_attempts_single_num = 100 # Prevent infinite loop for one number
        while attempts < max_attempts_single_num:
            new_number = random.randint(white_ball_range[0], white_ball_range[1])
            # Check if new number is not excluded and not already in current white_balls list (before modification)
            if new_number not in excluded_numbers and new_number not in white_balls:
                white_balls[i] = new_number
                break # Found a valid number for this position
            attempts += 1
        else:
            # Fallback if a number can't be found after many attempts for a specific position
            print(f"Warning: Could not find unique replacement for white ball at index {i}. Proceeding without replacement for this slot.")

    attempts_pb = 0
    max_attempts_pb = 100
    while attempts_pb < max_attempts_pb:
        new_powerball = random.randint(powerball_range[0], powerball_range[1])
        # Ensure new powerball is not excluded and is different from the original
        if new_powerball not in excluded_numbers and new_powerball != powerball:
            powerball = new_powerball
            break
        attempts_pb += 1
    else:
        print("Warning: Could not find a unique replacement for powerball. Keeping original.")

    white_balls = sorted([int(num) for num in white_balls]) # Ensure integers and sorted
    powerball = int(powerball)
    
    return white_balls, powerball

def find_common_pairs(df, top_n=10):
    if df.empty: return []
    pair_count = defaultdict(int)
    for _, row in df.iterrows():
        # Ensure numbers are converted to integers before sorting/pairing
        nums = sorted([int(row['Number 1']), int(row['Number 2']), int(row['Number 3']), int(row['Number 4']), int(row['Number 5'])])
        # Generate all unique pairs from the 5 white balls
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
        # Fallback to general generation if no common pairs or filters exclude all
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        if len(available_numbers) < 5:
             raise ValueError("Not enough numbers to generate 5 white balls after exclusions.")
        return sorted(random.sample(available_numbers, 5)) # Return sorted

    num1, num2 = random.choice(common_pairs) # Pick a common pair
    
    available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) 
                         if num not in excluded_numbers and num not in [num1, num2]]
    
    if len(available_numbers) < 3: # Need 3 more unique numbers
        # If not enough unique numbers for the remaining 3, try to generate 5 completely new ones
        # from the general available pool. This is a robust fallback.
        available_numbers_fallback = [n for n in range(white_ball_range[0], white_ball_range[1] + 1) if n not in excluded_numbers]
        if len(available_numbers_fallback) < 5:
            raise ValueError("Not enough numbers to generate 5 white balls even with fallback after exclusions.")
        return sorted(random.sample(available_numbers_fallback, 5))

    remaining_numbers = random.sample(available_numbers, 3)
    white_balls = sorted([num1, num2] + remaining_numbers) # Ensure all 5 are sorted
    return white_balls

def get_number_age_distribution(df):
    if df.empty: return []
    # Ensure Draw Date is datetime for sorting and comparisons
    df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'])
    all_draw_dates = sorted(df['Draw Date_dt'].drop_duplicates().tolist())
    
    all_miss_streaks = []

    for i in range(1, 70): # White balls range
        last_appearance_date = None
        # Efficiently check if 'i' is in any of the white ball columns for each row
        # Convert columns to integers first to ensure proper comparison if mixed types
        temp_df_filtered = df[(df['Number 1'].astype(int) == i) | (df['Number 2'].astype(int) == i) |
                              (df['Number 3'].astype(int) == i) | (df['Number 4'].astype(int) == i) |
                              (df['Number 5'].astype(int) == i)]
        
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()

        miss_streak_count = 0
        if last_appearance_date is not None:
            # Count draws since last appearance
            for d_date in reversed(all_draw_dates):
                if d_date > last_appearance_date:
                    miss_streak_count += 1
                else: # d_date <= last_appearance_date, so we've reached or passed it
                    break
            all_miss_streaks.append(miss_streak_count)
        else:
            all_miss_streaks.append(len(all_draw_dates)) # If never appeared, age is total draws

    for i in range(1, 27): # Powerballs range
        last_appearance_date = None
        temp_df_filtered = df[df['Powerball'].astype(int) == i] # Convert to int
        if not temp_df_filtered.empty:
            last_appearance_date = temp_df_filtered['Draw Date_dt'].max()

        miss_streak_count = 0
        if last_appearance_date is not None:
            for d_date in reversed(all_draw_dates):
                if d_date > last_appearance_date:
                    miss_streak_count += 1
                else:
                    break
            all_miss_streaks.append(miss_streak_count)
        else:
            all_miss_streaks.append(len(all_draw_dates))

    age_counts = pd.Series(all_miss_streaks).value_counts().sort_index()
    number_age_data = [{'age': int(age), 'count': int(count)} for age, count in age_counts.items()]
    
    return number_age_data

def get_co_occurrence_matrix(df):
    if df.empty: return [], 0
    co_occurrence = defaultdict(int)
    
    for index, row in df.iterrows():
        # Corrected: Accessing numbers correctly and converting to int
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
        powerball = int(row['Powerball']) # Ensure Powerball is an integer
        # Iterate through white ball "positions" (columns)
        for i in range(1, 6): # Positions 1 to 5
            col_name = f'Number {i}'
            if col_name in row and pd.notna(row[col_name]):
                position_frequency_data.append({
                    'powerball_number': powerball,
                    # This captures the white ball's *actual value* at that position,
                    # which is what the D3 chart uses for a complex rollup.
                    # If the intention was just to count how many times ANY number appeared at a position
                    # when a certain Powerball was drawn, the current structure is fine.
                    'white_ball_value_at_position': int(row[col_name]), # Actual white ball number at this column
                    'white_ball_position': i # The column number (1-5)
                })
    return position_frequency_data

# --- Global Data Loading and Pre-computation ---
# This block runs once when the Vercel serverless function container is initialized (cold start).
# It's crucial that this part initializes correctly, as other routes depend on 'df'.
df = pd.DataFrame()
# Fixed: Explicitly specify dtype for global last_draw initialization
last_draw = pd.Series(dtype='object') 

precomputed_white_ball_freq = pd.Series([], dtype=int)
precomputed_powerball_freq = pd.Series([], dtype=int)
precomputed_last_draw_date_str = "N/A"
precomputed_hot_numbers = pd.Series([], dtype=int)
precomputed_cold_numbers = pd.Series([], dtype=int)
precomputed_monthly_balls = {}
precomputed_number_age_data = []
precomputed_co_occurrence_data = []
precomputed_max_co_occurrence = 0
precomputed_powerball_position_data = []

try:
    df = load_historical_data_from_supabase() # Load from Supabase
    last_draw = get_last_draw(df)

    if not df.empty: # Only pre-compute if data was successfully loaded
        precomputed_white_ball_freq, precomputed_powerball_freq = frequency_analysis(df)
        precomputed_last_draw_date_str = last_draw['Draw Date']
        precomputed_hot_numbers, precomputed_cold_numbers = hot_cold_numbers(df, precomputed_last_draw_date_str)
        precomputed_monthly_balls = monthly_white_ball_analysis(df, precomputed_last_draw_date_str)
        
        precomputed_number_age_data = get_number_age_distribution(df)
        precomputed_co_occurrence_data, precomputed_max_co_occurrence = get_co_occurrence_matrix(df)
        precomputed_powerball_position_data = get_powerball_position_frequency(df)

except Exception as e:
    print(f"An error occurred during initial data loading or pre-computation: {e}")
    # Flash messages for frontend will handle UI feedback based on empty df


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
            # Ensure the date is formatted correctly for display, even if it's already a string
            last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
        except ValueError:
            pass # Keep as is if conversion fails
    return render_template('index.html', last_draw=last_draw_dict)

@app.route('/generate', methods=['POST'])
def generate():
    if df.empty:
        flash("Cannot generate numbers: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    odd_even_choice = request.form.get('odd_even_choice', 'Any')
    combo_choice = request.form.get('combo_choice', 'No Combo') # This parameter is not currently used in generate_powerball_numbers, consider removing or implementing.
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
        # Re-render the index page without generated numbers
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
        return redirect(url_for('index'))

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
        # Initial draw to potentially modify, or for fallback
        if not df.empty:
            random_row = df.sample(1).iloc[0]
            white_balls_base = [int(random_row['Number 1']), int(random_row['Number 2']), int(random_row['Number 3']), int(random_row['Number 4']), int(random_row['Number 5'])]
            powerball_base = int(random_row['Powerball'])
        else:
            # Fallback for empty df, though it should be caught earlier
            flash("Historical data is empty, cannot generate or modify numbers.", 'error')
            return redirect(url_for('index'))


        if use_common_pairs:
            common_pairs = find_common_pairs(df, top_n=20)
            if num_range:
                common_pairs = filter_common_pairs_by_range(common_pairs, num_range)
            
            if not common_pairs:
                flash("No common pairs found with the specified filter. Generating a random combination instead.", 'info')
                white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
            else:
                white_balls = generate_with_common_pairs(df, common_pairs, white_ball_range, excluded_numbers)
                powerball = random.randint(powerball_range[0], powerball_range[1]) # Powerball is still random
        else:
            white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, white_ball_range, powerball_range, excluded_numbers)
            
        max_attempts_unique = 100
        attempts_unique = 0
        # Ensure the generated combination is not an exact historical match
        while check_exact_match(df, white_balls) and attempts_unique < max_attempts_unique:
            if use_common_pairs:
                common_pairs_recheck = find_common_pairs(df, top_n=20)
                if num_range:
                    common_pairs_recheck = filter_common_pairs_by_range(common_pairs_recheck, num_range)
                if common_pairs_recheck:
                    white_balls = generate_with_common_pairs(df, common_pairs_recheck, white_ball_range, excluded_numbers)
                else: # If common pairs failed, resort to general generation
                    white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
            else: # For non-common pair modification
                random_row = df.sample(1).iloc[0]
                white_balls_base = [int(random_row['Number 1']), int(random_row['Number 2']), int(random_row['Number 3']), int(random_row['Number 4']), int(random_row['Number 5'])]
                powerball_base = int(random_row['Powerball'])
                white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, white_ball_range, powerball_range, excluded_numbers)
            attempts_unique += 1
        
        if attempts_unique == max_attempts_unique:
            flash("Could not find a unique modified combination after many attempts. Please try again.", 'error')
            return redirect(url_for('index'))

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
        # Re-render the index page without generated numbers
        last_draw_dict = last_draw.to_dict()
        if last_draw_dict.get('Draw Date') and last_draw_dict['Draw Date'] != 'N/A':
            try:
                last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
            except ValueError:
                pass
        return render_template('index.html', last_draw=last_draw_dict)


@app.route('/frequency_analysis')
def frequency_analysis_route():
    # Use pre-computed data
    white_ball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in precomputed_white_ball_freq.items()]
    powerball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in precomputed_powerball_freq.items()]
    
    return render_template('frequency_analysis.html', 
                           white_ball_freq=white_ball_freq_list, 
                           powerball_freq=powerball_freq_list)

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    # Use pre-computed data
    hot_numbers_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in precomputed_hot_numbers.items()]
    cold_numbers_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in precomputed_cold_numbers.items()]
    
    return render_template('hot_cold_numbers.html', 
                           hot_numbers=hot_numbers_list, 
                           cold_numbers=cold_numbers_list)

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    # Use pre-computed data
    monthly_balls = precomputed_monthly_balls
    
    return render_template('monthly_white_ball_analysis.html', 
                           monthly_balls=monthly_balls)

@app.route('/sum_of_main_balls')
def sum_of_main_balls_route():
    if df.empty:
        flash("Cannot perform analysis: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))
        
    sum_data_df, sum_freq_list, min_sum, max_sum, avg_sum = sum_of_main_balls(df)

    return render_template('sum_of_main_balls.html', 
                           sum_data=sum_data_df.to_dict('records'), 
                           sum_freq_list=sum_freq_list, # Pass frequency data for D3 chart
                           min_sum=min_sum, 
                           max_sum=max_sum, 
                           avg_sum=avg_sum)

@app.route('/find_results_by_sum', methods=['GET', 'POST']) # Allow GET for direct link, POST for form submission
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
    # If GET request or POST with invalid input, render the page with an empty results list
    return render_template('find_results_by_sum.html', 
                           results=results,
                           target_sum=target_sum_display)

@app.route('/simulate_multiple_draws', methods=['GET', 'POST']) # Allow GET for direct link, POST for form submission
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
            # Pass all required args for generate_powerball_numbers, even if not directly from form here
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
    number_age_data = precomputed_number_age_data

    return render_template('number_age_distribution.html',
                           number_age_data=number_age_data)

@app.route('/co_occurrence_analysis')
def co_occurrence_analysis_route():
    co_occurrence_data = precomputed_co_occurrence_data
    max_co_occurrence = precomputed_max_co_occurrence

    return render_template('co_occurrence_analysis.html',
                           co_occurrence_data=co_occurrence_data,
                           max_co_occurrence=max_co_occurrence)

@app.route('/powerball_position_frequency')
def powerball_position_frequency_route():
    powerball_position_data = precomputed_powerball_position_data

    return render_template('powerball_position_frequency.html',
                           powerball_position_data=powerball_position_data)

@app.route('/find_results_by_first_white_ball', methods=['GET', 'POST']) # Allow GET for direct link, POST for form submission
def find_results_by_first_white_ball():
    if df.empty:
        flash("Cannot find results: Historical data not loaded or is empty. Please check Supabase connection.", 'error')
        return redirect(url_for('index'))

    results_dict = []
    white_ball_number_display = None
    sort_by_year_flag = False

    if request.method == 'POST':
        white_ball_number_str = request.form.get('white_ball_number')
        sort_by_year_flag = request.form.get('sort_by_year') == 'on'

        if white_ball_number_str and white_ball_number_str.isdigit():
            white_ball_number = int(white_ball_number_str)
            white_ball_number_display = white_ball_number
            
            # Ensure 'Number 1' column is treated as int for comparison if mixed types
            # Added .copy() to avoid SettingWithCopyWarning
            results = df[df['Number 1'].astype(int) == white_ball_number].copy()

            if sort_by_year_flag:
                # Ensure 'Draw Date' is datetime before extracting year
                results['Year'] = pd.to_datetime(results['Draw Date'], errors='coerce').dt.year
                results = results.sort_values(by='Year')
            
            results_dict = results.to_dict('records')
        else:
            flash("Please enter a valid number for First White Ball Number.", 'error')

    return render_template('find_results_by_first_white_ball.html', 
                           results_by_first_white_ball=results_dict, 
                           white_ball_number=white_ball_number_display,
                           sort_by_year=sort_by_year_flag)

# --- NEW: Route for updating data via scheduled job ---
@app.route('/update_powerball_data', methods=['GET']) # Using GET for simplicity, POST is generally better for mutations
def update_powerball_data():
    # It's crucial to use the Service Role Key for inserts if RLS is enabled and restrictive.
    # Otherwise, the Anon Key (public) might be sufficient if RLS allows anon inserts.
    # For robust production, ensure SUPABASE_SERVICE_KEY is set as a Vercel environment variable.
    service_headers = _get_supabase_headers(is_service_key=True)
    anon_headers = _get_supabase_headers(is_service_key=False)

    try:
        # 1. Fetch latest draw date from Supabase to check if we need to update
        url_check_latest = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        params_check_latest = {
            'select': 'Draw Date', # Use exact column name
            'order': 'Draw Date.desc', # Use exact column name
            'limit': 1
        }
        response_check_latest = requests.get(url_check_latest, headers=anon_headers, params=params_check_latest)
        response_check_latest.raise_for_status()
        
        latest_db_draw_data = response_check_latest.json()
        last_db_draw_date = None
        if latest_db_draw_data:
            last_db_draw_date = latest_db_draw_data[0]['Draw Date'] # Access by exact column name
        
        print(f"Last draw date in Supabase: {last_db_draw_date}")

        # 2. Fetch latest actual Powerball draw from an external source
        # This is the PRIMARY PLACEHOLDER you NEED to replace with actual data fetching logic.
        # This simulation will insert a new draw every time it runs (unless the date matches).
        simulated_draw_date_dt = datetime.now() # Use current datetime for distinctness
        simulated_draw_date = simulated_draw_date_dt.strftime('%Y-%m-%d')
        simulated_numbers = sorted(random.sample(range(1, 70), 5))
        simulated_powerball = random.randint(1, 26)

        new_draw_data = {
            'Draw Date': simulated_draw_date, # Use exact column name
            'Number 1': simulated_numbers[0], # Use exact column name
            'Number 2': simulated_numbers[1],
            'Number 3': simulated_numbers[2],
            'Number 4': simulated_numbers[3],
            'Number 5': simulated_numbers[4],
            'Powerball': simulated_powerball # Use exact column name
        }
        
        print(f"Simulated new draw data: {new_draw_data}")

        # Compare with the latest in DB
        if new_draw_data['Draw Date'] == last_db_draw_date:
            print(f"Draw for {new_draw_data['Draw Date']} already exists. No update needed.")
            return "No new draw data. Database is up-to-date.", 200
        
        # 3. Insert the new draw into Supabase
        url_insert = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}"
        insert_response = requests.post(url_insert, headers=service_headers, data=json.dumps(new_draw_data))
        insert_response.raise_for_status() # Raise error for bad responses

        if insert_response.status_code == 201: # 201 Created is typical for successful POST
            print(f"Successfully inserted new draw: {new_draw_data}")
            # IMPORTANT: After updating the DB, the global 'df' and precomputed data
            # will only be updated on the *next cold start* of the Vercel function.
            # For immediate reflection, you would need to trigger a new deployment or
            # explicitly reload these global variables, which might be slow on every request.
            # For cron jobs, this is usually acceptable, as the next user request will get updated data.
            
            # Re-load and re-compute data after insertion to reflect immediately for next request
            global df, last_draw, precomputed_white_ball_freq, precomputed_powerball_freq, \
                   precomputed_last_draw_date_str, precomputed_hot_numbers, precomputed_cold_numbers, \
                   precomputed_monthly_balls, precomputed_number_age_data, precomputed_co_occurrence_data, \
                   precomputed_max_co_occurrence, precomputed_powerball_position_data

            df = load_historical_data_from_supabase()
            last_draw = get_last_draw(df)

            if not df.empty:
                precomputed_white_ball_freq, precomputed_powerball_freq = frequency_analysis(df)
                precomputed_last_draw_date_str = last_draw['Draw Date']
                precomputed_hot_numbers, precomputed_cold_numbers = hot_cold_numbers(df, precomputed_last_draw_date_str)
                precomputed_monthly_balls = monthly_white_ball_analysis(df, precomputed_last_draw_date_str)
                precomputed_number_age_data = get_number_age_distribution(df)
                precomputed_co_occurrence_data, precomputed_max_co_occurrence = get_co_occurrence_matrix(df)
                precomputed_powerball_position_data = get_powerball_position_frequency(df)


            return f"Data updated successfully with draw for {simulated_draw_date}.", 200
        else:
            print(f"Failed to insert data. Status: {insert_response.status_code}, Response: {insert_response.text}")
            return f"Error updating data: {insert_response.status_code} - {insert_response.text}", 500

    except requests.exceptions.RequestException as e:
        print(f"Network or HTTP error during update_powerball_data: {e}")
        # Print response content for more detailed error from Supabase
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
        return f"An internal error occurred: {e}", 500
