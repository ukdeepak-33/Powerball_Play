from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import random
from itertools import combinations
import math
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load historical data
def load_historical_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please provide a valid file path.")
    df = pd.read_csv(file_path, sep='\t')
    df['Draw Date'] = pd.to_datetime(df['Draw Date'])
    return df

# Get the last draw result
def get_last_draw(df):
    return df.iloc[-1]

# Check if the generated numbers match any historical draw
def check_exact_match(df, white_balls):
    for _, row in df.iterrows():
        historical_white_balls = row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
        if set(white_balls) == set(historical_white_balls):
            return True
    return False

# Generate random numbers for Powerball
def generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None):
    while True:
        # Generate white balls (5 numbers from the specified range, excluding excluded numbers)
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        white_balls = random.sample(available_numbers, 5)

        # Ensure two numbers are from Group A
        group_a_numbers = [num for num in white_balls if num in group_a]
        if len(group_a_numbers) < 2:
            continue

        # Generate Powerball (1 number from the specified range)
        powerball = random.randint(powerball_range[0], powerball_range[1])

        # Ensure the generated numbers do not match the last draw
        last_white_balls = df.iloc[-1][['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
        if set(white_balls) == set(last_white_balls) and powerball == df.iloc[-1]['Powerball']:
            continue

        # Ensure the generated numbers do not exactly match any previous 5 main numbers
        if check_exact_match(df, white_balls):
            continue

        # Check odd/even condition
        even_count = sum(1 for num in white_balls if num % 2 == 0)
        odd_count = 5 - even_count

        if odd_even_choice == "All Even" and even_count != 5:
            continue
        elif odd_even_choice == "All Odd" and odd_count != 5:
            continue
        elif odd_even_choice == "3 Even / 2 Odd" and (even_count != 3 or odd_count != 2):
            continue
        elif odd_even_choice == "3 Odd / 2 Even" and (odd_count != 3 or even_count != 2):
            continue
        elif odd_even_choice == "1 Even / 4 Odd" and (even_count != 1 or odd_count != 4):
            continue
        elif odd_even_choice == "1 Odd / 4 Even" and (odd_count != 1 or even_count != 4):
            continue

        # Check combo condition (only if combo is selected)
        if combo_choice != "No Combo":
            if combo_choice == "2-combo":
                combo_size = 2
            elif combo_choice == "3-combo":
                combo_size = 3
            else:
                combo_size = 0

            if combo_size > 0:
                combos = list(combinations(white_balls, combo_size))
                if not combos:
                    continue

        # Check high/low balance condition
        if high_low_balance is not None:
            low_numbers = [num for num in white_balls if num <= 34]
            high_numbers = [num for num in white_balls if num >= 35]
            if len(low_numbers) < high_low_balance[0] or len(high_numbers) < high_low_balance[1]:
                continue

        break

    return white_balls, powerball

# Check if generated numbers match historical data
def check_historical_match(df, white_balls, powerball):
    for _, row in df.iterrows():
        historical_white_balls = row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
        historical_powerball = row['Powerball']
        if set(white_balls) == set(historical_white_balls) and powerball == historical_powerball:
            return row['Draw Date']
    return None

# Frequency analysis of white balls and Powerball
def frequency_analysis(df):
    # Frequency of white balls (1–69)
    white_balls = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)

    # Frequency of Powerball (1–26)
    powerball_freq = df['Powerball'].value_counts().reindex(range(1, 27), fill_value=0)

    return white_ball_freq, powerball_freq

# Hot and cold numbers analysis
def hot_cold_numbers(df, last_draw_date):
    # Filter data for the last year
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    recent_data = df[df['Draw Date'] >= one_year_ago]

    # Frequency of white balls in the last year
    white_balls = recent_data[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts()

    # Hot numbers (14 most frequent)
    hot_numbers = white_ball_freq.nlargest(14)

    # Cold numbers (14 least frequent)
    cold_numbers = white_ball_freq.nsmallest(14)

    return hot_numbers, cold_numbers

# Monthly white ball analysis
def monthly_white_ball_analysis(df, last_draw_date):
    # Filter data for the last 6 months
    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    recent_data = df[df['Draw Date'] >= six_months_ago]

    # Group by month and get the white balls
    recent_data['Month'] = recent_data['Draw Date'].dt.to_period('M')
    monthly_balls = recent_data.groupby('Month')[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].apply(lambda x: x.values.flatten())

    return monthly_balls

# Sum of main balls analysis
def sum_of_main_balls(df):
    # Calculate the sum of the 5 white balls for each draw
    df['Sum'] = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    return df[['Draw Date', 'Sum']]

# Find past results with a specific sum of white balls
def find_results_by_sum(df, target_sum):
    # Calculate the sum of the 5 white balls for each draw if 'Sum' column does not exist
    if 'Sum' not in df.columns:
        df['Sum'] = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    # Filter rows where the sum matches the target sum
    results = df[df['Sum'] == target_sum]
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Sum']]

# Save generated numbers to a file
def save_generated_numbers(white_balls, powerball, file_path="generated_numbers.txt"):
    with open(file_path, "a") as file:
        file.write(f"White Balls: {white_balls}, Powerball: {powerball}\n")
    print(f"Numbers saved to {file_path}")

# Simulate multiple draws
def simulate_multiple_draws(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
    results = []
    for _ in range(num_draws):
        white_balls, powerball = generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers)
        results.append(white_balls + [powerball])
    
    # Flatten the results and count frequencies
    all_numbers = [num for draw in results for num in draw]
    freq = pd.Series(all_numbers).value_counts().sort_index()
    
    print(f"Frequency of Numbers in {num_draws} Draws:")
    print(freq)

# Function to calculate combinations (n choose k)
def calculate_combinations(n, k):
    return math.comb(n, k)

# Function to calculate winning probability
def winning_probability(white_ball_range, powerball_range):
    # Calculate total white ball combinations
    total_white_balls = white_ball_range[1] - white_ball_range[0] + 1
    white_ball_combinations = calculate_combinations(total_white_balls, 5)

    # Calculate total Powerball numbers
    total_powerballs = powerball_range[1] - powerball_range[0] + 1

    # Calculate total combinations
    total_combinations = white_ball_combinations * total_powerballs

    # Calculate probability as "1 in X" and as a percentage
    probability_1_in_x = f"1 in {total_combinations:,}"
    probability_percentage = f"{1 / total_combinations * 100:.10f}%"

    return probability_1_in_x, probability_percentage

# Function to calculate partial match probabilities
def partial_match_probabilities(white_ball_range, powerball_range):
    total_white_balls = white_ball_range[1] - white_ball_range[0] + 1
    total_powerballs = powerball_range[1] - powerball_range[0] + 1

    # Calculate probabilities for matching 0 to 5 white balls and Powerball
    probabilities = {}
    for matched_white_balls in range(6):
        # Combinations for matched white balls
        matched_combinations = calculate_combinations(5, matched_white_balls)
        # Combinations for unmatched white balls
        unmatched_combinations = calculate_combinations(total_white_balls - 5, 5 - matched_white_balls)
        # Total combinations for this scenario
        total_scenario_combinations = matched_combinations * unmatched_combinations * total_powerballs
        # Probability
        probability = total_scenario_combinations / (calculate_combinations(total_white_balls, 5) * total_powerballs)
        probabilities[f"Match {matched_white_balls} White Balls + Powerball"] = f"{1 / probability:,.0f} to 1"

    return probabilities

# Export analysis results to CSV
def export_analysis_results(df, file_path="analysis_results.csv"):
    df.to_csv(file_path, index=False)
    print(f"Analysis results saved to {file_path}")

# Find last draw dates for individual numbers
def find_last_draw_dates_for_numbers(df, white_balls, powerball):
    """
    Find the last draw date for each individual number (white balls and Powerball).
    Returns a dictionary with the last draw date for each number.
    """
    last_draw_dates = {}

    # Check last draw date for each white ball
    for number in white_balls:
        # Iterate through the historical data in reverse order (most recent first)
        for _, row in df[::-1].iterrows():
            historical_white_balls = row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
            if number in historical_white_balls:
                last_draw_dates[f"White Ball {number}"] = row['Draw Date']
                break  # Stop searching after finding the most recent draw date

    # Check last draw date for the Powerball
    for _, row in df[::-1].iterrows():
        if powerball == row['Powerball']:
            last_draw_dates[f"Powerball {powerball}"] = row['Draw Date']
            break  # Stop searching after finding the most recent draw date

    return last_draw_dates

# Modify 3 out of 5 white balls and the Powerball
def modify_combination(df, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers):
    """
    Modify 3 out of 5 white balls and the Powerball in a previously drawn combination.
    """
    # Randomly select 3 indices to modify
    indices_to_modify = random.sample(range(5), 3)
    
    # Generate new numbers for the selected indices
    for i in indices_to_modify:
        while True:
            new_number = random.randint(white_ball_range[0], white_ball_range[1])
            if new_number not in excluded_numbers and new_number not in white_balls:
                white_balls[i] = new_number
                break
    
    # Generate a new Powerball
    while True:
        new_powerball = random.randint(powerball_range[0], powerball_range[1])
        if new_powerball not in excluded_numbers and new_powerball != powerball:
            powerball = new_powerball
            break
    
    # Convert all numbers to native Python integers
    white_balls = [int(num) for num in white_balls]
    powerball = int(powerball)
    
    return white_balls, powerball

# Default file path
file_path = '/Users/bunny/Powerball_App/powerball_results_02.tsv'

# Load historical data
try:
    df = load_historical_data(file_path)
except FileNotFoundError as e:
    print(e)
    exit()

# Get the last draw result
last_draw = get_last_draw(df)

# Group A numbers
group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]

# Default ranges and excluded numbers
white_ball_range = (1, 69)
powerball_range = (1, 26)
excluded_numbers = []


@app.route('/')
def index():
    last_draw = get_last_draw(df)  # Call the function and store the result
    last_draw_dict = last_draw.to_dict()  # Convert the Series to a dictionary
    return render_template('index.html', last_draw=last_draw_dict)

@app.route('/generate', methods=['POST'])
def generate():
    odd_even_choice = request.form.get('odd_even_choice', 'All Even')
    combo_choice = request.form.get('combo_choice', 'No Combo')
    white_ball_min = int(request.form.get('white_ball_min', 1))
    white_ball_max = int(request.form.get('white_ball_max', 69))
    white_ball_range = (white_ball_min, white_ball_max)
    powerball_min = int(request.form.get('powerball_min', 1))
    powerball_max = int(request.form.get('powerball_max', 26))
    powerball_range = (powerball_min, powerball_max)
    excluded_numbers = [int(num.strip()) for num in request.form.get('excluded_numbers', '').split(",")] if request.form.get('excluded_numbers') else []
    high_low_balance = tuple(map(int, request.form.get('high_low_balance', '').split())) if request.form.get('high_low_balance') else None

    # Generate the numbers
    white_balls, powerball = generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance)

    # Find the last draw date for each individual number
    last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

    # Fetch the last draw result and convert it to a dictionary
    last_draw = get_last_draw(df)
    last_draw_dict = last_draw.to_dict()

    # Render the template with the generated numbers and last draw dates
    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw_dict, 
                           last_draw_dates=last_draw_dates)

@app.route('/generate_modified', methods=['POST'])
def generate_modified():
    # Randomly select a previously drawn combination
    random_row = df.sample(1).iloc[0]
    white_balls = random_row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist()
    powerball = random_row['Powerball']

    # Modify 3 out of 5 white balls and the Powerball
    white_balls, powerball = modify_combination(df, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers)

    # Ensure the modified combination has never been drawn before
    while check_exact_match(df, white_balls):
        white_balls, powerball = modify_combination(df, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers)

    # Find the last draw date for each individual number
    last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

    # Fetch the last draw result and convert it to a dictionary
    last_draw = get_last_draw(df)
    last_draw_dict = last_draw.to_dict()

    # Render the template with the generated numbers and last draw dates
    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw_dict, 
                           last_draw_dates=last_draw_dates)

@app.route('/frequency_analysis')
def frequency_analysis_route():
    white_ball_freq, powerball_freq = frequency_analysis(df)
    last_draw = get_last_draw(df)  # Fetch the last draw result
    return render_template('index.html', 
                           white_ball_freq=white_ball_freq.to_dict(), 
                           powerball_freq=powerball_freq.to_dict(), 
                           last_draw=last_draw.to_dict())

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    hot_numbers, cold_numbers = hot_cold_numbers(df, last_draw['Draw Date'])
    return render_template('index.html', 
                           hot_numbers=hot_numbers.to_dict(), 
                           cold_numbers=cold_numbers.to_dict(), 
                           last_draw=last_draw.to_dict())

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    monthly_balls = monthly_white_ball_analysis(df, last_draw['Draw Date'])
    return render_template('index.html', 
                           monthly_balls=monthly_balls.to_dict(), 
                           last_draw=last_draw.to_dict())

@app.route('/sum_of_main_balls')
def sum_of_main_balls_route():
    sum_data = sum_of_main_balls(df)
    return render_template('index.html', 
                           sum_data=sum_data.to_dict('records'), 
                           last_draw=last_draw.to_dict())

@app.route('/find_results_by_sum', methods=['POST'])
def find_results_by_sum_route():
    target_sum = int(request.form.get('target_sum'))
    results = find_results_by_sum(df, target_sum)
    return render_template('index.html', 
                           results=results.to_dict('records'), 
                           last_draw=last_draw.to_dict())

@app.route('/simulate_multiple_draws', methods=['POST'])
def simulate_multiple_draws_route():
    num_draws = int(request.form.get('num_draws'))
    simulate_multiple_draws(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws)
    return redirect(url_for('index'))

@app.route('/winning_probability')
def winning_probability_route():
    probability_1_in_x, probability_percentage = winning_probability(white_ball_range, powerball_range)
    return render_template('index.html', 
                           probability_1_in_x=probability_1_in_x, 
                           probability_percentage=probability_percentage, 
                           last_draw=last_draw.to_dict())

@app.route('/partial_match_probabilities')
def partial_match_probabilities_route():
    probabilities = partial_match_probabilities(white_ball_range, powerball_range)
    return render_template('index.html', 
                           probabilities=probabilities, 
                           last_draw=last_draw.to_dict())

@app.route('/export_analysis_results')
def export_analysis_results_route():
    export_analysis_results(df)
    flash("Analysis results exported to analysis_results.csv", 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

    if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
