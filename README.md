# Powerball_App

The Powerball Simulator Flask Application is a web-based tool that allows users to generate Powerball numbers, analyze historical Powerball data, and calculate winning probabilities. The application is built using Python and the Flask framework, providing an interactive and user-friendly interface for Powerball enthusiasts.The Powerball app is designed to simulate Powerball number generation, analyze historical data, and provide insights into winning probabilities. Below is a revised breakdown of its functionality:

# Features

**Generate Powerball Numbers:**

Functionality: Generates random Powerball numbers based on user-defined constraints.

Constraints:

Odd/Even Combination: Users can choose specific odd/even distributions (e.g., 3 even and 2 odd).

Combo Type: Users can specify if they want 2-number or 3-number combinations.

Range for White Balls and Powerball: Users can set custom ranges for white balls (1–69) and Powerball (1–26).

Excluded Numbers: Users can exclude specific numbers from being generated.

High/Low Balance: Users can specify a balance between high (35–69) and low (1–34) numbers.

Output: Generates 5 white balls and 1 Powerball number that meet the constraints.


# Historical Data Analysis:

Functionality: Analyzes historical Powerball data to provide insights.

Features:

Frequency Analysis: Shows how often each white ball and Powerball number has been drawn.

Hot and Cold Numbers: Identifies the most and least frequently drawn numbers in the last year.

Monthly White Ball Analysis: Provides a breakdown of white balls drawn in the last 6 months.

Sum of Main Balls: Calculates the sum of the 5 white balls for each draw and allows users to search for draws with a specific sum.

Last Draw Dates for Numbers: Finds the most recent draw date for each individual number (white balls and Powerball).


# Winning Probability Calculator:
Functionality: Calculates the probability of winning and partial matches.

Features:

Winning Probability: Calculates the overall probability of winning the Powerball jackpot.

Partial Match Probabilities: Calculates the odds of matching a subset of white balls and the Powerball.


# Export Results:

Functionality: Allows users to export analysis results and save generated numbers.

Features:

Export Analysis Results: Exports frequency analysis, hot/cold numbers, and other insights to a CSV file.

Save Generated Numbers: Saves generated Powerball numbers to a text file.

1.Export generated numbers and analysis results to CSV files.

# Web Interface

Functionality: Provides a user-friendly web interface for interacting with the app.

Features:

Forms: Users can input constraints and preferences for number generation.

Results Display: Displays generated numbers, analysis results, and historical data insights.

Flashed Messages: Provides feedback (e.g., success or error messages) to users.

# Prerequisites

Before running the application, ensure you have the following installed:

i.Python 3.7 or higher

ii.Flask

iii.Pandas

iv.NumPy

v.Plotly

vi.Matplotlib

You can install the required dependencies using the requirements.txt file.

# Installation

# Clone the Repository:

**bash**

git clone https://github.com/your-repo/Powerball_App.git

cd powerball_app

# Install Dependencies:

**bash**

pip install -r requirements.txt

**Download Historical Data:**

Place your Powerball historical data file (e.g., powerball_results_02.tsv) in the project directory.

Update the file_path variable in app.py to point to your data file.

# Run the Application:

**bash**

python app.py

**Access the Application:**

Open your web browser and navigate to http://127.0.0.1:5000/.

# Running the Application

Install the dependencies:

bash pip install -r requirements.txt

Run the Flask application:

bash python app.py

Open your web browser and navigate to http://127.0.0.1:5000/ to interact with the Powerball Simulator.

This Flask application provides a web interface for generating Powerball numbers, performing frequency analysis, and other features available in the original script.


# Usage

Generate Numbers:

Use the form to set constraints (e.g., odd/even combinations, excluded numbers).

Click "Generate Numbers" to generate Powerball numbers.

Analyze Historical Data:

Use the analysis links (e.g., Frequency Analysis, Hot/Cold Numbers) to view insights.

Calculate Probabilities:

Use the "Winning Probability" and "Partial Match Probabilities" links to view odds.

Simulate Draws:

Use the "Simulate Multiple Draws" form to simulate multiple Powerball draws.

Export Data:

Click "Export Analysis Results" to save analysis data to a CSV file.

# File Structure

app.py: Main Flask application file.

templates/index.html: HTML template for the web interface.

static/styles.css: CSS file for styling the web interface.

powerball_results_02.tsv: Historical Powerball data file (must be provided by the user).

<img width="702" alt="image" src="https://github.com/user-attachments/assets/1093d73d-af5d-4ec6-bed1-700a754c92e0" />


# Customization

**Historical Data:**

Replace the powerball_results_02.tsv file with your own Powerball historical data file.

Ensure the file is in TSV format and contains columns for Draw Date, Number 1, Number 2, Number 3, Number 4, Number 5, and Powerball.

**Group A Numbers:**

Modify the group_a list in app.py to customize the Group A numbers used in the number generation process.

**Advanced Filters:**

Adjust the filters in the generate_powerball_numbers function to add or modify constraints.

# Screenshots

Homepage

Homepage

Generated Numbers

Generated Numbers

Frequency Analysis

Frequency Analysis

# Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

# Fork the repository.

Create a new branch for your feature or bugfix.

Commit your changes and push to the branch.

Submit a pull request.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Contact

For questions or feedback, please contact:

Your Name Deepak Uppari

Email: ukdeepak09@gmail.com

GitHub:github.com/ukdeepak-33


---

### Key Changes in the Revised App:
1. **Improved Functionality**:
   - Added **last draw dates for individual numbers** to track when each number was last drawn.
   - Enhanced **probability calculations** for partial matches.
   - Added **export functionality** for analysis results.

2. **User Interface**:
   - Improved forms and results display in the web interface.
   - Added flashed messages for user feedback.

3. **Documentation**:
   - Created a detailed `README.md` file for easy setup and usage.

Let me know if you need further adjustments!

Enjoy using the Powerball Simulator Flask Application! Good luck with your Powerball numbers! 🎱🍀

This README.md provides a comprehensive guide to the Powerball Simulator Flask application, making it easy for users to set up, use, and contribute to the project.

