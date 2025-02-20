# Powerball_app

The Powerball Simulator Flask Application is a web-based tool that allows users to generate Powerball numbers, analyze historical Powerball data, and calculate winning probabilities. The application is built using Python and the Flask framework, providing an interactive and user-friendly interface for Powerball enthusiasts.

# Features

**Generate Powerball Numbers:**

1.Customize number ranges for white balls and Powerball.

2.Exclude specific numbers.

3.Apply advanced filters such as prime numbers, multiples, and high/low balance.

4.Choose odd/even combinations and combo types.

# Historical Data Analysis:

1.Frequency analysis of white balls and Powerball numbers.

2.Hot and cold numbers analysis (most and least frequent numbers in the last year).

3.Monthly white ball frequency analysis.

4.Sum of main balls analysis.

# Winning Probability Calculator:

1.Calculate the probability of winning the jackpot.

2.Calculate partial match probabilities (e.g., matching 3 white balls + Powerball).

3.Simulate Multiple Draws:

4.Simulate multiple Powerball draws to analyze number frequencies.

# Export Results:

1.Export generated numbers and analysis results to CSV files.

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

git clone https://github.com/your-repo/powerball_app.git

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

1. Generate Powerball Numbers
   
   On the homepage, customize the number ranges, excluded numbers, and advanced filters.
   
   Click Generate Numbers to generate Powerball numbers based on your preferences.

2. Analyze Historical Data
   
   Use the Analysis section to perform various analyses:
   
   **Frequency Analysis:**
   
    View the frequency of white balls and Powerball numbers.
   
   **Hot and Cold Numbers:**
   
   Identify the most and least frequent numbers in the last year.
   
   **Monthly White Ball Analysis:**
   
   Analyze white ball frequencies over the last 6 months.
   
   **Sum of Main Balls:**
   
   Calculate the sum of the 5 white balls for each draw.

4. Winning Probability Calculator
   
   Use the Winning Probability section to calculate:
   
   The probability of winning the jackpot.
   
   Partial match probabilities (e.g., matching 3 white balls + Powerball).

5. Simulate Multiple Draws
   Enter the number of draws to simulate and click Simulate to analyze the frequency of numbers across multiple draws.

6. Export Results
   Click Export Analysis Results to save the analysis results to a CSV file.

# File Structure

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

Enjoy using the Powerball Simulator Flask Application! Good luck with your Powerball numbers! 🎱🍀

This README.md provides a comprehensive guide to the Powerball Simulator Flask application, making it easy for users to set up, use, and contribute to the project.

