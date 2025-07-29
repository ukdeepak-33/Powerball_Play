Powerball Analysis & Generation Application
This Flask-based web application provides comprehensive tools for Powerball lottery analysis, number generation, and historical data tracking. It allows users to explore various statistical trends, generate numbers based on different strategies, and track their generated picks.

The application leverages historical Powerball draw data, stored and managed via Supabase, to power its analytical features and smart number generation.

Features
Latest Draw Display: View the most recent official Powerball draw.

Number Generation Strategies:

Random Generation: Generate sets of numbers with filters for odd/even splits, sum ranges, high/low balance, and excluded numbers.

Group A Strategy: Generate numbers ensuring a specific count of "Group A" numbers (a predefined set of historically significant numbers).

User Starting Pair: Generate numbers starting with two user-defined white balls, with the remaining numbers generated in ascending order from specific tens ranges.

AI-Powered Smart Picks: Utilize an AI assistant to analyze historical trends and suggest "smart" number combinations.

Comprehensive Analysis Tools:

Frequency Analysis: See how often each white ball and Powerball has been drawn.

Hot & Cold Numbers: Identify the most and least frequently drawn numbers over the last year.

Monthly Trends: Analyze white ball and Powerball frequencies month-by-month, including "miss streaks."

Sum of Main Balls Analysis: Explore the sums of the five white balls, including frequency and historical ranges.

Sum Trends & Gaps: Detailed analysis of sum ranges, including most/least frequent sums and missing sums.

Number Age Distribution: Track how many draws each number has "missed" since its last appearance.

Co-occurrence Analysis: Visualize which pairs of white balls appear together most often.

Triplets Analysis: Identify the most frequent three-number combinations.

Odd/Even Trends: Analyze the distribution of odd and even numbers in recent draws.

Consecutive Numbers Trends: Track the frequency of consecutive number pairs in recent and yearly draws.

Yearly White Ball Trends: Visualize white ball frequencies across different years.

Powerball Frequency by Year: See how often each Powerball has been drawn per year.

Grouped Patterns Analysis: Analyze frequency of pairs and triplets within specific number ranges (e.g., 10s, 20s) across all years.

Grouped Patterns Yearly Comparison: Compare grouped patterns year-by-year for selected number ranges.

Boundary Crossing Pairs Trends: Analyze the frequency of pairs that cross "tens" boundaries (e.g., 9-10, 19-20).

Special Patterns Analysis: Identify patterns like "tens-apart" numbers, "same last digit" numbers, and "repeating digit" numbers.

Search Functionality:

Search by Sum: Find historical draws that sum up to a specific total.

Search by First White Ball: Find historical draws where a specific number appeared as the first white ball.

Strict Positional Search: Search for draws where specific numbers appeared in exact positions.

Data Management:

Save Generated Picks: Save individual or multiple generated number sets to your database.

Manually Add Official Draws: Add new official draw results to your database.

Update Data (Simulated): Simulate adding a new draw to keep your historical data fresh (for demonstration purposes if not connected to a live API).

Export Analysis Results: Export the full historical dataset to a CSV file.

Historical Match Analysis:

Analyze your generated or manually selected picks against historical draws to see how many times they would have matched various prize tiers.

User-Friendly Interface: Responsive design with a clean, intuitive layout built with Tailwind CSS.

Technologies Used
Backend: Flask (Python)

Data Analysis: Pandas (Python)

Database: Supabase (PostgreSQL backend with REST API)

Frontend: HTML, CSS (Tailwind CSS), JavaScript (Chart.js for visualizations)

AI Integration: Google Gemini API

Setup and Installation
Follow these steps to get the application up and running on your local machine.

1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name> # e.g., cd powerball-analysis-app

2. Set up a Python Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

(If you don't have a requirements.txt file, you'll need to create one. You can generate it after installing Flask, Pandas, Requests, and NumPy: pip freeze > requirements.txt)
Minimum required packages: Flask, pandas, requests, numpy, python-dotenv (if using .env for local variables).

4. Supabase Configuration
This application uses Supabase for historical data storage and generated number tracking.

Create a Supabase Project:

Go to Supabase and create a new project.

Create Tables:

powerball_draws: This table will store historical Powerball draw results.

Columns: id (Primary Key, UUID, default gen_random_uuid()), Draw Date (DATE), Number 1 (INT), Number 2 (INT), Number 3 (INT), Number 4 (INT), Number 5 (INT), Powerball (INT).

generated_powerball_numbers: This table will store user-generated and saved picks.

Columns: id (Primary Key, UUID, default gen_random_uuid()), generated_date (TIMESTAMPZ, default now()), number_1 (INT), number_2 (INT), number_3 (INT), number_4 (INT), number_5 (INT), powerball (INT).

Get Supabase Credentials:

In your Supabase project dashboard, navigate to Settings > API.

Copy your Project URL (e.g., https://abcdefghijk.supabase.co).

Copy your anon public key.

Copy your service_role secret key.

Set Environment Variables:

Create a .env file in the root of your project (same level as index.py).

Add the following lines, replacing the placeholders with your actual Supabase keys:

SUPABASE_URL="YOUR_SUPABASE_PROJECT_URL"
SUPABASE_ANON_KEY="YOUR_SUPABASE_ANON_KEY"
SUPABASE_SERVICE_KEY="YOUR_SUPABASE_SERVICE_ROLE_KEY"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY" # Optional, only needed for AI features

Important: For production deployments (e.g., Render, Vercel), set these as environment variables directly in your hosting platform's settings, not in a .env file.

5. Populate Historical Data (Optional but Recommended)
The application works best with historical data. You can manually add data via the "Manually Add Official Draw" form, or import a CSV of historical Powerball draws into your powerball_draws Supabase table.

6. Run the Application
flask run

The application should now be running at http://127.0.0.1:5000/.

Usage
Navigate through the sidebar to access different analysis and generation tools.

Home (/): Generate numbers, view the latest draw, and manually add official draws.

Analysis Pages: Explore various statistical breakdowns and trends.

My Jackpot Pick (/my_jackpot_pick): Manually select numbers and analyze them against historical data.

Generated History (/generated_numbers_history): View all your saved generated picks and analyze them against official draws.

AI Assistant (/ai_assistant): Chat with an AI that provides insights based on your historical Powerball data.

Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are welcome!

License
[Specify your license here, e.g., MIT License]
