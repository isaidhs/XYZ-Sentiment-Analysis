# Import necessary libraries
from google_play_scraper import Sort, reviews_all, reviews
import pandas as pd
import os

def get_reviews(app_id, sleep_milliseconds=0, lang='id', country='id', sort=Sort.NEWEST):
    """
    Fetches all reviews for a given Google Play app ID.
    
    Parameters:
    - app_id (str): Google Play app ID.
    - sleep_milliseconds (int): Delay between requests.
    - lang (str): Language for reviews.
    - country (str): Country code.
    - sort (Sort): Sorting order of reviews.

    Returns:
    - list of dict: A list of reviews for the specified app.
    """
    # Map sort string to Sort enum
    sort_option = Sort.NEWEST if sort.lower() == "newest" else Sort.MOST_RELEVANT
    result = reviews(
        app_id,
        # sleep_milliseconds=sleep_milliseconds,
        lang=lang,
        country=country,
        sort=sort_option,
        count=1000
    )
    return result

def save_reviews_to_csv(app_id, app_name, output_dir="data/raw", sleep_milliseconds=0, lang='id', country='id', sort="newest"):
    """
    Fetches reviews for a specific app ID and saves them to a CSV file.
    
    Parameters:
    - app_id (str): Google Play app ID.
    - app_name (str): Name of the app to use in the CSV filename.
    - output_dir (str): Directory to save the CSV file.
    - sleep_milliseconds (int): Delay between requests.
    - lang (str): Language for reviews.
    - country (str): Country code.
    - sort (str): Sorting order of reviews ("newest" or "most_relevant").
    """
    print(f"Fetching reviews for {app_name}...")
    reviews = get_reviews(app_id, sleep_milliseconds=sleep_milliseconds, lang=lang, country=country, sort=sort)
    df = pd.DataFrame(reviews[0])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"{app_name}_reviews.csv")
    df.to_csv(csv_path, index=False)
    print(f"Reviews for {app_name} saved to {csv_path}")