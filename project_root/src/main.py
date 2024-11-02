import yaml
from src.scraping.scrape_reviews import save_reviews_to_csv

# Load config.yaml
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    # Extract app list and other scraping settings from config
    apps = config['apps']
    scraping_settings = config['scraping']

    # Loop through each app and scrape reviews
    for app in apps:
        app_id = app['id']
        app_name = app['name']
        save_reviews_to_csv(
            app_id=app_id,
            app_name=app_name,
            output_dir="data/raw",
            sleep_milliseconds=scraping_settings['sleep_milliseconds'],
            lang=scraping_settings['language'],
            country=scraping_settings['country'],
            sort=scraping_settings['sort']
        )

if __name__ == "__main__":
    main()