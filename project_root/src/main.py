import yaml
from src.scraping.scrape_reviews import save_reviews_to_csv
from src.modeling.topic_modeling import run_topic_modeling

def load_config(config_path="config/config.yaml"):
    """
    Load configuration from a YAML file.
    
    Parameters:
    - config_path (str): Path to the configuration YAML file.
    
    Returns:
    - dict: Configuration settings.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def run_scraping(config):
    # Extract app list and scraping settings from config
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

def run_topic_modeling_pipeline(config):
    # Get topic modeling settings from the config
    topic_modeling_settings = config['topic_modeling']
    input_file = topic_modeling_settings['input_file']

    # Run topic modeling
    run_topic_modeling(input_file=input_file)

def main():
    # Load the configuration
    config = load_config()

    # Run scraping
    run_scraping(config)

    # Run topic modeling
    run_topic_modeling_pipeline(config)
    
if __name__ == "__main__":
    main()