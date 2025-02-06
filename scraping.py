import json
from datetime import datetime, timedelta
from time import sleep
import re
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_relative_date(relative_date):
    now = datetime.now()

    if 'hour' in relative_date or 'hours' in relative_date:
        hours = int(re.search(r'\d+', relative_date).group())
        return now - timedelta(hours=hours)
    elif 'day' in relative_date or 'days' in relative_date:
        days = int(re.search(r'\d+', relative_date).group())
        return now - timedelta(days=days)
    elif 'week' in relative_date or 'weeks' in relative_date:
        weeks = int(re.search(r'\d+', relative_date).group())
        return now - timedelta(weeks=weeks)
    elif 'month' in relative_date or 'months' in relative_date:
        months = int(re.search(r'\d+', relative_date).group())
        return now - timedelta(days=months * 30)
    elif 'year' in relative_date or 'years' in relative_date:
        years = int(re.search(r'\d+', relative_date).group())
        return now - timedelta(days=years * 365)
    else:
        return now

def preprocess_data(data):
    processed_data = []
    for entry in data:
        comment_text = entry.get('comment', '').strip()
        user = entry.get('user', 'Unknown')

        likes = entry.get('likes', '0')
        if isinstance(likes, str) and 'K' in likes:
            try:
                likes = int(float(likes.replace('K', '').strip()) * 1000)
            except ValueError:
                likes = 0
        else:
            likes = int(likes) if str(likes).isdigit() else 0

        number_responses = entry.get('number_responses', '0')
        if any(char.isdigit() for char in str(number_responses)):
            try:
                number_responses = int(re.search(r'\d+', str(number_responses)).group())
            except (ValueError, AttributeError):
                number_responses = 0
        else:
            number_responses = 0

        processed_entry = {
            'number': entry.get('number', 0),
            'user': user,
            'comment_text': comment_text,
            'time_ago': entry.get('time_ago', ''),
            'likes': likes,
            'number_responses': number_responses
        }

        processed_data.append(processed_entry)

    return processed_data

def scrape_youtube_comments(video_urls: list):
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(10)
    driver.maximize_window()

    comments: list[dict] = list()
    unique_comments = set()

    try:
        for url in video_urls:
            logger.info(f"Scrapping: {url}")
            driver.get(url)
            sleep(10)

            last_comment_count = 0
            scroll_attempts = 0
            max_scroll_attempts = 10

            while scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollBy(0, 700);")
                sleep(3)
                WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH, "//*[@id='content-text']")))

                try:
                    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH, "//*[@id='content-text']")))
                except:
                    logger.warning("Timeout ao carregar comentários. Continuando...")
                    break

                comments_we = driver.find_elements(By.XPATH, "//*[@id='content-text']")
                users_we = driver.find_elements(By.XPATH, "//*[@id='author-text']//span")
                date_we = driver.find_elements(By.XPATH, "//*[contains(@id, 'published-time-text')]")
                likes_we = driver.find_elements(By.XPATH, "//*[contains(@id, 'vote-count-middle')]")
                responses_we = driver.find_elements(By.XPATH, "//*[contains(@class, 'yt-core-attributed-string--white-space-no-wrap')]")

                current_comment_count = len(comments_we)

                if current_comment_count == last_comment_count:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0

                last_comment_count = current_comment_count

                for user, comment, date, like, response in zip(users_we, comments_we, date_we, likes_we, responses_we):
                    comment_text = comment.text
                    date_text = date.text if date.text else "0 seconds ago"
                    comment_date = parse_relative_date(date_text)
                    comment_id = f"{user.text}_{comment_text}"

                    if comment_id not in unique_comments:
                        unique_comments.add(comment_id)
                        comments.append({
                            "number": len(comments),
                            "user": user.text,
                            "comment": comment_text,
                            "likes": like.text if like.text else "0",
                            "number_responses": response.text if response.text else "0",
                            "time_ago": comment_date.strftime("%Y-%m-%d")
                        })

                if scroll_attempts >= max_scroll_attempts:
                    logger.info("Número máximo de tentativas de rolagem atingido. Parando...")
                    break

    except Exception as e:
        logger.fatal(f"An exception occurred while collecting comments from: '{url}'.", exc_info=True)
    finally:
        processed_comments = preprocess_data(comments)

        with open(f"youtube_comments_EM", "w", encoding="utf-8") as f:
            json.dump(dict(comments=processed_comments), f, ensure_ascii=False, indent=4)

        driver.quit()

video_urls = ["https://www.youtube.com/watch?v=RK91Ji6GCZ8","https://www.youtube.com/watch?v=Crr7j0udrc4","https://www.youtube.com/watch?v=H_a9bfeVG04",
              "https://www.youtube.com/watch?v=nyRFFP9yA0s","https://www.youtube.com/watch?v=joV-9FFoA3Q","https://www.youtube.com/watch?v=gDkuwRx14hQ",
              "https://www.youtube.com/watch?v=smQNNo2a9xc","https://www.youtube.com/watch?v=GMG-ZEG_VU4"]
scrape_youtube_comments(video_urls)