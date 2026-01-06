from urllib.parse import quote_plus
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import dotenv
import os
import tqdm
import random

dotenv.load_dotenv()
CHROME_PROFILE_PATH = os.getenv("CHROME_PROFILE_PATH")
CHROME_EXECUTABLE_PATH = os.getenv("CHROME_EXECUTABLE_PATH")


# Global Constants
PEOPLE_ALSO_ASK_QUESTION_CSX = ".CSkcDe"
PEOPLE_ALSO_ASK_QUESTION_BUTTON_CSX = ".L3Ezfd"
CLICK_COUNT_THRESHOLD = 10
TIMEOUT = 5000 # 5s


async def scrape_paa_unbiased(
    query,
    query_id_=None,
):
    '''
    Scrapes the Google AIO output.
    '''
    
    p = await async_playwright().start()
    browser = await p.chromium.launch_persistent_context(
        CHROME_PROFILE_PATH,
        headless=False,
        executable_path=CHROME_EXECUTABLE_PATH,
    )
        
    if browser is None:
        raise NotImplementedError()
    
    data = {}
    
    URL = "https://www.google.com/search?q=" + quote_plus(query)
    try:
        page = await browser.new_page()
        await page.goto(URL, timeout=TIMEOUT)
        await page.wait_for_selector(PEOPLE_ALSO_ASK_QUESTION_CSX, timeout=150000)
        
        click_history = []
        click_count = 0
        elements_prev = []
        while click_count < CLICK_COUNT_THRESHOLD:
            elements_curr = await page.locator(PEOPLE_ALSO_ASK_QUESTION_CSX).all_inner_texts()
            index_set = [i for i in range(len(elements_curr)) if elements_curr[i] not in elements_prev]
            index_to_click = random.choice(index_set)
            element = elements_curr[index_to_click]
            data[element] = {
                'question_rank': len(data),
                'click_history': click_history.copy()
            }
            
            await page.locator(PEOPLE_ALSO_ASK_QUESTION_BUTTON_CSX).nth(index_to_click).click(timeout=TIMEOUT)
            await page.wait_for_timeout(3000)
            click_count += 1
            click_history.append(element)
            elements_prev = elements_curr

    except Exception as e:
        print(f"Try to scrape random sample PAA {f" for query {query_id_}" if query_id_ is not None else ""}. Error occured: {e}")
    
    await browser.close()
    await p.stop()
    
    return data



with open('./data/queries.txt', 'r') as f:
    queries = f.read().splitlines()

res = pd.DataFrame(columns=['query_no', 'query', 'question_rank', 'question', 'click_history'])
for i in tqdm.trange(len(queries)):
    query = queries[i]
    for k in range(1, 3):
        init_seed = (37 * i * k + 3 * k + 42) % 100000
        random.seed(init_seed)
        data = asyncio.run(scrape_paa_unbiased(query=query, query_id_=i))
        for key in data:
            res.loc[len(res), :] = [i, query, data[key]['question_rank'], key, data[key]['click_history']]
    if i % 10 == 0:
        res.to_csv('./data/paa_unbiased.csv', index=False)

res.to_csv('./data/paa_unbiased.csv', index=False)