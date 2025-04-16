import asyncio
import datetime

import mastodon
import json

from bs4 import BeautifulSoup


thw_key_words = [
    "Aufräumarbeiten",
    "Bergung",
    "Dammbruch",
    "Dammschäden",
    "Dauerregen",
    "Deichbruch",
    "Deichschäden",
    "Einsturz",
    "Erdrutsch",
    "Evakuierung",
    "Extremwetterlage",
    "Freiwillige Helfer",
    "Geröll",
    "Gewitter",
    "Hangrutschungen",
    "Hilfsaktion",
    "Hochwasser",
    "Katastrophe",
    "Krisenstab",
    "Luftrettung",
    "Murgang",
    "Niederschlag",
    "Notunterkunft",
    "Pegel",
    "Platzregen",
    "Retentionsfläche",
    "Rettungskräfte",
    "Sandsäcke",
    "Schneeschmelze",
    "Schlammlawine",
    "Schutt",
    "Starkregen",
    "Stromausfall",
    "Sturmböe",
    "Sturzkflut",
    "Tornado",
    "Trümmer",
    "Überflutung",
    "Überschwemmung",
    "Unwetter",
    "Wasserrettung",
    "Wiederaufbau"]

# Own defined keywords
storm_key_words = [
    "sturm",
    "orkan",
    "unwetter",
    "regen",
    "hagel",
    "gewitter",
    "wind",
    "blitz",
    "wetter",
    "wasser",
    "flut",
    "hochwasser",
    "überschwemmung",
    "landunter",
    "starkregen",
    "niederschlag",
    "wetterwarnung",
    "hilfe",
    "evakuierung",
    "gefahrenlage",
    "gefahr",
    "kaputt",
    "zerstört",
    "katastrophe",
    "rettung",
    "retten",
    "scherben",
    "untergegangen",
    "netz",
    "strom",
    "ausfall",
    "stromausfall",
    "schaden",
    "schäden",
    "verletzt",
    "verletzte",
    "verletzung",
    "tot",
    "vermisst",
    "ertrunken",
    "damm",
    "deich",
    "überflutet",
    "überflutung",
    "polizei",
    "feuerwehr"]
storm_key_words = thw_key_words + storm_key_words

# Listener for Mastodon events
class Listener(mastodon.StreamListener):
    def __init__(self, post_queue):
        super().__init__()
        self.posts = post_queue

    def on_update(self, status):
        m_text = BeautifulSoup(status.content, 'html.parser').text
        m_lang = status.language
        if m_lang is None:
            m_lang = 'unknown'
        m_user = status.account.username

        app = ''
        # attribute only available on local
        if hasattr(status, 'application'):
            if status.application is not None:
                app = status.application.get('name')

        now_dt = datetime.datetime.now()

        value_dict = {
            'm_id': status.id,
            'created_at': int(now_dt.strftime('%s')),
            'created_at_str': now_dt.strftime('%Y %m %d %H:%M:%S'),
            'app': app,
            'platform': 'mastodon',
            'url': status.url,
            'lang': m_lang,
            'favourites': status.favourites_count,
            'username': m_user,
            'text': m_text
        }

        if "de" == value_dict.get("lang", "unknown"):
            self.posts.append(value_dict)

class PostCollector:
    def __init__(self, url="https://mastodon.social",
                 token = "c4mnrZIKJG0fzTfWIhRnqsh7uDtDncP-OXs3YmlSGTM"):
        self.url = url
        self.token = token


        self.post_queue = []
        self.listener = Listener(self.post_queue)
        self.stream_handler = None
        # Set up streaming to listen for public events

        self._running = False

    def start(self):
        """Starts the WebSocket connection in the background."""
        if not self._running:
            self._running = True
            # Start the WebSocket connection concurrently
            mastodon_api = mastodon.Mastodon(
                access_token=self.token,
                api_base_url=self.url
            )
            self.stream_handler = mastodon_api.stream_public(self.listener, run_async=True, local=False,
                                                             reconnect_async=True)

    def stop(self):
        """Stops the WebSocket connection."""
        self._running = False
        self.stream_handler.cancel()

    def get_all_posts(self, num_posts=32):
        """Returns all posts collected so far."""
        posts = []
        while not len(self.post_queue) == 0 and len(posts) < num_posts:
            post = self.post_queue.pop(0) # Non-blocking call
            posts.append(post)
            for key_word in storm_key_words:
                if key_word.lower() in post['text'].lower():
                    #posts.append(post)
                    break
        return posts



def listen_mastodon():
    listener = PostCollector()
    listener.start()
    while True:
        posts = listener.get_all_posts()
        for post in posts:
            print(json.dumps(post, indent=4))

        #asyncio.sleep(1)



if __name__ == "__main__":
    listen_mastodon()
