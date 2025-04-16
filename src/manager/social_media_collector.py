import json
import time

from src.manager.mastodon_firehose import PostCollector as MastodonCollector
from src.manager.blue_sky_firehose import PostCollector as BlueskyCollector

class SocialMediaCollector:
    def __init__(self):
        self.mastodon_collector = MastodonCollector()
        self.bluesky_collector = BlueskyCollector()


    def start(self):
        self.mastodon_collector.start()
        self.bluesky_collector.start()

    def stop(self):
        self.mastodon_collector.stop()
        self.bluesky_collector.stop()


    def get_all_posts(self):
        mastodon_posts = self.mastodon_collector.get_all_posts()
        print(f"Got {len(mastodon_posts)} mastodon posts")

        bs_posts = self.bluesky_collector.get_all_posts()
        print(f"Got {len(bs_posts)} bluesky posts")

        return mastodon_posts + bs_posts


if __name__ == "__main__":
    collector = SocialMediaCollector()
    collector.start()
    while True:
        posts = collector.get_all_posts()
        for post in posts:
            print(json.dumps(post, indent=4))
        time.sleep(1)