import asyncio
import threading

import websockets
import json

# Keywords used by THW for another project
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




class PostCollector:
    def __init__(self, url="wss://jetstream2.us-west.bsky.network/subscribe"):
        self.url = url
        self.post_queue = asyncio.Queue()
        self._running = False

    async def _connect_to_websocket(self):
        while True:
            try:
                async with websockets.connect(self.url) as websocket:
                    print("Connected to WebSocket server!")

                    while True:
                        await websocket.ping()  # Keep the connection alive
                        message = await websocket.recv()

                        try:
                            data = json.loads(message)  # Assuming the message is JSON
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            continue  # Skip this message if it's not valid JSON

                        if data.get("kind") == "commit":
                            if data["commit"].get("operation") == "create":
                                collection = data["commit"].get("collection")
                                if collection == "app.bsky.feed.post":
                                    post = data["commit"].get("record")
                                    if post and "de" in set(post.get("langs", [])):
                                        # Add the new post to the queue
                                        post["platform"] = "bluesky"
                                        await self.post_queue.put(post)
            except websockets.exceptions.ConnectionClosedError:
                print("Connection closed, reconnecting...")

    def start(self):
        """Starts the WebSocket connection in the background."""
        if not self._running:
            self._running = True
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Start the event loop in a separate thread
            thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            thread.start()

            # Schedule the WebSocket task
            asyncio.run_coroutine_threadsafe(self._connect_to_websocket(), self.loop)

    def stop(self):
        """Stops the WebSocket connection."""
        self._running = False
        self.loop.call_soon_threadsafe(self.loop.stop)

    def get_all_posts(self, num_posts=32):
        """Returns all posts collected so far."""
        posts = []
        while not self.post_queue.empty() and len(posts) < num_posts:
            post = self.post_queue.get_nowait()  # Non-blocking call
            posts.append(post)
            for key_word in storm_key_words:
                if key_word.lower() in post['text'].lower():
                    #posts.append(post)
                    break
        return posts


# Example usage:
async def main():
    # Create the PostCollector instance
    collector = PostCollector()

    # Start the collector (run the WebSocket connection in the background)
    collector.start()


    # Get all posts collected so far
    print("All collected posts:")
    while True:
        all_posts = collector.get_all_posts()
        print(all_posts)
        await asyncio.sleep(5)

    # Stop the collector
    collector.stop()


# Run the event loop
if __name__ == "__main__":
    asyncio.run(main())
