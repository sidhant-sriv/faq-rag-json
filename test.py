import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


def send_to_discord_webhook(question: str) -> None:
    """
    Sends the unanswered question to a configured Discord webhook URL.
    """

    if not DISCORD_WEBHOOK_URL:
        logging.warning("No DISCORD_WEBHOOK_URL is set. Skipping Discord notification.")
        return

    payload = {"content": f"An unanswered question was asked:\n**{question}**"}

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        # Discord webhook returns status 204 on success
        if response.status_code != 204:
            logging.error(
                f"Failed to send question to Discord. Status: {response.status_code}, Response: {response.text}"
            )
    except Exception as ex:
        logging.error(f"Exception while sending to Discord: {ex}")


send_to_discord_webhook("Test for webhook")
