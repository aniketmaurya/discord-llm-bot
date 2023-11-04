import os

import discord
from dotenv import load_dotenv

from qna.retriever import create_retriever

load_dotenv()


TOKEN = os.environ.get("DISCORD_BOT_TOKEN")


class MyClient(discord.Client):
    retriever = create_retriever()

    def search_document(self, query):
        return self.retriever.get_relevant_documents(query=query)[0].page_content

    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return
        print(f"Message from {message.author}: {message.content}")

        if message.content.startswith("!help"):
            query = message.content.replace("!help", "")
            result = self.search_document(query=query)[:2000]
            await message.reply(result, mention_author=True)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(TOKEN)
