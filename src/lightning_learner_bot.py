import os

import discord
from dotenv import load_dotenv

from qna.retriever import LightningRetriever

load_dotenv()


TOKEN = os.environ.get("LEARNER_BOT_TOKEN")


class MyClient(discord.Client):
    retriever = LightningRetriever()

    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return
        print(f"Message from {message.author}: {message.content}")

        if message.content.startswith("!help"):
            query = message.content.replace("!help", "")
            result = self.retriever(query=query)
            document = result["document"]
            distance = result["distance"]
            source = result["source"]

            if distance >= 0.6:
                thought = f"It seems like I am not so sure about this question but I have tried my best to answer based on my knowledge. I am reading **{source}** to formulate an answer for you. Please give me a moment..."
                await message.reply(thought, mention_author=True)

            await message.reply(document[:2000], mention_author=True)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(TOKEN)
