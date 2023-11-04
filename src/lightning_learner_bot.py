import os
from discord import Message, User
import asyncio
from concurrent.futures import ThreadPoolExecutor as Executor
from loguru import logger

import discord
from dotenv import load_dotenv

from qna.retriever import LightningRetriever
from qna.llms import LlamaCppLLM

load_dotenv()


TOKEN = os.environ.get("LEARNER_BOT_TOKEN")


class MyClient(discord.Client):
    retriever = LightningRetriever()
    llm = LlamaCppLLM(lazy=True, n_ctx=1028)
    pool = Executor(2)

    def run_in_loop(self, query, document):
        result = self.llm(query=query, document=document)
        return result

    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message: Message):
        if message.author.id == self.user.id:
            return

        logger.info(message.content)
        logger.info(self.user.mention)
        # message.content.startswith("!help"):
        if message.content.startswith(self.user.mention):
            await message.channel.typing()
            query = message.content.replace(self.user.mention, "")
            result = self.retriever(query=query)
            document = result["document"]
            distance = result["distance"]
            source = result["source"]
            logger.info(f"query: {query}\ndistance: {distance}")

            if distance >= 0.7 and distance < 1.1:
                thought = f"I am still learning and will try my best to answer you on what I know. I am reading **{source}** to formulate an answer for you. Please give me a moment..."
                await message.reply(thought, mention_author=True)

            elif distance >= 1.1:
                thought = f"Sorry, I didn't you? Could you please try rephrasing or providing more context so that I can help better?"
                await message.reply(thought, mention_author=True)
                return

            loop = asyncio.get_running_loop()
            llm_output = await loop.run_in_executor(
                self.pool, self.run_in_loop, query, document
            )
            output = (
                f"This is what I was able to understand from {source}. I still have a lot to learn so please excuse me if I am wrong...\n\n"
                + llm_output
            )
            try:
                await message.reply(output, mention_author=True)
            except Exception as e:
                logger.exception(e)
                await message.reply(
                    "Sorry I faced some issue while getting back to you!"
                )


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(TOKEN)
