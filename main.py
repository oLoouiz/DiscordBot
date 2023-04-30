import discord
import os
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
idCanal = os.getenv('idCanal')

intents = discord.Intents.all()
client = discord.Client(command_prefix='!', intents=intents)


@client.event
async def on_ready():
    print('Logado como: {0.user}'.format(client))


@client.event
async def on_message(message):
    print("message-->", message)
    if message.author == client.user:
        return

    if message.channel.id == idCanal:
        if message.content.startswith('tchau'):
            await message.channel.send('Tchau!')

        if message.content.startswith('oi'):
            await message.channel.send(file=discord.File('oi.jpg'))

    if message.content == "!apagar":
        await message.channel.send("Apagando todas as mensagens do canal...")
        await message.channel.purge(limit=None)

client.run("token do bot")
