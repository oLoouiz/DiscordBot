import discord
import os
from dotenv import load_dotenv
import requests
import tensorflow as tf
import cv2

load_dotenv()

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
idCanal = os.getenv('idCanal')
idBot = os.getenv("idBot")

intents = discord.Intents.all()
client = discord.Client(command_prefix='!', intents=intents)


@client.event
async def on_ready():
    print('Logado como: {0.user}'.format(client))


@client.event
async def on_message(message):

    #inicio bot ia identificador

    a_id = message.author.id
    if (a_id != idBot):
        if message.content.startswith('/identifique'):  #precisaria colocar /identifique junto com a imagen
   
            for x in message.attachments:
                print("attachment-->",x.url)
                d_url = requests.get(x.url)
                file_name = x.url.split('/')[-1]
                with open(file_name, "wb") as f:
                    f.write(d_url.content)

            loaded_model = tf.keras.saving.load_model("model_ants-bees_V4.h5")
            image_size = (256, 256)

            imagem = cv2.imread(file_name)
            imagem = cv2.resize(imagem, image_size)
            imagem = imagem.reshape((1,256,256,3))
            imagem = tf.cast(imagem/255. ,tf.float32)

            if(loaded_model.predict(imagem) >= 0.5):
                await message.channel.send('É uma abelha 🐝')
            else:
                await message.channel.send('É uma formiga 🐜')
            os.remove(file_name)
#fim 

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



