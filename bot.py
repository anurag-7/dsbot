from discord.ext import commands
import discord
import logging
import os
import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

except ModuleNotFoundError:
    pass


frmt = '[%(asctime)-15s] [%(levelname)s] %(message)s'
log = logging.getLogger('discord')
logging.basicConfig(level=20, format=frmt, datefmt='%m/%d/%Y | %I:%M:%S')


intents = discord.Intents.default()
intents.webhooks = True
intents.guild_reactions = True
intents.emojis = True
bot = commands.Bot(command_prefix='a!', intents=intents, help_command=None)
bot.PATH = os.path.dirname(os.path.abspath(__file__))

@bot.event
async def on_ready():
    log.log(20, " READY ")

@bot.event
async def on_message(message):
    if message.content.lower().startswith(">gojobhai"):
        embed = discord.Embed(color=0x36393f)
        embed.set_image(url="https://cdn.discordapp.com/attachments/313405009536614402/894852023323881532/gojo-satoru-3-1.png")
        await message.channel.send(embed=embed)

    await bot.process_commands(message)

cogs = (
    'jishaku',
    'cogs.date_solver'
)

for cog in cogs:
    bot.load_extension(cog)

bot.run("ODgzNjkxNDE3MDQwNDcwMDI3.YTNnxA.8Rvv1oPfU4_Cgq8Zq3VivhfI5wY")