from discord.ext import commands
import discord
import logging
import os
import asyncio

try:
    import uvloop  # type: ignore
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

except ModuleNotFoundError:
    pass


frmt = '[%(asctime)-15s] [%(levelname)s] %(message)s'
log = logging.getLogger('discord')
logging.basicConfig(level=20, format=frmt, datefmt='%m/%d/%Y | %I:%M:%S')

def get_prefix(bot, message):
    if message.content.lower().startswith('a!'):
        message_prefix = message.content[:2]
        return message_prefix
    else:
        return 'a!'

intents = discord.Intents.default()
intents.webhooks = True
intents.guild_reactions = True
intents.emojis = True
bot = commands.Bot(command_prefix=get_prefix, intents=intents, help_command=None, case_insensitive=True)
bot.PATH = os.path.dirname(os.path.abspath(__file__))

@bot.event
async def on_ready():
    log.log(20, " READY ")

@bot.event
async def on_message(message):
    if message.content.lower().startswith(">gojobhai"):
        embed = discord.Embed(color=0x36393f)
        embed.set_image(
            url="https://cdn.discordapp.com/attachments/313405009536614402/896864488135987230/gojo-satoru-2-7309.png"
        )
        await message.channel.send(embed=embed)

    await bot.process_commands(message)

cogs = (
    'jishaku',
    'cogs.date_solver',
    'cogs.cards'
)

for cog in cogs:
    bot.load_extension(cog)

bot.run("ODgzNjkxNDE3MDQwNDcwMDI3.YTNnxA.8Rvv1oPfU4_Cgq8Zq3VivhfI5wY")
