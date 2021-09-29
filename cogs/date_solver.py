import asyncio
import contextlib
import functools
import os
import re
from copy import deepcopy

import aiohttp
import aiosqlite
import cv2
import discord
import numpy as np
import pytesseract
from discord.ext import commands
from discord.ext.commands.cooldowns import BucketType

from .ext.datesolver import game

default_maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

translate = {
    "tree": 0,
    "airport": 0,
    "gas": 15,

    "taco": 25,
    "ballroom": 30,
    "coffee": 35,
    "juice": 40,
    "theater": 45,
    "restaurant": 50,
    "nightclub": 55,
    "fair": 60,
    "sandwhich": 65,

    "home": 105,
    "shopping": 110,
    "ring": 115,
    "flower": 120,

    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,

    15: "GAS",
    25: "TACO",
    30: "BALLROOM",
    35: "COFFEE",
    40: "JUICE",
    45: "THEATER",
    50: "RESTAURANT",
    55: "NIGHTCLUB",
    60: "FAIR",
    65: "SANDWICH",

    105: "HOME",
    110: "SHOPPING",
    115: "RING",
    120: "FLOWER",

    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

EMOJIS = {
    15: "⛽",
    25: "🌮",
    30: "💃🏻",
    35: "☕",
    40: "🧃",
    45: "🎭",
    50: "🍝",
    55: "🍹",
    60: "🎡",
    65: "🥪",

    105: "🏠",
    110: "🛍️",
    115: "💍",
    120: "🌼",

    0: "🔼",
    1: "🔽",
    2: "◀️",
    3: "▶️"
}

def pretty_maze(maze):
    length = len(maze[0])
    frmt = "{:>12}" * length
    for l in maze:
        print(frmt.format(*l), '\n')

def get_stats(image):

    cropped = image[0: 145, 715: 770]
    string = pytesseract.image_to_string(cropped, config="-c tessedit_char_whitelist=1234567890 --psm 11")
    stats = [int(i) for i in re.findall(r"\d{1,3}", string, re.MULTILINE)]
    total = (100 - stats.pop()) // 4
    return stats, total

class DateSolver(commands.Cog):

    def __init__(self, bot):
        template_path = f"{bot.PATH}/templates"
        self.bot = bot

        task = self.bot.loop.create_task(self.db_open())
        self.bot.loop.run_until_complete(task)

        self.templates = {
            f.split('.')[0]: cv2.imread(f'{template_path}/{f}') for f in os.listdir(template_path)
        }

    async def db_close(self):

        await self.bot.DB.close()

    async def db_open(self):
        DB = os.path.join(self.bot.PATH, "dsbot.db")

        self.bot.DB = await aiosqlite.connect(DB)
        await self.bot.DB.execute("PRAGMA read_uncommitted = true")

    def cog_unload(self):

        task = self.bot.loop.create_task(self.db_close())
        self.bot.loop.run_until_complete(task)

    def match_template(self, image):

        result = 0
        ring = False
        for name, t in self.templates.items():
            res = cv2.matchTemplate(image, t, cv2.TM_CCORR_NORMED)
            if np.amax(res) >= 0.99:
                result = int(translate[name])
                ring = True if result == 115 else ring
                break
        return result, ring

    def make_map(self, image):
        maze = deepcopy(default_maze)

        source = [(260, 150, 61), (250, 187, 66),
                  (237, 225, 72), (221, 276, 80),
                  (205, 335, 89), (180, 410, 101),
                  (147, 505, 117)]

        x_checks = [
            [(250, 200), (314, 200), (379, 200), (441, 200), (508, 200)],
            [(236, 235), (309, 235), (378, 235), (447, 235), (516, 235)],
            [(228, 278), (298, 278), (378, 278), (453, 278), (528, 278)],
            [(199, 330), (285, 330), (371, 330), (457, 330), (543, 330)],
            [(180, 395), (275, 395), (370, 395), (460, 395), (560, 395)],
            [(148, 480), (256, 480), (364, 480), (480, 480), (585, 480)],
        ]

        y_checks = [
            [(308, 175), (367, 175), (426, 175), (485, 175)],
            [(297, 211), (363, 211), (429, 211), (495, 175)],
            [(289, 247), (360, 247), (431, 247), (502, 247)],
            [(278, 286), (356, 286), (434, 286), (512, 286)],
            [(263, 348), (351, 348), (439, 348), (527, 348)],
            [(245, 420), (344, 420), (444, 420), (544, 420)],
            [(223, 500), (337, 500), (451, 500), (565, 500)]
        ]

        for i, k in zip(range(2, 13, 2), x_checks):
            for j, coords in zip(range(1, 10, 2), k):
                maze[i][j] = int(sum(image[coords[1], coords[0]]) == 252)

        for i, k in zip(range(1, 15, 2), y_checks):
            for j, coords in zip(range(2, 11, 2), k):
                maze[i][j] = int(sum(image[coords[1], coords[0]]) == 252)

        ring = False
        for i, (x, y, diff) in zip(range(1, 14, 2), source):
            for j in range(1, 10, 2):
                maze[i][j], r = self.match_template(image[y: y + 35, x: x + 35])  # type: ignore
                ring = True if r else ring
                x += diff

        ori = 2 if sum(image[573, 389]) == 523 else 3
        return maze, ori, ring

    async def fetch_image(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                b = np.frombuffer(await resp.read(), dtype=np.uint8)
                image = cv2.imdecode(b, cv2.IMREAD_COLOR)
                return image

    def game_setup(self, maze, base_ori, stats, total, ring):
        return game(maze, base_ori, stats, total, ring)

    @commands.command(name='solveurl')
    @commands.is_owner()
    async def _solveurl(self, ctx, *, url: str):
        image = await self.fetch_image(url)

        def predicate(reaction, user):
            return str(reaction.emoji) in ('✅', '❌') and user == ctx.author

        maze, base_ori, ring = self.make_map(image)
        embed = discord.Embed(color=discord.Color.greyple())

        if ring:
            embed.description = "💍 | Ring Found, Try Solving with the Ring ?"
            m = await ctx.reply(embed=embed)
            await m.add_reaction('✅')
            await m.add_reaction('❌')

            try:
                reaction, user = await ctx.bot.wait_for('reaction_add', timeout=20, check=predicate)

                if str(reaction.emoji) == '❌':
                    ring = False

                embed.description = "Solving.."

                await m.edit(embed=embed)

            except asyncio.TimeoutError:
                return
            finally:
                with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                    await m.clear_reactions()
        else:
            embed.description = "Solving.."
            m = await ctx.reply(embed=embed)
            ring = False

        stats, total = [100, 50, 50, 75], 0
        partial = functools.partial(self.game_setup, maze, base_ori, stats, total, ring)
        affection, solution = await self.bot.loop.run_in_executor(None, partial)

        if affection != -1:
            cursor = await self.bot.DB.execute("SELECT emoji FROM users WHERE user_id = ?", (ctx.author.id,))
            val = await cursor.fetchone()
            if val and val[0]:
                path = ' '.join(EMOJIS[x] for x in solution if x != -1)
            else:
                path = f"`{', '.join(translate[x] for x in solution if x != -1)}`"

            await m.edit(content=f"`{affection} AP` | {path}", embed=None)
            await cursor.close()
        else:
            if ring:
                await m.edit(content="No Solution with Ring Found :(", embed=None)
            else:
                await m.edit(content="No Solution Found :(", embed=None)

    @commands.command(name='emoji')
    async def _emoji(self, ctx):

        await self.bot.DB.execute("""INSERT INTO users (user_id, emoji) VALUES (?, ?)
                                     ON CONFLICT(user_id) DO UPDATE SET emoji = NOT emoji""", (ctx.author.id, True))
        await self.bot.DB.commit()
        cursor = await self.bot.DB.execute("SELECT emoji FROM users WHERE user_id = ?", (ctx.author.id,))
        val = (await cursor.fetchone())[0]
        embed = discord.Embed(color=0x36393f)
        if val:
            embed.description = r"\✔️ | Path will be shown as Emoji."
        else:
            embed.description = r"\📝 | Path will be shown as Text."

        await ctx.reply(embed=embed)

    @commands.command(name='solve')
    @commands.max_concurrency(1, per=BucketType.user, wait=False)
    async def _solve(self, ctx):
        await ctx.send("Please do **`k!vi`**")

        def predicate(message):
            if message.author.id == 646937666251915264 and message.embeds and message.channel.id == ctx.channel.id:
                embed = message.embeds[0]
                return embed.title == "Date Minigame" and f"Visitor · <@{ctx.author.id}>" in embed.description

        try:
            message = await self.bot.wait_for('message', timeout=30, check=predicate)
        except asyncio.TimeoutError:
            return

        url = message.embeds[0].image.url
        image = await self.fetch_image(url)

        def pred(reaction, user):
            return str(reaction.emoji) in ('✅', '❌') and user == ctx.author

        maze, base_ori, ring = self.make_map(image)
        embed = discord.Embed(color=discord.Color.greyple())

        if ring:
            embed.description = "💍 | Ring Found, Try Solving with the Ring ?"
            m = await ctx.reply(embed=embed)
            await m.add_reaction('✅')
            await m.add_reaction('❌')

            try:
                reaction, _ = await ctx.bot.wait_for('reaction_add', timeout=20, check=pred)

                if str(reaction.emoji) == '❌':
                    ring = False

                embed.description = "Solving.."
                await m.edit(embed=embed)
            except asyncio.TimeoutError:
                return
            finally:
                with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                    await m.clear_reactions()
        else:
            embed.description = "Solving.."
            m = await ctx.reply(embed=embed)
            ring = False

        stats, total = [100, 50, 50, 75], 0
        partial = functools.partial(self.game_setup, maze, base_ori, stats, total, ring)
        affection, solution = await self.bot.loop.run_in_executor(None, partial)

        if affection != -1:
            cursor = await self.bot.DB.execute("SELECT emoji FROM users WHERE user_id = ?", (ctx.author.id,))
            val = (await cursor.fetchone())
            if val and val[0]:
                path = ' '.join(EMOJIS[x] for x in solution if x != -1)
            else:
                path = f"`{', '.join(translate[x] for x in solution if x != -1)}`"
            await m.edit(content=f"`{affection} AP` | {path}", embed=None)
            await cursor.close()
        else:
            if ring:
                await m.edit(content="No Solution with Ring Found :(", embed=None)
            else:
                await m.edit(content="No Solution Found :(", embed=None)


    @_solve.error
    async def solve_error(self, ctx, error):
        if isinstance(error, commands.MaxConcurrencyReached):
            await ctx.send(f"{ctx.author.mention}, a previous command is waiting for `k!vi`.")

    @commands.command(name='help')
    async def _help(self, ctx):

        embed = discord.Embed(title='**__Help__**', color=0x36393f)

        content_a = r"""> \✔️ Best AR / AP Path.
        > \✔️ Option to take the Ring path (or go for just AP).
        """
        content_b = r"""> \❌ Trying to solve maps when the car isn't at starting position.
        > \❌ Trying to get a better path by running the command again.
        > \❌ Trying to solve another person's map using an image.
        """
        content_c = r"""> \❕ Bot is not public / invite-able currently (if ever).
        > \❕ Bot is free to use as of now.
        > \❕ To use it, simply do `a!solve` followed by `k!vi` with **date map** open.
        > \❕ Toggle Emoji Path by using `a!emoji` command.
        """
        embed.add_field(name=r"\🔹 What works", value=content_a, inline=False)
        embed.add_field(name=r"\🔸 What doesn't work", value=content_b, inline=False)
        embed.add_field(name=r"\📙 Additionally", value=content_c, inline=False)
        embed.set_footer(text="For bug reports, refer to: anu#1337")
        await ctx.send(embed=embed)

def setup(bot):
    bot.add_cog(DateSolver(bot))
