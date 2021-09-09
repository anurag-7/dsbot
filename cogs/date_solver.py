import asyncio
import math
import os
from copy import deepcopy

import cv2
import numpy as np
from discord.ext import commands
import aiohttp
import functools

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

places = {
    "home": {},
    "shopping": {},
    "gas": {"gas": 100},
    "flower": {"mood": 100},
    "taco": {"food": 60},
    "ballroom": {"mood": 100, "drink": -15, "food": -15},
    "coffee": {"drink": 60},
    "juice": {"drink": 60},
    "theater": {"mood": 60},
    "restaurant": {"food": 60},
    "nightclub": {"drink": 40, "mood": 40},
    "fair": {"mood": 40, "food": 20, "drink": 20},
    "sandwhich": {"food": 40, "drink": 20}
}

moveset = {
    'UP': [(-2, 0, 'UP'), (-1, -1, 'LEFT'), (-1, 1, 'RIGHT')],
    'DOWN': [(2, 0, 'DOWN'), (1, 1, 'RIGHT'), (1, -1, 'LEFT')],
    'RIGHT': [(-1, 1, 'UP'), (0, 2, 'RIGHT'), (1, 1, 'DOWN')],
    'LEFT': [(-1, -1, 'UP'), (0, -2, 'LEFT'), (1, -1, 'DOWN')]
}

interaction_set = {
    'UP': [(0, -1), (0, 1)],
    'DOWN': [(0, -1), (0, 1)],
    'RIGHT': [(1, 0), (-1, 0)],
    'LEFT': [(1, 0), (-1, 0)]
}

def pretty_maze(maze):
    length = len(maze[0])
    frmt = "{:>12}" * length
    for l in maze:
        print(frmt.format(*l), '\n')


class DateSolver(commands.Cog):

    def __init__(self, bot):
        template_path = f"{bot.PATH}/templates"
        self.bot = bot
        self.templates = {
            f.split('.')[0]: cv2.imread(f'{template_path}/{f}') for f in os.listdir(template_path)
        }

    def match_template(self, image):

        result = ''

        for name, t in self.templates.items():
            res = cv2.matchTemplate(image, t, cv2.TM_CCORR_NORMED)
            if np.amax(res) >= 0.99:
                result = name
                break
        return result


    def apply_int(self, int_type, **kwargs):
        res = {k: min(100, v + places[int_type].get(k, 0)) for k, v in kwargs.items()}
        return res.values()

    def make_map(self, image):
        maze = deepcopy(default_maze)

        dim = (35, 35)
        source = [(260, 150, 61), (250, 187, 66),
                (237, 225, 72), (221, 276, 80),
                (205, 335, 89), (180, 410, 101),
                (147, 505, 117)]

        x_checks = [
            [(250, 200), (314, 200), (379, 200), (441, 200), (508, 200)],
            [(236, 235), (309, 235), (378, 235), (447, 235), (516, 235)],
            [(228, 278), (303, 278), (378, 278), (453, 278), (528, 278)],
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

        for i, (x, y, diff) in zip(range(1, 14, 2), source):
            for j in range(1, 10, 2):
                maze[i][j] = self.match_template(image[y: y + dim[1], x: x + dim[0]])  # type: ignore
                x += diff

        ori = 'LEFT' if sum(image[573, 389]) == 523 else 'RIGHT'
        return maze, ori

    async def fetch_image(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                b = np.frombuffer(await resp.read(), dtype=np.uint8)
                image = cv2.imdecode(b, cv2.IMREAD_COLOR)
                return image

    def game(self, image):
        maze, base_ori = self.make_map(image)
        best, rpath = -1, None

        def solve(x, y, ori, gas=100, food=50, drink=50, mood=75, path=[], shopping=False, total=0, bl=dict()):
            global places, moveset, interaction_set
            nonlocal best, rpath

            for stat in (food, mood, gas, drink):
                if stat <= 0:
                    return

            if total == 25:

                ap = math.ceil((food + drink + mood) / 6) + shopping * 30

                if ap > best:
                    best = ap
                    rpath = path

                return

            interactions = [
                (y + py, x + px) for py, px in interaction_set[ori]
                if 0 <= y + py < 15 and 0 <= x + px < 11
            ]

            moves = [
                (y + my, x + mx, mori) for my, mx, mori in moveset[ori]
                if 0 <= y + my < 15 and 0 <= x + mx < 11 and maze[y + my][x + mx]
            ]

            for iy, ix in interactions:

                interaction = maze[iy][ix]
                if interaction not in places or bl.get((iy, ix), -2) > total - 1:
                    continue

                new_bl = {}
                for k, v in bl.items():
                    new_bl[k] = v
                if interaction == 'shopping':
                    new_bl[(iy, ix)] = total+100
                    solve(x, y, ori, gas=gas, food=food - 4, drink=drink - 6, mood=mood - 8, path=path + [interaction.upper()],
                        shopping=True, total=total + 1, bl=new_bl)

                elif interaction == 'flower':
                    new_bl[(iy, ix)] = total+100
                    g, d, f, m = [min(100, v + places[interaction].get(k, 0)) for k, v in (('gas', gas), ('drink', drink), ('food', food), ('mood', mood))]

                    solve(x, y, ori, gas=g, food=f - 4, drink=d - 6, mood=m - 8, path=path + [interaction.upper()],
                        shopping=shopping, total=total + 1, bl=new_bl)

                elif interaction == 'home':

                    rem = (25 - len(path)) * 4
                    ap =  math.ceil((food + drink + mood) / 6 * (1 - rem/100)) + shopping * 30
                    if ap > best:
                        best = ap
                        rpath = path + ["HOME"]

                else:
                    g, f, m, d = self.apply_int(interaction, gas=gas, food=food, mood=mood, drink=drink)
                    new_bl[(iy, ix)] = total+10
                    solve(x, y, ori, gas=g, food=f - 4, drink=d - 6, mood=m - 8, path=path + [interaction.upper()],
                        shopping=shopping, total=total + 1, bl=new_bl)

            for my, mx, mori in moves:
                solve(mx, my, mori, gas=gas - 10, food=food - 4, drink=drink - 6, mood=mood - 8, path=path + [mori],
                    shopping=shopping, total=total + 1, bl=bl)

        solve(5, 14, base_ori)

        return best, rpath

    @commands.command(name='solveurl')
    @commands.is_owner()
    async def _solveurl(self, ctx, *, url: str):
        image = await self.fetch_image(url)
        partial = functools.partial(self.game, image)
        m = await ctx.send("Solving...")
        affection, solution = await self.bot.loop.run_in_executor(None, partial)

        if solution:
            await m.edit(content=f"{ctx.author.mention} | `{affection} AP` | `{', '.join(solution)}`")
        else:
            await m.edit(content=f"No Solution Found :(")

    @commands.command(name='solve')
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
        partial = functools.partial(self.game, image)
        m = await ctx.send("Solving...")
        affection, solution = await self.bot.loop.run_in_executor(None, partial)

        if solution:
            await m.edit(content=f"{ctx.author.mention} | `{affection} AP` | `{', '.join(solution)}`")
        else:
            await m.edit(content=f"No Solution Found :(")

def setup(bot):
    bot.add_cog(DateSolver(bot))
