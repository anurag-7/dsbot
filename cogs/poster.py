import aiohttp
import discord
from discord.ext import commands, tasks
from dataclasses import dataclass
import time
import asyncio
import contextlib
import aiosqlite
import os

@dataclass
class ChannelCycle:
    next_post: float
    server_id: int
    channel_id: int
    cycle_time: int
    webhook_url: str

class Poster(commands.Cog):
    def __init__(self, bot) -> None:
        self.bot = bot
        self.db = bot.DB
        self.load_finished = False
        self.load_channels.start()
        self.cycler.start()

    async def db_close(self):
        await self.bot.DB.close()

    @tasks.loop(count=1)
    async def load_channels(self):
        await asyncio.sleep(5)
        current_time = time.time()
        cur = await self.db.execute('''
                              SELECT * from cycles;
                              ''')
        res = await cur.fetchall()
        self.channels = {
            c[1]: ChannelCycle(current_time, *c) for c in res
        }
        self.load_finished = True

    @tasks.loop(seconds=15)
    async def cycler(self):

        while not self.load_finished:
            await asyncio.sleep(5)

        async with aiohttp.ClientSession() as session:
            current_time = time.time()
            for channel_id, channel in self.channels.items():
                if channel.next_post < current_time:
                    cur = await self.db.execute('''
                                            SELECT * FROM posts WHERE channel_id = ?
                                            ORDER BY posted_on ASC''',
                                            (channel_id,))
                    row = await cur.fetchone()
                    if row is None:
                        continue
                    _, desc, posted_on = row

                    with contextlib.suppress(discord.Forbidden, discord.HTTPException, discord.NotFound):
                        text_channel = await self.bot.fetch_channel(channel_id)
                        message = await text_channel.fetch_message(posted_on)
                        await message.delete()

                    webhook = discord.Webhook.from_url(channel.webhook_url, adapter=discord.AsyncWebhookAdapter(session))
                    message = await webhook.send(desc, wait=True)
                    await self.db.execute('UPDATE posts SET posted_on = ? WHERE posted_on = ?',
                                          (message.id, posted_on)
                    )
                    await self.db.commit()
                    channel.next_post += channel.cycle_time

    @commands.command("addchannel")
    async def _addchannel(self, ctx, channel: discord.TextChannel, webhook_url: str):
        await self.db.execute('''
                              INSERT INTO cycles (server_id, channel_id, cycle_time, webhook_url)
                              VALUES (?, ?, ?, ?)
                              ''', (ctx.guild.id, channel.id, 1800, webhook_url))
        await self.db.commit()
        self.channels[channel.id] = ChannelCycle(
            time.time(),
            ctx.guild.id,
            channel.id,
            1800,
            webhook_url
        )

        await ctx.send(f"Channel {channel.mention} added to be cycled.")

    @commands.command("removechannel")
    async def _removechannel(self, ctx, channel: discord.TextChannel):
        await self.db.execute('''
                               DELETE FROM cycles WHERE channel_id = ?
                               ''', (channel.id,))
        await self.db.commit()
        if channel.id in self.channels:
            del self.channels[channel.id]

        await ctx.send(f"Channel {channel.mention} successfully removed from mention.")

    @commands.command("removepost")
    async def _removepost(self, ctx, message_id):
        cur = await self.db.execute('''
                                    SELECT * FROM posts where posted_on = ?
                                    ''', (message_id,))
        res = await cur.fetchone()

        if not res:
            await ctx.send("Channel Post not found")
            return

        channel_id, *_ = res
        await self.db.execute('''
                               DELETE FROM posts WHERE posted_on = ?
                               ''', (message_id,))
        await self.db.commit()

        with contextlib.suppress(discord.NotFound, discord.Forbidden, discord.HTTPException):
            channel = await self.bot.fetch_channel(channel_id)
            message = await channel.fetch_message(message_id)
            await message.delete()
            await ctx.send("Successfully removed Channel Post")

    @commands.command("addpost")
    async def _addpost(self, ctx, channel: discord.TextChannel, *, post: str):

        await self.db.execute('''
                              INSERT INTO posts (channel_id, posted_on, desc)
                              VALUES (?, ?, ?)
                              ''', (channel.id, time.time() - 100000, post))

        await self.db.commit()
        if channel.id in self.channels:
            self.channels[channel.id].next_post = time.time()

        await ctx.channel.send(f"Successfully created the Post for channel {channel.mention}")

    @commands.command("channeltime")
    async def _channeltime(self, ctx, channel: discord.TextChannel, minutes: float):
        if not 0.5 <= minutes <= 1440:
            await ctx.send("Timer can only be set between 0.5 and 1440 minutes")
            return

        if channel.id not in self.channels:
            await ctx.send("Channel does not exist in database")
            return

        seconds = minutes * 60

        if self.channels[channel.id].cycle_time > seconds:
            self.channels[channel.id].next_post -= self.channels[channel.id].cycle_time - seconds

        self.channels[channel.id].cycle_time = int(seconds)

        await self.db.execute('''
                              UPDATE cycles SET cycle_time = ?
                              ''', (seconds,))
        await self.db.commit()

        await ctx.send(f"Updated Channel Timer to every `{minutes}` minutes.")

def setup(bot):
    bot.add_cog(Poster(bot))
