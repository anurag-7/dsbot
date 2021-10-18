import functools
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import aiohttp
import cv2
import discord
import numpy as np
from discord.ext import commands

TESSEROCR_MODE: bool = True

try:
    import tesserocr  # type: ignore
except ModuleNotFoundError:
    import pytesseract
    TESSEROCR_MODE = False


POINTS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    1: (np.array([(20, 27), (190, 27), (214, 49), (214, 67), (207, 74), (29, 74), (20, 67)]),
        np.array([(20, 277), (215, 277), (214, 322), (205, 330), (20, 330)])),

    2: (np.array([(28, 24), (202, 24), (224, 50), (224, 83), (197, 83), (180, 71), (78, 71), (57, 83), (28, 83)]),
        np.array([(27, 278), (58, 277), (78, 262), (177, 262), (195, 277), (226, 277), (226, 326), (218, 332), (42, 332), (27, 326)])),

    3: (np.array([(38, 42), (204, 42), (222, 58), (220, 79), (204, 79), (194, 88), (61, 88), (54, 78), (38, 78)]),
        np.array([(35, 293), (75, 293), (88, 283), (166, 283), (181, 293), (221, 293), (221, 339), (56, 341), (35, 327)]))
}


@dataclass(frozen=True)
class KarutaDrop:
    name: str
    series: str
    ed: int

def get_cv_image(image, color='BGR'):
    bytes_per_pixel = image.shape[2] if len(image.shape) == 3 else 1
    height, width = image.shape[:2]
    bytes_per_line = bytes_per_pixel * width

    if bytes_per_pixel != 1 and color != 'RGB':
        image = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color}2RGB'))
    elif bytes_per_pixel == 1 and image.dtype == bool:
        image = np.packbits(image, axis=1)
        bytes_per_line = image.shape[1]
        width = bytes_per_line * 8
        bytes_per_pixel = 0

    return (width, height, bytes_per_pixel, bytes_per_line)

def get_cuts(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    poly = cv2.boundingRect(points)
    x, y, w, h = poly

    cropped = image[y:y + h, x:x + w]

    points = points - points.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)

    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    result = cv2.bitwise_and(cropped, cropped, mask=mask)

    background = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(background, background, mask=mask)

    return result + background


def segment_and_classify(image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    y, x, *_ = image.shape
    amount = 3 if x < 850 else 4

    cropped: Generator[np.ndarray, None, None] = (image[0: y, n: n + x // amount] for n in range(0, x, x // amount + 1))
    cuts = []

    for img in cropped:
        yi, xi = img[:, :, 3].nonzero()

        min_x, max_x = np.min(xi), np.max(xi)
        min_y, max_y = np.min(yi), np.max(yi)

        _, cut = cv2.threshold(img[min_y: max_y, min_x: max_x], 80, 255, cv2.THRESH_BINARY)

        if max_y > 395:
            ed = 3
        elif max_x > 263:
            ed = 2
        else:
            ed = 1

        top, bottom = POINTS[ed]
        name = get_cuts(cut, top)
        series = get_cuts(cut, bottom)
        cuts.append((name, series, ed))

    return cuts

def get_drop_text(image: np.ndarray, api: Optional["tesserocr.PyTessBaseAPI"] = None) -> List[KarutaDrop]:

    cuts = segment_and_classify(image)
    card_drops = []

    if api is None:
        for name, series, ed in cuts:
            t_name = pytesseract.image_to_string(name).replace('\n', ' ')
            t_series = pytesseract.image_to_string(series).replace('\n', ' ')

            card_drops.append(KarutaDrop(t_name, t_series, ed))

    else:
        for name, series, ed in cuts:
            api.SetImageBytes(name.tobytes(), *get_cv_image(name))
            t_name = api.GetUTF8Text().replace('\n', ' ')

            api.SetImageBytes(series.tobytes(), *get_cv_image(series))
            t_series = api.GetUTF8Text().replace('\n', ' ')

            card_drops.append(KarutaDrop(t_name, t_series, ed))

        api.Clear()

    return card_drops


class CardDrops(commands.Cog):

    def __init__(self, bot) -> None:
        self.bot = bot

        if TESSEROCR_MODE:
            self.ocr_api = tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK,
                                                   oem=tesserocr.OEM.TESSERACT_ONLY)
        else:
            self.ocr_api = None

    def cog_unload(self):
        if self.ocr_api:
            self.ocr_api.End()

    async def fetch_image(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                buffer = np.frombuffer(await resp.read(), dtype=np.uint8)
                image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
                return image

    @commands.command(name='droptest')
    async def _droptest(self, ctx, url: str):
        image = await self.fetch_image(url)
        partial = functools.partial(get_drop_text, image, self.ocr_api)
        t = time.time()
        cards = await self.bot.loop.run_in_executor(None, partial)

        to_send = '\n'.join(f"{c.name} | {c.series} | ED{c.ed}" for c in cards)

        await ctx.send(f"```ocaml\n{to_send}\n\n> Time Taken: {time.time() - t:.3f}s```")

def setup(bot):
    bot.add_cog(CardDrops(bot))
