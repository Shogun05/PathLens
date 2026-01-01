import cv2
import requests
import numpy as np
import threading

# ---------------------------------------------------------------------
# Low-level tile download
# ---------------------------------------------------------------------

def download_tile(url, headers, channels):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None

    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(arr, 1 if channels == 3 else -1)


def download_single_tile(x, y, z, url, headers, channels):
    """
    Download ONE tile by x, y, z.
    Used for Stage 0 (multi-tile city download).
    """
    tile_url = url.format(x=x, y=y, z=z)
    return download_tile(tile_url, headers, channels)

# ---------------------------------------------------------------------
# Mercator projection helpers
# ---------------------------------------------------------------------
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates

def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)

    x = scale * (0.5 + lon / 360.0)
    y = scale * (
        0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi)
    )
    return x, y


def latlon_to_tile(lat, lon, zoom):
    scale = 1 << zoom
    x, y = project_with_scale(lat, lon, scale)
    return int(x), int(y)

# ---------------------------------------------------------------------
# EXISTING FUNCTION (unchanged behavior)
# ---------------------------------------------------------------------

def download_image(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    zoom: int,
    url: str,
    headers: dict,
    tile_size: int = 256,
    channels: int = 3
) -> np.ndarray:
    """
    Downloads a rectangular map region as ONE stitched image.
    (Kept for backward compatibility.)
    """

    scale = 1 << zoom

    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y

    img = np.zeros((img_h, img_w, channels), np.uint8)

    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile = download_tile(
                url.format(x=tile_x, y=tile_y, z=zoom),
                headers,
                channels
            )
            if tile is None:
                continue

            tl_rel_x = tile_x * tile_size - tl_pixel_x
            tl_rel_y = tile_y * tile_size - tl_pixel_y

            br_rel_x = tl_rel_x + tile_size
            br_rel_y = tl_rel_y + tile_size

            img_x_l = max(0, tl_rel_x)
            img_x_r = min(img_w, br_rel_x)
            img_y_l = max(0, tl_rel_y)
            img_y_r = min(img_h, br_rel_y)

            cr_x_l = max(0, -tl_rel_x)
            cr_y_l = max(0, -tl_rel_y)

            cr_x_r = cr_x_l + (img_x_r - img_x_l)
            cr_y_r = cr_y_l + (img_y_r - img_y_l)

            img[img_y_l:img_y_r, img_x_l:img_x_r] = \
                tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]

    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        t = threading.Thread(target=build_row, args=(tile_y,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return img

# ---------------------------------------------------------------------
# Utility (unchanged)
# ---------------------------------------------------------------------

def image_size(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    zoom: int,
    tile_size: int = 256
):
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y
