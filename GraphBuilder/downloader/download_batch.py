import copy
import os
import json
import re
import cv2
import numpy as np
import argparse
import sys
from image_downloading import download_single_tile

# ------------------------------------------------------------------
# Paths & defaults
# ------------------------------------------------------------------

file_dir = os.path.dirname(__file__)
default_prefs = {
    "url": "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "tile_size": 256,
    "output_tile_size": 2048,
    "channels": 3,
    "zoom": 17,
    "target_resolution_mpp": 0.5,
    "headers": {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/99.0.4844.82 Safari/537.36"
        )
    }
}

EARTH_RES_AT_EQUATOR = 156543.03392

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def latlon_to_tile(lat, lon, zoom):
    scale = 1 << zoom
    siny = np.sin(lat * np.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360.0)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return int(x), int(y)

def meters_per_pixel(lat, zoom):
    return (EARTH_RES_AT_EQUATOR * np.cos(np.deg2rad(lat))) / (2 ** zoom)

def build_composite_tile(x_start, y_start, zoom, tiles_per_side, base_tile_size, output_tile_size, prefs):
    block_dim = tiles_per_side * base_tile_size
    channels = prefs["channels"]
    canvas = np.zeros((block_dim, block_dim, channels), np.uint8)
    downloaded = False

    for rel_y in range(tiles_per_side):
        for rel_x in range(tiles_per_side):
            tile = download_single_tile(
                x=x_start + rel_x,
                y=y_start + rel_y,
                z=zoom,
                url=prefs["url"],
                headers=prefs["headers"],
                channels=channels,
            )

            if tile is None:
                continue

            downloaded = True
            tile_h, tile_w = tile.shape[:2]
            px_y = rel_y * base_tile_size
            px_x = rel_x * base_tile_size
            canvas[px_y:px_y + tile_h, px_x:px_x + tile_w] = tile

    if not downloaded:
        return None

    if canvas.shape[0] < output_tile_size or canvas.shape[1] < output_tile_size:
        padded = np.zeros((output_tile_size, output_tile_size, channels), np.uint8)
        padded[: canvas.shape[0], : canvas.shape[1]] = canvas[:output_tile_size, :output_tile_size]
        return padded

    return canvas[:output_tile_size, :output_tile_size]

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------

def download_for_bbox(bbox, output_dir, zoom=17):
    """
    Download tiles for a bbox [north, south, east, west]
    """
    north, south, east, west = bbox
    lat1, lon1 = north, west  # Top-Left
    lat2, lon2 = south, east  # Bottom-Right
    
    prefs = copy.deepcopy(default_prefs)
    os.makedirs(output_dir, exist_ok=True)

    resolution_mpp = meters_per_pixel((lat1+lat2)/2, zoom)

    tl_x, tl_y = latlon_to_tile(lat1, lon1, zoom)
    br_x, br_y = latlon_to_tile(lat2, lon2, zoom)
    
    # Ensure correct order
    tl_x, br_x = min(tl_x, br_x), max(tl_x, br_x)
    tl_y, br_y = min(tl_y, br_y), max(tl_y, br_y)

    output_tile_size = int(prefs.get("output_tile_size", 2048))
    base_tile_size = int(prefs["tile_size"])
    tiles_per_side = int(np.ceil(output_tile_size / base_tile_size))
    
    print(f"Downloading grid: x[{tl_x}-{br_x}], y[{tl_y}-{br_y}]")
    print(f"Total composites: {((br_x-tl_x)//tiles_per_side + 1) * ((br_y-tl_y)//tiles_per_side + 1)}")

    metadata = {
        "zoom": zoom,
        "base_tile_size": base_tile_size,
        "output_tile_size": output_tile_size,
        "tiles_per_side": tiles_per_side,
        "resolution_mpp": resolution_mpp,
        "bbox": [north, south, east, west],
        "tiles": []
    }

    count = 0
    for x in range(tl_x, br_x + 1, tiles_per_side):
        for y in range(tl_y, br_y + 1, tiles_per_side):
            print(f"Downloading tile x={x}, y={y}...")
            composite = build_composite_tile(
                x_start=x, y_start=y, zoom=zoom,
                tiles_per_side=tiles_per_side,
                base_tile_size=base_tile_size,
                output_tile_size=output_tile_size,
                prefs=prefs
            )

            if composite is None:
                print(f"  Failed/Empty tile x={x}, y={y}")
                continue

            fname = f"z{zoom}_x{x}_y{y}_composite.png"
            cv2.imwrite(os.path.join(output_dir, fname), composite)

            metadata["tiles"].append({
                "file": fname,
                "x_range": [x, min(x + tiles_per_side - 1, br_x)],
                "y_range": [y, min(y + tiles_per_side - 1, br_y)],
                "z": zoom
            })
            count += 1

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Downloaded {count} images to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True, 
                        help="North South East West")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zoom", type=int, default=17)
    
    args = parser.parse_args()
    
    download_for_bbox(args.bbox, args.output_dir, args.zoom)

if __name__ == "__main__":
    main()
