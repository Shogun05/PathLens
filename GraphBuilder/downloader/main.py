import copy
import os
import json
import re
import cv2
import numpy as np

from image_downloading import download_single_tile

# ------------------------------------------------------------------
# Paths & defaults
# ------------------------------------------------------------------

file_dir = os.path.dirname(__file__)
prefs_path = os.path.join(file_dir, 'preferences.json')

default_prefs = {
    "url": "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "tile_size": 256,
    "output_tile_size": 2048,
    "channels": 3,
    "zoom": 17,
    "target_resolution_mpp": 0.5,
    "dir": os.path.join(file_dir, "sat_images"),
    "headers": {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/99.0.4844.82 Safari/537.36"
        )
    }
}

coord_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?")
EARTH_RES_AT_EQUATOR = 156543.03392  # meters per pixel @ zoom 0


def _safe_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def take_input():
    print("Enter bounding box coordinates")
    print('Format: "lat, lon"\n')

    tl = input("Top-left corner: ")
    br = input("Bottom-right corner: ")
    zoom = input("Zoom level (press Enter to use default zoom 17): ")

    return tl, br, zoom


def parse_coordinates(raw: str):
    matches = coord_pattern.findall(raw)
    if len(matches) < 2:
        raise ValueError("Invalid coordinate format")
    return float(matches[0]), float(matches[1])


def latlon_to_tile(lat, lon, zoom):
    scale = 1 << zoom
    siny = np.sin(lat * np.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)

    x = scale * (0.5 + lon / 360.0)
    y = scale * (
        0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi)
    )
    return int(x), int(y)


def meters_per_pixel(lat, zoom):
    return (EARTH_RES_AT_EQUATOR * np.cos(np.deg2rad(lat))) / (2 ** zoom)


def zoom_for_resolution(lat, target_mpp):
    if target_mpp is None or target_mpp <= 0:
        return None
    zoom_float = np.log2(
        (EARTH_RES_AT_EQUATOR * np.cos(np.deg2rad(lat))) / target_mpp
    )
    zoom = int(np.clip(np.round(zoom_float), 0, 23))
    return zoom


def build_composite_tile(
    x_start,
    y_start,
    zoom,
    tiles_per_side,
    base_tile_size,
    output_tile_size,
    prefs,
):
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
# Main logic (Stage 0)
# ------------------------------------------------------------------

def run():
    # Load or create preferences
    if not os.path.isfile(prefs_path):
        with open(prefs_path, "w") as f:
            json.dump(default_prefs, f, indent=2)
        print("preferences.json created. Re-run the script.")
        return

    with open(prefs_path, "r") as f:
        loaded_prefs = json.load(f)

    prefs = copy.deepcopy(default_prefs)
    prefs.update({k: v for k, v in loaded_prefs.items() if k != "headers"})
    prefs["headers"].update(loaded_prefs.get("headers", {}))

    tiles_dir = prefs["dir"]
    os.makedirs(tiles_dir, exist_ok=True)

    # ---- USER INPUT ----
    tl_raw, br_raw, zoom_raw = take_input()

    lat1, lon1 = parse_coordinates(tl_raw)
    lat2, lon2 = parse_coordinates(br_raw)
    center_lat = (lat1 + lat2) / 2.0
    default_zoom = _safe_int(prefs.get("zoom")) or default_prefs["zoom"]

    zoom_input = zoom_raw.strip()
    if zoom_input:
        zoom = int(zoom_input)
    else:
        zoom = default_zoom

    resolution_mpp = meters_per_pixel(center_lat, zoom)

    # ---- TILE RANGE ----
    tl_x, tl_y = latlon_to_tile(lat1, lon1, zoom)
    br_x, br_y = latlon_to_tile(lat2, lon2, zoom)

    print(f"\nDownloading tiles:")
    print(f"x: {tl_x} → {br_x}")
    print(f"y: {tl_y} → {br_y}\n")

    output_tile_size = int(prefs.get("output_tile_size", 2048))
    base_tile_size = int(prefs["tile_size"])
    tiles_per_side = int(np.ceil(output_tile_size / base_tile_size))
    actual_resolution = resolution_mpp

    print(f"Composite size      : {output_tile_size}px")
    print(f"Tiles per composite: {tiles_per_side}x{tiles_per_side}")
    print(f"Resolution estimate: {actual_resolution:.2f} m/px at zoom {zoom}\n")

    metadata = {
        "zoom": zoom,
        "base_tile_size": base_tile_size,
        "output_tile_size": output_tile_size,
        "tiles_per_side": tiles_per_side,
        "target_resolution_mpp": actual_resolution,
        "actual_resolution_mpp": actual_resolution,
        "bbox": {
            "top_left": [lat1, lon1],
            "bottom_right": [lat2, lon2]
        },
        "tiles": []
    }

    count = 0

    for x in range(tl_x, br_x + 1, tiles_per_side):
        for y in range(tl_y, br_y + 1, tiles_per_side):
            composite = build_composite_tile(
                x_start=x,
                y_start=y,
                zoom=zoom,
                tiles_per_side=tiles_per_side,
                base_tile_size=base_tile_size,
                output_tile_size=output_tile_size,
                prefs=prefs,
            )

            if composite is None:
                continue

            fname = f"z{zoom}_x{x}_y{y}_composite.png"
            cv2.imwrite(os.path.join(tiles_dir, fname), composite)

            metadata["tiles"].append({
                "file": fname,
                "x_range": [x, min(x + tiles_per_side - 1, br_x)],
                "y_range": [y, min(y + tiles_per_side - 1, br_y)],
                "z": zoom
            })

            count += 1

    with open(os.path.join(file_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Download complete")
    print(f"Images saved  : {count}")
    print(f"Tiles folder  : {tiles_dir}")
    print(f"Metadata     : metadata.json")


# ------------------------------------------------------------------

if __name__ == "__main__":
    run()
