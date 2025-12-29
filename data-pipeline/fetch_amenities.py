#!/usr/bin/env python3
"""
Fetch amenity data for all areas in Bengaluru using Overpass API.
Automatically discovers all areas from OSM and saves results as JSON files.
"""
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


def get_bengaluru_areas() -> List[Dict]:
    """
    Fetch all neighborhoods, suburbs, and administrative areas within Bengaluru
    from the Overpass API. Uses separate queries to avoid timeout.
    
    Returns:
        List of dictionaries containing area information (name, type, id)
    """
    url = "https://overpass-api.de/api/interpreter"
    areas = []
    seen_names = set()
    
    # Split into multiple smaller queries to avoid timeout
    queries = [
        # Query 1: Suburbs and neighbourhoods (nodes)
        """
        [out:json][timeout:30];
        node["place"~"suburb|neighbourhood"](12.7,77.4,13.2,77.8);
        out tags;
        """,
        # Query 2: Administrative boundaries level 10
        """
        [out:json][timeout:30];
        relation["boundary"="administrative"]["admin_level"="10"](12.7,77.4,13.2,77.8);
        out tags;
        """,
    ]
    
    try:
        print("🔍 Fetching list of areas in Bengaluru from Overpass API...")
        
        for idx, query in enumerate(queries, 1):
            print(f"  Query {idx}/{len(queries)}...", end=" ")
            r = requests.post(url, data=query, timeout=45)
            r.raise_for_status()
            data = r.json()
            
            for element in data["elements"]:
                tags = element.get("tags", {})
                name = tags.get("name")
                
                if name and name not in seen_names:
                    areas.append({
                        "name": name,
                        "type": element["type"],
                        "id": element["id"],
                        "place_type": tags.get("place", tags.get("boundary", "unknown"))
                    })
                    seen_names.add(name)
            
            print(f"✓ Got {len(data['elements'])} items")
            time.sleep(1)  # Rate limiting between queries
        
        # Sort by name for consistent ordering
        areas.sort(key=lambda x: x["name"])
        
        print(f"✓ Found {len(areas)} unique areas in Bengaluru")
        return areas
    
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error fetching Bengaluru areas: {e}")
        return []
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return []


def osm_id_to_area_id(osm_id: int, osm_type: str) -> int:
    """
    Convert OSM ID to Overpass API area ID.
    
    Args:
        osm_id: OSM element ID
        osm_type: OSM element type (node, way, relation)
    
    Returns:
        Area ID for Overpass API
    """
    if osm_type == "relation":
        return 3600000000 + osm_id
    elif osm_type == "way":
        return 2400000000 + osm_id
    else:  # node
        return 3600000000 + osm_id


def get_amenities_for_area(area_info: Dict) -> List[Dict]:
    """
    Fetch all amenities within an area using Overpass API.
    Uses the area's OSM geometry to query amenities.
    
    Args:
        area_info: Dictionary with area information (name, type, id)
    
    Returns:
        List of amenity elements
    """
    area_name = area_info["name"]
    osm_id = area_info["id"]
    osm_type = area_info["type"]
    
    # Convert to area ID
    area_id = osm_id_to_area_id(osm_id, osm_type)
    
    # If it's a node (point), we need to get its coordinates first and search around it
    if osm_type == "node":
        # For nodes, use a two-step query: get node location, then query around it
        query = f"""
        [out:json][timeout:45];
        node({osm_id})->.center;
        (
          node["amenity"](around.center:1000);
          way["amenity"](around.center:1000);
          node["shop"](around.center:1000);
          way["shop"](around.center:1000);
          node["leisure"](around.center:1000);
          way["leisure"](around.center:1000);
        );
        out tags center;
        """
    else:
        query = f"""
        [out:json][timeout:45];
        (
          node["amenity"](area:{area_id});
          way["amenity"](area:{area_id});
          node["shop"](area:{area_id});
          way["shop"](area:{area_id});
          node["leisure"](area:{area_id});
          way["leisure"](area:{area_id});
        );
        out tags center;
        """
    
    url = "https://overpass-api.de/api/interpreter"
    
    # Retry logic with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.post(url, data=query, timeout=60)
            r.raise_for_status()
            elements = r.json()["elements"]
            print(f"✓ Found {len(elements)} amenities in {area_name}")
            return elements
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(f"⏱️  Timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ Timeout after {max_retries} attempts for {area_name}")
                return []
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s
                    print(f"⏱️  Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Rate limit exceeded for {area_name}")
                    return []
            elif e.response.status_code >= 500:  # Server errors
                if attempt < max_retries - 1:
                    wait_time = 10
                    print(f"⏱️  Server error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Server error for {area_name}: {e}")
                    return []
            else:
                print(f"❌ HTTP error for {area_name}: {e}")
                return []
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error for {area_name}: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error for {area_name}: {e}")
            return []
    
    return []


def save_amenities_with_metadata(area_name: str, area_data: Dict, cache_dir: Path):
    """
    Save amenity data with metadata to JSON file in cache directory.
    
    Args:
        area_name: Name of the area
        area_data: Dictionary containing area metadata and amenities
        cache_dir: Directory to save cache files
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from area name
    filename = area_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_") + ".json"
    filepath = cache_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(area_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved {area_data['amenity_count']} amenities to {filepath}")


def fetch_all_areas(areas: List[Dict], cache_dir: Path, batch_size: int = 5, batch_delay: float = 2.0, rest_delay: float = 10.0):
    """
    Fetch amenity data for all areas in batches to avoid rate limiting.
    
    Args:
        areas: List of area dictionaries with name, type, id
        cache_dir: Directory to save cache files
        batch_size: Number of areas to fetch before taking a rest (default: 5)
        batch_delay: Delay between requests within a batch in seconds (default: 2s)
        rest_delay: Delay after completing a batch in seconds (default: 10s)
    """
    results = {
        "successful": [],
        "failed": [],
        "skipped": []
    }
    
    total = len(areas)
    requests_in_batch = 0
    
    for idx, area_info in enumerate(areas, 1):
        area_name = area_info["name"]
        print(f"\n[{idx}/{total}] Processing: {area_name}")
        print("=" * 60)
        
        # Check if already cached
        filename = area_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_") + ".json"
        filepath = cache_dir / filename
        if filepath.exists():
            print(f"⏭️  Skipping {area_name} (already cached)")
            results["skipped"].append(area_name)
            continue
        
        print(f"✓ OSM ID: {area_info['id']} (type: {area_info['type']})")
        
        # Fetch amenities
        amenities = get_amenities_for_area(area_info)
        if amenities:
            # Add area metadata to the saved data
            area_data = {
                "area_name": area_name,
                "osm_id": area_info["id"],
                "osm_type": area_info["type"],
                "place_type": area_info.get("place_type", "unknown"),
                "amenity_count": len(amenities),
                "amenities": amenities
            }
            save_amenities_with_metadata(area_name, area_data, cache_dir)
            results["successful"].append(area_name)
            requests_in_batch += 1
        else:
            print(f"⚠️  No amenities found for {area_name}")
            results["failed"].append(area_name)
        
        # Batch-based rate limiting
        if idx < total:
            if requests_in_batch >= batch_size:
                print(f"🛌 Batch of {batch_size} completed. Resting for {rest_delay}s...")
                time.sleep(rest_delay)
                requests_in_batch = 0
            else:
                print(f"⏳ Waiting {batch_delay}s before next request...")
                time.sleep(batch_delay)
    
    return results


def print_summary(results: Dict):
    """Print summary of fetching results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Successful: {len(results['successful'])}")
    print(f"⏭️  Skipped (cached): {len(results['skipped'])}")
    print(f"❌ Failed (no amenities): {len(results['failed'])}")
    
    if results['failed']:
        print("\nAreas with no amenities found:")
        for area in results['failed'][:10]:  # Show first 10
            print(f"  - {area}")
        if len(results['failed']) > 10:
            print(f"  ... and {len(results['failed']) - 10} more")


def main():
    """Main execution function."""
    # Setup cache directory (data-pipeline/ at project root)
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent  # Go up from data-pipeline/ to project root
    cache_dir = project_dir / "data" / "raw" / "bengaluru" / "bengaluru_amenities"
    
    print("🚀 Fetching amenity data for Bengaluru areas...")
    print(f"📁 Cache directory: {cache_dir}")
    print("=" * 60)
    
    # First, get list of all areas in Bengaluru from OSM
    areas = get_bengaluru_areas()
    
    if not areas:
        print("❌ Failed to fetch areas list. Exiting.")
        return
    
    print(f"📍 Total areas to fetch: {len(areas)}")
    print("=" * 60)
    
    # Save the areas list for reference
    areas_file = project_dir / "data" / "raw" / "bengaluru" / "bengaluru_areas_list.json"
    areas_file.parent.mkdir(parents=True, exist_ok=True)
    with open(areas_file, 'w', encoding='utf-8') as f:
        json.dump(areas, f, indent=2, ensure_ascii=False)
    print(f"📋 Areas list saved to: {areas_file}\n")
    
    # Fetch all areas in batches: 5 areas with 2s gaps, then 10s rest
    results = fetch_all_areas(areas, cache_dir, batch_size=5, batch_delay=2.0, rest_delay=10.0)
    
    # Print summary
    print_summary(results)
    
    # Save summary file
    summary_file = cache_dir / "_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
