#!/usr/bin/env python3
"""
Exploratory Data Analysis for Bengaluru Amenities Cache
Analyzes all JSON files to understand data composition and transit coverage
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

def analyze_amenities():
    cache_dir = Path("../scripts/cache/bengaluru_amenities")
    json_files = [f for f in cache_dir.glob("*.json") if f.name != "_summary.json"]
    
    # Counters for different categories
    amenity_types = Counter()
    shop_types = Counter()
    leisure_types = Counter()
    transit_types = Counter()
    transit_details = []
    
    print(f"📊 Analyzing {len(json_files)} area files...")
    print("=" * 80)
    
    total_amenities = 0
    areas_with_transit = 0
    
    for json_file in tqdm(json_files, desc="Processing areas"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            area_name = data.get("area_name", json_file.stem)
            area_transit_count = 0
            
            for item in data.get("amenities", []):
                tags = item.get("tags", {})
                total_amenities += 1
                
                # Main categories
                if "amenity" in tags:
                    amenity_types[tags["amenity"]] += 1
                if "shop" in tags:
                    shop_types[tags["shop"]] += 1
                if "leisure" in tags:
                    leisure_types[tags["leisure"]] += 1
                
                # Transit detection - comprehensive check
                is_transit = False
                transit_type = None
                
                if "public_transport" in tags:
                    transit_type = f"public_transport={tags['public_transport']}"
                    is_transit = True
                elif "railway" in tags:
                    transit_type = f"railway={tags['railway']}"
                    is_transit = True
                elif "highway" in tags and tags["highway"] == "bus_stop":
                    transit_type = "bus_stop"
                    is_transit = True
                elif tags.get("amenity") == "bus_station":
                    transit_type = "bus_station"
                    is_transit = True
                
                if is_transit:
                    transit_types[transit_type] += 1
                    area_transit_count += 1
                    transit_details.append({
                        "area": area_name,
                        "type": transit_type,
                        "name": tags.get("name", "Unnamed"),
                        "osmid": item.get("id"),
                        "tags": tags
                    })
            
            if area_transit_count > 0:
                areas_with_transit += 1
        
        except Exception as e:
            print(f"\n⚠️  Error processing {json_file.name}: {e}")
            continue
    
    # Print results
    print("\n" + "=" * 80)
    print("🎯 OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total areas analyzed: {len(json_files)}")
    print(f"Total amenities/POIs: {total_amenities:,}")
    print(f"Areas with transit data: {areas_with_transit}")
    print(f"Transit coverage: {areas_with_transit/len(json_files)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("🏢 TOP 30 AMENITY TYPES")
    print("=" * 80)
    for amenity, count in amenity_types.most_common(30):
        pct = count / total_amenities * 100
        print(f"{amenity:35s} {count:7,d}  ({pct:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("🛒 TOP 20 SHOP TYPES")
    print("=" * 80)
    for shop, count in shop_types.most_common(20):
        pct = count / total_amenities * 100
        print(f"{shop:35s} {count:7,d}  ({pct:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("🎾 TOP 15 LEISURE TYPES")
    print("=" * 80)
    for leisure, count in leisure_types.most_common(15):
        pct = count / total_amenities * 100
        print(f"{leisure:35s} {count:7,d}  ({pct:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("🚌 TRANSIT DATA (Bus Stops, Metro Stations, etc.)")
    print("=" * 80)
    if transit_types:
        total_transit = sum(transit_types.values())
        print(f"Total transit points found: {total_transit:,}\n")
        for transit, count in sorted(transit_types.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_transit * 100
            print(f"{transit:45s} {count:6,d}  ({pct:5.2f}%)")
        
        # Show some examples
        print("\n" + "-" * 80)
        print("📍 Sample Transit Locations (first 20):")
        print("-" * 80)
        for detail in transit_details[:20]:
            print(f"{detail['type']:25s} | {detail['name']:40s} | {detail['area']}")
    else:
        print("❌ NO TRANSIT DATA FOUND!")
        print("\n⚠️  The amenity query did NOT include bus stops or metro stations.")
        print("   The Overpass query only searched for:")
        print("   - amenity=*")
        print("   - shop=*")
        print("   - leisure=*")
        print("\n💡 To get transit data, you need to add these queries:")
        print("   - highway=bus_stop")
        print("   - public_transport=*")
        print("   - railway=station")
    
    print("\n" + "=" * 80)
    print("📈 CATEGORY BREAKDOWN")
    print("=" * 80)
    categories = {
        "Amenities": sum(amenity_types.values()),
        "Shops": sum(shop_types.values()),
        "Leisure": sum(leisure_types.values()),
        "Transit": sum(transit_types.values())
    }
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_amenities * 100 if total_amenities > 0 else 0
        print(f"{cat:20s} {count:7,d}  ({pct:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("🔍 DIVERSITY METRICS")
    print("=" * 80)
    print(f"Unique amenity types: {len(amenity_types)}")
    print(f"Unique shop types: {len(shop_types)}")
    print(f"Unique leisure types: {len(leisure_types)}")
    print(f"Unique transit types: {len(transit_types)}")
    
    # Save detailed transit data
    if transit_details:
        transit_df = pd.DataFrame(transit_details)
        output_path = Path("transit_locations.csv")
        transit_df[["area", "type", "name", "osmid"]].to_csv(output_path, index=False)
        print(f"\n💾 Transit locations saved to: {output_path}")
    
    return {
        "amenity_types": amenity_types,
        "shop_types": shop_types,
        "leisure_types": leisure_types,
        "transit_types": transit_types,
        "transit_details": transit_details,
        "total_amenities": total_amenities
    }

if __name__ == "__main__":
    results = analyze_amenities()
