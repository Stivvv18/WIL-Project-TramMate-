#!/usr/bin/env python3

from importlib.resources import path
from turtle import poly, rt
import csv, json, zipfile, statistics
from pathlib import Path
from collections import defaultdict

GTFS_ZIP = Path("data/gtfs/latest_gtfs.zip")
OUT_DIR = Path("data/curated"); OUT_DIR.mkdir(parents=True, exist_ok=True)
POLY_PATH= Path("data/curated/cbd_polygon.geojson") # create once
# adjust if your feed uses different numbers; 0 is standard for tram/light rail
TRAM_ROUTE_TYPES = {0, 900, 902, 2} # tolerant set; keep what matches your feed

def load_geojson_polygon(path: Path):
    gj = json.loads(path.read_text(encoding="utf-8"))
    if gj.get("type") == "FeatureCollection":
        gj = gj["features"][0]["geometry"]
    if gj.get("type") == "Polygon":
        # use the outer ring only
       coords = gj["coordinates"][0]
    elif gj.get("type") == "MultiPolygon":
       coords = gj["coordinates"][0][0]
    else:
        raise ValueError("Unsupported GeoJSON geometry")
    # return as list of (lon,lat)
    return [(float(x), float(y)) for x, y in coords]


def point_in_polygon(lon, lat, polygon):
    # Ray casting algorithm
    x, y = lon, lat
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersect = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 -
y1 + 1e-12) + x1)
        if intersect:
            inside = not inside
    return inside


def read_txt(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as f:
        rows = list(csv.DictReader((line.decode("utf-8-sig") for line in f)))
        return rows
    
def main():
    assert GTFS_ZIP.exists(), f"Missing {GTFS_ZIP}"
    assert POLY_PATH.exists(), f"Missing {POLY_PATH} (draw FTZ polygon once)"

    poly = load_geojson_polygon(POLY_PATH)

    with zipfile.ZipFile(GTFS_ZIP, 'r') as z:
        routes = read_txt(z, "routes.txt")
        trips = read_txt(z, "trips.txt")
        stop_times = read_txt(z, "stop_times.txt")
        stops = read_txt(z, "stops.txt")

    # Filter the tram routes (robustly)
    tram_routes = []
    for r in routes:
        try:
            rt = int(float(r.get('route_type', 0)))
        except Exception:
            rt = 0
        if rt in TRAM_ROUTE_TYPES:
            tram_routes.append(r)
    tram_route_ids = {r['route_id'] for r in tram_routes}

    # Write tram_routes.csv
    out_routes = []
    for r in tram_routes:
        out_routes.append({
            'route_id': r.get('route_id',''),
            'route_no': r.get('route_short_name',''),
            'route_name': r.get('route_long_name') or r.get('route_desc','')
        })
    with (OUT_DIR/"tram_routes.csv").open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['route_id','route_no','route_name'])
        w.writeheader(); w.writerows(out_routes)

    # Stops + FTZ flag
    out_stops = []
    in_ftz = set()
    for s in stops:
        lat = float(s.get('stop_lat', '0'))
        lon = float(s.get('stop_lon', '0'))
        inside = point_in_polygon(lon, lat, poly)
        if inside:
            in_ftz.add(s['stop_id'])
        out_stops.append({
            'stop_id': s.get('stop_id',''),
            'stop_name': s.get('stop_name',''),
            'lat': lat,
            'lon': lon,
            'wheelchair_boarding': s.get('wheelchair_boarding','')
        })
    with (OUT_DIR/"tram_stops.csv").open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['stop_id','stop_name','lat','lon','wheelchair_boarding'])
        w.writeheader(); w.writerows(out_stops)

    with (OUT_DIR/"stops_in_ftz.csv").open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['stop_id'])
        for sid in sorted(in_ftz):
            w.writerow([sid])

    # Map stop_ids to route_ids using trips + stop_times
    route_stops = defaultdict(lambda: defaultdict(list)) # route_id -> stop_id -> [sequences]

    # route for each trip
    trip_to_route = {t['trip_id']: t['route_id'] for t in trips if t['route_id'] in tram_route_ids}
    
    for st in stop_times:
        trip_id = st['trip_id']
        route_id = trip_to_route.get(trip_id)
        if route_id is None:
            continue
        sid = st['stop_id']
        if sid not in in_ftz:
            continue
        try:
            seq = int(st.get('stop_sequence','0'))
        except Exception:
            continue
        route_stops[route_id][sid].append(seq)

    # Aggregate: median sequence → order
    rows = []
    sid_to_name = {s['stop_id']: s['stop_name'] for s in stops}
    rid_to_no = {r['route_id']: r.get('route_short_name','') for r in tram_routes}
    for rid, stops_dict in route_stops.items():
        for sid, seqs in stops_dict.items():
            med = statistics.median(seqs) if seqs else 0
            rows.append({
                'route_id': rid,
                'route_no': rid_to_no.get(rid, ''),
                'stop_id': sid,
                'stop_name': sid_to_name.get(sid, ''),
                'order_in_cbd': f"{med:09.2f}"
            })

    # Sort by route + order and write
    rows.sort(key=lambda r: (r['route_id'], r['order_in_cbd']))
    with (OUT_DIR/"route_stops_in_cbd.csv").open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['route_id','route_no','stop_id','stop_name','order_in_cbd'])
        w.writeheader(); w.writerows(rows)

        print("Done: tram_routes.csv, tram_stops.csv, stops_in_ftz.csv, route_stops_in_cbd.csv")

if __name__ == "__main__":
    main()
