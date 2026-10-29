import json, csv, sys, re
geo = json.load(open(sys.argv[1]))
stops = list(csv.DictReader(open(sys.argv[2], newline='', encoding='utf-8-sig')))
def pip(pt, ring):
    x,y = pt; inside=False; n=len(ring)
    for i in range(n):
        x1,y1 = ring[i]; x2,y2 = ring[(i+1)%n]
        if (y1>y) != (y2>y):
            xi = x1 + (y-y1)*(x2-x1)/(y2-y1)
            if xi > x: inside = not inside
    return inside
def in_mp(pt, geom):
    polys = geom['coordinates'] if geom['type']=='MultiPolygon' else [geom['coordinates']]
    for poly in polys:
        if pip(pt, poly[0]) and not any(pip(pt, h) for h in poly[1:]): return True
    return False
areas = [(f['properties']['community'], int(f['properties']['area_numbe']), f['geometry']) for f in geo['features']]
# dedupe stations by MAP_ID (parent station)
stations = {}
for s in stops:
    m = re.match(r'\(([-\d.]+), ([-\d.]+)\)', s['Location'])
    lat, lon = float(m.group(1)), float(m.group(2))
    st = stations.setdefault(s['MAP_ID'], {'map_id': s['MAP_ID'], 'station': s['STATION_NAME'], 'desc': s['STATION_DESCRIPTIVE_NAME'], 'lat': lat, 'lon': lon, 'lines': set()})
    for ln in ['RED','BLUE','G','BRN','P','Y','Pnk','O']:
        if s[ln]=='true': st['lines'].add(ln)
print('stop rows', len(stops), 'stations', len(stations))
out = []
for st in stations.values():
    hit = [(n,num) for n,num,g in areas if in_mp((st['lon'],st['lat']), g)]
    st['ca'] = hit[0][0] if hit else ''
    st['ca_num'] = hit[0][1] if hit else ''
    if len(hit)>1: print('MULTI', st['station'], hit)
    out.append(st)
out.sort(key=lambda s:(s['ca_num'] if s['ca_num']!='' else 999, s['station']))
with open(sys.argv[3],'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['map_id','station','descriptive_name','lat','lon','lines','community_area','community_area_number'])
    for s in out: w.writerow([s['map_id'],s['station'],s['desc'],s['lat'],s['lon'],'|'.join(sorted(s['lines'])),s['ca'],s['ca_num']])
for s in out: print(s['ca_num'], s['ca'], '|', s['map_id'], s['station'], '|', s['desc'], '|', '/'.join(sorted(s['lines'])))
