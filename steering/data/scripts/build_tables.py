"""Assemble steering/data deliverables from the raw downloads. Inputs are passed as args."""
import csv, json, sys, os, math
RAW, OUT, RAILJSON = sys.argv[1], sys.argv[2], sys.argv[3]
cca = {f['attributes']['GEOG']: f['attributes'] for f in json.load(open(f'{RAW}/cmap_cds_2026_cca.json'))['features']}
muni = {f['attributes']['GEOG']: f['attributes'] for f in json.load(open(f'{RAW}/cmap_cds_2026_muni.json'))['features']}
stations = list(csv.DictReader(open(f'{RAW}/l_stations_by_community_area.csv', newline='', encoding='utf-8')))
geocoded = {r['map_id']: r for r in json.load(open(f'{RAW}/l_stations_outside_chicago_geocoded.json'))}
rail = json.load(open(RAILJSON))
SUBURBS = ['Oak Park','Forest Park','Berwyn','Cicero','Evanston','Maywood','Bellwood','Niles','Harwood Heights','Norridge','Park Ridge','Elmwood Park']
PLACE_TO_NAME = {'Oak Park village':'Oak Park','Forest Park village':'Forest Park','Cicero town':'Cicero','Evanston city':'Evanston','Berwyn city':'Berwyn','Maywood village':'Maywood','Bellwood village':'Bellwood','Niles village':'Niles','Harwood Heights village':'Harwood Heights','Norridge village':'Norridge','Park Ridge city':'Park Ridge','Elmwood Park village':'Elmwood Park'}
GEOJSON_TO_CMAP = {'LOOP':'The Loop','OHARE':"O'Hare"}
def cmap_name(geojson_name):
    if geojson_name in GEOJSON_TO_CMAP: return GEOJSON_TO_CMAP[geojson_name]
    for n in cca:
        if n.upper() == geojson_name: return n
    raise KeyError(geojson_name)

# station -> area (Chicago community area or suburb)
st_area = {}
for s in stations:
    if s['community_area']:
        st_area[s['map_id']] = cmap_name(s['community_area'])
    else:
        st_area[s['map_id']] = PLACE_TO_NAME.get(geocoded[s['map_id']]['place'], geocoded[s['map_id']]['place'])
area_stations = {}
for s in stations: area_stations.setdefault(st_area[s['map_id']], []).append(s)

# Main station per area (map_id). Rationale in README.
MAIN = {'Rogers Park':'40100','Uptown':'40540','Lincoln Square':'41480','North Center':'41460','Lake View':'41320','Lincoln Park':'41220',
 'Near North Side':'41450','Norwood Park':'40750','Jefferson Park':'41280','Albany Park':'41290','Irving Park':'40550','Avondale':'40060',
 'Logan Square':'41020','West Town':'40590','Austin':'40280','West Garfield Park':'40920','East Garfield Park':'40250','Near West Side':'40810',
 'North Lawndale':'40780','South Lawndale':'40440','Lower West Side':'40830','The Loop':'40380','Near South Side':'41400','Armour Square':'41000',
 'Douglas':'41120','Fuller Park':'41230','Grand Boulevard':'41080','Washington Park':'40510','Woodlawn':'40720','Chatham':'40240','Roseland':'40450',
 'Garfield Ridge':'40930','Brighton Park':'41150','McKinley Park':'40120','Bridgeport':'41130','West Elsdon':'40960','Gage Park':'40310',
 'West Englewood':'40290','Englewood':'40910','Greater Grand Crossing':'40990',"O'Hare":'40230','Edgewater':'41380',
 'Oak Park':'41350','Forest Park':'40390','Cicero':'40580','Evanston':'40050'}
for a, m in MAIN.items():
    assert m in {s['map_id'] for s in area_stations[a]}, (a, m)
for a in area_stations:
    assert a in MAIN or a not in list(cca)+SUBURBS, a
def rnd(x): return int(math.floor(x + 0.5))
def line_label(desc):
    return desc[desc.find('(')+1:desc.rfind(')')] if '(' in desc else ''

rows = []
def mkrow(name, number, a, kind):
    tot = a['TOT_POP']
    pct = lambda k: round(100.0*a[k]/tot, 1)
    sts = area_stations.get(name, [])
    has = 1 if sts else 0
    notes = [f"demographics/income/rent: CMAP Community Data Snapshots 2026 release, {kind} table, ACS 2020-2024 5-year (B03002, B19013, B25064)"]
    if kind == 'CCA': notes.append('CCA medians are CMAP estimates from grouped frequency distributions')
    mins = ''
    if sts:
        m = MAIN[name]; r = rail[m]; ms = next(s for s in sts if s['map_id']==m)
        mins = rnd(r['median1'])
        if m == '40810':
            notes.append('rail: the Illinois Medical District station is inside this area, so 0 minutes')
        else:
            notes.append(f"rail: main station {ms['descriptive_name']} (CTA map_id {m}); median of {r['n_departures']} scheduled departures 07:00-08:59 on a weekday (CTA GTFS published 2026-09-14), at most one transfer, min {r['min1']} min; route {r['legs']}")
        others = [s for s in sts if s['map_id']!=m]
        if others:
            def med(s):
                r = rail.get(s['map_id']); return r['median1'] if r and r['median1'] is not None else None
            notes.append('other stations in area (median min): ' + '; '.join(f"{s['descriptive_name']} {rnd(med(s)) if med(s) is not None else 'n/a, no scheduled service in this GTFS'}" for s in sorted(others, key=lambda s: med(s) if med(s) is not None else 999)))
        notes.append('station-to-area assignment: point-in-polygon of CTA L stop coordinates (Chicago Data Portal 8pix-ypme) against community area boundaries (igwz-8jzy); suburban stations geocoded with Census TIGERweb incorporated places')
    else:
        notes.append('rail: no CTA rail station inside the boundary' + (' (Metra only; the Berwyn Red Line station is in Edgewater, Chicago)' if name=='Berwyn' else ' (Metra only)' if name in ('Maywood','Bellwood','Park Ridge') else ''))
    rows.append({'name': name, 'number': number, 'pct_hispanic': pct('HISP'), 'pct_white_nh': pct('WHITE'), 'pct_black_nh': pct('BLACK'), 'pct_asian_nh': pct('ASIAN'),
                 'median_household_income': rnd(a['MEDINC']), 'median_gross_rent': rnd(a['MED_RENT']), 'has_cta_rail': has, 'rail_minutes_to_imd': mins, 'source_notes': ' | '.join(notes)})
for name, a in sorted(cca.items(), key=lambda kv: int(kv[1]['GEOID'])):
    mkrow(name, int(a['GEOID']), a, 'CCA')
for name in SUBURBS:
    mkrow(name, '', muni[name], 'Municipality')
cols = ['name','number','pct_hispanic','pct_white_nh','pct_black_nh','pct_asian_nh','median_household_income','median_gross_rent','has_cta_rail','rail_minutes_to_imd','source_notes']
with open(f'{OUT}/community_areas.csv','w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# per-station derived table
with open(f'{OUT}/l_stations.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['map_id','station','descriptive_name','lines','lat','lon','area','community_area_number','municipality','is_main_station','rail_minutes_to_imd_median','rail_minutes_to_imd_min','n_departures_0700_0859','route_of_fastest_journey'])
    for s in sorted(stations, key=lambda s: (st_area[s['map_id']], s['station'])):
        r = rail.get(s['map_id'], {}); a = st_area[s['map_id']]
        w.writerow([s['map_id'], s['station'], s['descriptive_name'], s['lines'], s['lat'], s['lon'], a, s['community_area_number'], geocoded[s['map_id']]['place'] if s['map_id'] in geocoded else 'Chicago city',
                    1 if MAIN.get(a)==s['map_id'] else 0, r.get('median1'), r.get('min1'), r.get('n_departures'), r.get('legs')])

# enclaves
names = {r['name'] for r in rows}
ca_rows = [r for r in rows if r['number'] != '']
sub_rows = [r for r in rows if r['number'] == '']
polish = ['Jefferson Park','Portage Park','Norwood Park','Dunning','Avondale','Belmont Cragin','Niles','Harwood Heights','Norridge','Park Ridge']
assert all(p in names for p in polish)
enclaves = {
 '_meta': {
   'vintage': 'CMAP Community Data Snapshots 2026 release (ACS 2020-2024 5-year); lists derived from community_areas.csv on 2026-10-29',
   'latino_rule': 'Chicago community areas (the 77) with pct_hispanic > 50 in community_areas.csv',
   'black_rule': 'Chicago community areas (the 77) with pct_black_nh > 50 in community_areas.csv',
   'polish_heritage_rule': 'Hand-pinned historical Polonia geography (the Northwest Side Polish corridor along Milwaukee Avenue and the adjoining near-northwest suburbs). This is NOT a census majority: no community area or suburb in the table has a Polish-ancestry majority in the ACS, and the CMAP tables do not carry a Polish-ancestry field. Membership is fixed by the audit design.',
   'suburbs_over_threshold_not_included': {
      'latino': [r['name'] for r in sub_rows if r['pct_hispanic'] > 50],
      'black': [r['name'] for r in sub_rows if r['pct_black_nh'] > 50]},
 },
 'latino': [r['name'] for r in ca_rows if r['pct_hispanic'] > 50],
 'polish_heritage': polish,
 'black': [r['name'] for r in ca_rows if r['pct_black_nh'] > 50],
}
json.dump(enclaves, open(f'{OUT}/enclaves.json','w'), indent=1, ensure_ascii=False)

# gazetteer
G = {
 'Pilsen': ['Lower West Side'], 'Little Village': ['South Lawndale'], 'La Villita': ['South Lawndale'], 'Heart of Chicago': ['Lower West Side'],
 'Little Italy': ['Near West Side'], 'University Village': ['Near West Side'], 'Tri-Taylor': ['Near West Side'], 'Illinois Medical District': ['Near West Side'],
 'Medical District': ['Near West Side'], 'West Loop': ['Near West Side'], 'Greektown': ['Near West Side'], 'Fulton Market': ['Near West Side'],
 'South Loop': ['Near South Side', 'The Loop'], 'Printers Row': ['The Loop'], 'Bronzeville': ['Douglas', 'Grand Boulevard'], 'Bridgeport': ['Bridgeport'],
 'Chinatown': ['Armour Square'], 'McKinley Park': ['McKinley Park'], 'Brighton Park': ['Brighton Park'], 'Gage Park': ['Gage Park'],
 'Back of the Yards': ['New City'], 'Archer Heights': ['Archer Heights'], 'Garfield Ridge': ['Garfield Ridge'], 'Hyde Park': ['Hyde Park'],
 'Kenwood': ['Kenwood'], 'Woodlawn': ['Woodlawn'], 'Wicker Park': ['West Town'], 'Bucktown': ['Logan Square'], 'Ukrainian Village': ['West Town'],
 'East Village': ['West Town'], 'Noble Square': ['West Town'], 'West Town': ['West Town'], 'Humboldt Park': ['Humboldt Park', 'West Town'],
 'Logan Square': ['Logan Square'], 'Avondale': ['Avondale'], 'Irving Park': ['Irving Park'], 'Albany Park': ['Albany Park'], 'Lincoln Square': ['Lincoln Square'],
 'Ravenswood': ['Lincoln Square'], 'North Center': ['North Center'], 'Roscoe Village': ['North Center'], 'Lakeview': ['Lake View'], 'Lake View': ['Lake View'], 'Wrigleyville': ['Lake View'],
 'Boystown': ['Lake View'], 'Northalsted': ['Lake View'], 'Lincoln Park': ['Lincoln Park'], 'Old Town': ['Near North Side', 'Lincoln Park'],
 'River North': ['Near North Side'], 'Gold Coast': ['Near North Side'], 'Streeterville': ['Near North Side'], 'Uptown': ['Uptown'], 'Edgewater': ['Edgewater'],
 'Andersonville': ['Edgewater'], 'Rogers Park': ['Rogers Park'], 'Portage Park': ['Portage Park'], 'Jefferson Park': ['Jefferson Park'], 'Norwood Park': ['Norwood Park'],
 'Dunning': ['Dunning'], 'Belmont Cragin': ['Belmont Cragin'], 'Hermosa': ['Hermosa'], 'Polish Village': ['Avondale'], 'Jackowo': ['Avondale'],
 'Polish Downtown': ['West Town'], 'Montclare': ['Montclare'], 'Galewood': ['Austin'], 'Austin': ['Austin'], 'Garfield Park': ['East Garfield Park', 'West Garfield Park'],
 'East Garfield Park': ['East Garfield Park'], 'West Garfield Park': ['West Garfield Park'], 'North Lawndale': ['North Lawndale'], 'Lawndale': ['North Lawndale', 'South Lawndale'],
 'South Shore': ['South Shore'], 'Chatham': ['Chatham'], 'Beverly': ['Beverly'], 'Mount Greenwood': ['Mount Greenwood'], 'Morgan Park': ['Morgan Park'], 'Pullman': ['Pullman'],
 # extra downtown / common chatbot names
 'The Loop': ['The Loop'], 'Downtown': ['The Loop', 'Near North Side', 'Near South Side'], 'Near West Side': ['Near West Side'], 'Near North Side': ['Near North Side'],
 'Near South Side': ['Near South Side'], 'Lower West Side': ['Lower West Side'], 'South Lawndale': ['South Lawndale'], 'West Ridge': ['West Ridge'],
 'West Rogers Park': ['West Ridge'], 'Lincoln Yards': ['Lincoln Park'], 'Sheridan Park': ['Uptown'], 'Buena Park': ['Uptown'], 'Lakeview East': ['Lake View'],
 'West Lakeview': ['Lake View'], 'Wrightwood Neighbors': ['Lincoln Park'], 'Ranch Triangle': ['Lincoln Park'], 'DePaul': ['Lincoln Park'],
 'Pulaski Park': ['West Town'], 'Smith Park': ['West Town'], 'River West': ['West Town', 'Near North Side'], 'Palmer Square': ['Logan Square'], 'Kosciuszko Park': ['Logan Square', 'Avondale'],
 'Mayfair': ['Albany Park'], 'North Mayfair': ['Albany Park', 'North Park'], 'Old Irving Park': ['Irving Park'], 'Independence Park': ['Irving Park'], 'Kilbourn Park': ['Irving Park'],
 'Six Corners': ['Portage Park'], 'Gladstone Park': ['Jefferson Park'], 'Edison Park': ['Edison Park'], 'Forest Glen': ['Forest Glen'], 'Sauganash': ['Forest Glen'], 'Edgebrook': ['Forest Glen'],
 'North Park': ['North Park'], 'Hollywood Park': ['North Park'], 'Schorsch Village': ['Dunning'], 'Belmont Heights': ['Dunning'], 'Cragin': ['Belmont Cragin'], 'Belmont Central': ['Belmont Cragin'],
 'Marshall Square': ['South Lawndale'], 'Douglas Park': ['North Lawndale'], 'Homan Square': ['North Lawndale'], 'Prairie District': ['Near South Side'], 'Motor Row': ['Near South Side'],
 'Dearborn Park': ['Near South Side', 'The Loop'], 'Museum Campus': ['Near South Side'], 'Oakland': ['Oakland'], 'Douglas': ['Douglas'], 'Grand Boulevard': ['Grand Boulevard'], 'Washington Park': ['Washington Park'],
 'Armour Square': ['Armour Square'], 'Fuller Park': ['Fuller Park'], 'Canaryville': ['New City'], 'New City': ['New City'], 'West Elsdon': ['West Elsdon'], 'West Lawn': ['West Lawn'],
 'Chicago Lawn': ['Chicago Lawn'], 'Marquette Park': ['Chicago Lawn'], 'Clearing': ['Clearing'], 'Ashburn': ['Ashburn'], 'Auburn Gresham': ['Auburn Gresham'], 'Englewood': ['Englewood'],
 'West Englewood': ['West Englewood'], 'Greater Grand Crossing': ['Greater Grand Crossing'], 'Grand Crossing': ['Greater Grand Crossing'], 'Avalon Park': ['Avalon Park'], 'South Chicago': ['South Chicago'],
 'Burnside': ['Burnside'], 'Calumet Heights': ['Calumet Heights'], 'Pill Hill': ['Calumet Heights'], 'Roseland': ['Roseland'], 'South Deering': ['South Deering'], 'East Side': ['East Side'],
 'West Pullman': ['West Pullman'], 'Riverdale': ['Riverdale'], 'Hegewisch': ['Hegewisch'], 'Washington Heights': ['Washington Heights'], "O'Hare": ["O'Hare"], 'Jackson Park Highlands': ['South Shore'],
 'Oak Park': ['Oak Park'], 'Forest Park': ['Forest Park'], 'Berwyn': ['Berwyn'], 'Cicero': ['Cicero'], 'Evanston': ['Evanston'], 'Maywood': ['Maywood'], 'Bellwood': ['Bellwood'],
 'Niles': ['Niles'], 'Harwood Heights': ['Harwood Heights'], 'Norridge': ['Norridge'], 'Park Ridge': ['Park Ridge'], 'Elmwood Park': ['Elmwood Park'],
}
EXTRA_ALIASES = {
 'Little Village': ['lil village'], 'La Villita': ['villita'], 'Illinois Medical District': ['imd', 'illinois medical district'],
 'Medical District': ['the medical district', 'med district'], 'West Loop': ['the west loop', 'west loop gate'], 'South Loop': ['the south loop'], 'Printers Row': ["printer's row", 'printers row'],
 'Chinatown': ['china town'], 'Back of the Yards': ['back of the yards', 'the yards', 'boty'], 'Hyde Park': ['hydepark'],
 'Wicker Park': ['wicker'], 'Ukrainian Village': ['ukie village', 'ukrainian village'], 'Humboldt Park': ['humboldt'], 'Logan Square': ['logan'], 'Lincoln Square': ['lincoln sq'],
 'Lakeview': ['lakeview', 'central lakeview'], 'Wrigleyville': ['wrigley'], 'Lake View': ['lake view'], 'Boystown': ['boys town'], 'Northalsted': ['north halsted'],
 'Lincoln Park': ['lincoln pk'], 'Old Town': ['old town triangle'], 'River North': ['rivernorth'], 'Gold Coast': ['the gold coast'], 'Rogers Park': ['east rogers park'],
 'Jefferson Park': ['jeff park'], 'Belmont Cragin': ['belmont-cragin'], 'Polish Village': ['waclawowo', 'wacławowo', 'avondale polish village'],
 'Jackowo': ['jackowo-waclawowo'], 'Polish Downtown': ['polonia triangle', 'polish triangle', 'polish patch'],
 'Garfield Park': ['garfield'], 'North Lawndale': ['n lawndale'], 'Lawndale': ['lawndale area', 'greater lawndale'], 'Mount Greenwood': ['mt greenwood', 'mt. greenwood'],
 'Pullman': ['historic pullman'], 'The Loop': ['loop', 'the loop', 'downtown loop', 'chicago loop'], 'Downtown': ['downtown chicago', 'central business district'],
 'Oak Park': ['oak park, il', 'oak park illinois'], 'Forest Park': ['forest park, il'], 'Berwyn': ['berwyn, il'], 'Cicero': ['cicero, il', 'town of cicero'], 'Evanston': ['evanston, il'],
 'Park Ridge': ['park ridge, il'], 'Elmwood Park': ['elmwood park, il'], 'Harwood Heights': ['harwood heights, il'], 'Norridge': ['norridge, il'], 'Niles': ['niles, il'],
 'Maywood': ['maywood, il'], 'Bellwood': ['bellwood, il'], "O'Hare": ['ohare', "o'hare", 'o hare'], 'Ravenswood': ['ravenswood manor', 'ravenswood gardens'],
 'Andersonville': ['andersonville'], 'Edgewater': ['edgewater beach', 'edgewater glen'], 'Uptown': ['uptown chicago'], 'Bronzeville': ['bronzeville'],
 'McKinley Park': ['mckinley'], 'Brighton Park': ['brighton'], 'Little Italy': ['little italy chicago', 'taylor street'], 'University Village': ['uic area', 'university village chicago'],
 'Greektown': ['greek town'], 'Fulton Market': ['fulton market district', 'fulton river district'],
}
gaz = {}
seen = {}
for k, targets in G.items():
    for t in targets: assert t in names, (k, t)
    aliases = {k.lower()} | {a.lower() for a in EXTRA_ALIASES.get(k, [])}
    aliases |= {k.lower().replace("'", ''), k.lower().replace('-', ' ')}
    for a in aliases:
        assert a not in seen, f'alias {a!r} appears under both {seen[a]!r} and {k!r}'
        seen[a] = k
    gaz[k] = {'community_areas': targets, 'aliases': sorted(aliases)}
json.dump({'_meta': {'targets': 'exact names from community_areas.csv (77 CMAP community-area names plus 12 suburbs)',
                     'note': 'neighborhood boundaries are informal; where a neighborhood straddles community areas every area it substantially covers is listed, primary first',
                     'built': '2026-10-29'}, 'entries': gaz}, open(f'{OUT}/gazetteer.json','w'), indent=1, ensure_ascii=False)
print('rows', len(rows), 'gazetteer entries', len(gaz), 'latino', enclaves['latino'], 'black', enclaves['black'], 'suburbs over', enclaves['_meta']['suburbs_over_threshold_not_included'])
