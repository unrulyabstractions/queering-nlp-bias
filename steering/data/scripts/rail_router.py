"""RAPTOR over the CTA rail GTFS. For every rail parent station, the scheduled
weekday-morning journey time to Illinois Medical District (40810) or Polk (41030),
whichever arrives first, with at most one transfer (and, for reference, up to two).
Departures: every scheduled train leaving the origin 07:00-08:59 that the optimal
journey actually boards (zero initial wait); wrong-direction trains are not departures."""
import csv, sys, json, collections, statistics
G = sys.argv[1]; OUT = sys.argv[2]
TARGETS = {'40810', '41030'}
WIN = (7*3600, 9*3600)
PEN_SAME = 60      # seconds to change trains inside one station complex
PEN_WALK = 180     # seconds for a transfers.txt link between two station complexes
MAXR = 3           # rounds: 1 = direct, 2 = one transfer, 3 = two transfers

def hms(s):
    h,m,sec = s.split(':'); return int(h)*3600+int(m)*60+int(sec)

stops = {r['stop_id']: r for r in csv.DictReader(open(f'{G}/stops.txt', newline='', encoding='utf-8-sig'))}
parent = {sid: (r['parent_station'] or sid) for sid, r in stops.items()}
pname = {sid: r['stop_name'] for sid, r in stops.items() if r['location_type'] == '1'}

cal = {r['service_id']: r for r in csv.DictReader(open(f'{G}/calendar.txt', newline='', encoding='utf-8-sig'))}
wed_services = {s for s, r in cal.items() if r['wednesday'] == '1'}
trips = {r['trip_id']: r for r in csv.DictReader(open(f'{G}/rail_trips.txt', newline='', encoding='utf-8')) if r['service_id'] in wed_services}
print('weekday (Wednesday) services used:', sorted(wed_services & {t['service_id'] for t in trips.values()}), 'trips', len(trips))

st = collections.defaultdict(list)
for r in csv.DictReader(open(f'{G}/rail_stop_times.txt', newline='', encoding='utf-8')):
    if r['trip_id'] in trips:
        st[r['trip_id']].append((int(r['stop_sequence']), hms(r['arrival_time']), hms(r['departure_time']), parent[r['stop_id']]))
for t in st: st[t].sort()

# route patterns: (route_id, stop sequence) -> list of trips (each: list of (arr,dep)) sorted by first departure
patterns = collections.defaultdict(list)
for tid, rows in st.items():
    key = (trips[tid]['route_id'], tuple(r[3] for r in rows))
    patterns[key].append((rows[0][2], tid, [(r[1], r[2]) for r in rows]))
for k in patterns: patterns[k].sort()
pat_list = list(patterns.keys())
routes_at = collections.defaultdict(list)   # parent stop -> [(pattern index, position)]
for pi, (rid, seq) in enumerate(pat_list):
    for pos, s in enumerate(seq): routes_at[s].append((pi, pos))
print('patterns', len(pat_list), 'rail parent stations with service', len(routes_at))

# footpaths between station complexes from transfers.txt (platform ids -> parents)
foot = collections.defaultdict(set)
for r in csv.DictReader(open(f'{G}/transfers.txt', newline='', encoding='utf-8-sig')):
    a, b = parent.get(r['from_stop_id'], r['from_stop_id']), parent.get(r['to_stop_id'], r['to_stop_id'])
    if a != b: foot[a].add(b); foot[b].add(a)
print('footpath links between distinct station complexes:', sum(len(v) for v in foot.values())//2)

def raptor(origin, tau):
    INF = 10**9
    best = {origin: tau}; label = [{origin: tau}]; back = {}
    marked = {origin}
    for k in range(1, MAXR+1):
        lab = {}
        # routes to scan
        q = {}
        for s in marked:
            for pi, pos in routes_at[s]:
                if pi not in q or pos < q[pi]: q[pi] = pos
        for pi, pos0 in q.items():
            rid, seq = pat_list[pi]; cur = None  # (trip rows, boarded pos)
            for pos in range(pos0, len(seq)):
                s = seq[pos]
                if cur is not None:
                    arr = cur[0][pos][0]
                    if arr < best.get(s, INF) and arr < min(best.get(t, INF) for t in TARGETS):
                        best[s] = arr; lab[s] = arr; back[(k, s)] = (rid, seq[cur[1]], cur[2], arr, cur[0][cur[1]][1])
                # can we board an earlier trip here?
                if s in label[k-1]:
                    t_ready = label[k-1][s] + (0 if k == 1 else (PEN_SAME if back.get((k-1, s), (None,))[0] != 'walk' else 0))
                    for dep0, tid, rows in patterns[pat_list[pi]]:
                        if rows[pos][1] >= t_ready:
                            if cur is None or rows[pos][1] < cur[0][pos][1]:
                                cur = (rows, pos, tid)
                            break
        # footpaths
        for s, a in list(lab.items()):
            for b in foot[s]:
                ab = a + PEN_WALK
                if ab < best.get(b, INF):
                    best[b] = ab; lab[b] = ab; back[(k, b)] = ('walk', s, None, ab, None)
        label.append(lab); marked = set(lab)
        if not marked: break
    return best, back, label

def legs(back, k, s):
    out = []
    while k > 0 and (k, s) in back:
        rid, frm, tid, arr, dep = back[(k, s)]
        out.append((rid, frm, s, dep))
        s = frm
        if rid != 'walk': k -= 1
    return list(reversed(out))

results = {}
for origin in sorted(routes_at):
    if origin in TARGETS:
        results[origin] = {'name': pname.get(origin), 'n_departures': 0, 'min1': 0, 'median1': 0, 'min2': 0, 'median2': 0, 'legs': ''}
        continue
    deps = sorted({rows[pos][1] for pi, pos in routes_at[origin] for _, _, rows in patterns[pat_list[pi]] if WIN[0] <= rows[pos][1] < WIN[1]})
    t1 = []; t2 = []; best_leg = None
    for tau in deps:
        best, back, label = raptor(origin, tau)
        # arrival with <=1 transfer: best target label achieved by round <= 2
        def arr_by(rmax):
            a = None
            for k in range(1, min(rmax, len(label)-1)+1):
                for t in TARGETS:
                    if t in label[k] and (a is None or label[k][t] < a[0]): a = (label[k][t], k, t)
            return a
        a1 = arr_by(2); a2 = arr_by(3)
        if a1:
            lg = legs(back, a1[1], a1[2])
            if lg and lg[0][3] == tau:          # the journey boards a train leaving at tau: a real departure, no initial wait
                t1.append((a1[0]-tau)/60)
                if best_leg is None or (a1[0]-tau) < best_leg[0]: best_leg = (a1[0]-tau, lg)
        if a2:
            lg2 = legs(back, a2[1], a2[2])
            if lg2 and lg2[0][3] == tau: t2.append((a2[0]-tau)/60)
    def desc(lg):
        return ' -> '.join(f"{rid}:{pname.get(frm,frm)}>{pname.get(to,to)}" if rid!='walk' else f"walk:{pname.get(frm,frm)}>{pname.get(to,to)}" for rid, frm, to, dep in lg)
    results[origin] = {'name': pname.get(origin), 'n_departure_events': len(deps), 'n_departures': len(t1), 'n_reached1': len(t1),
                       'min1': round(min(t1),1) if t1 else None, 'median1': round(statistics.median(t1),1) if t1 else None,
                       'min2': round(min(t2),1) if t2 else None, 'median2': round(statistics.median(t2),1) if t2 else None,
                       'legs': desc(best_leg[1]) if best_leg else ''}
json.dump(results, open(OUT, 'w'), indent=1)
for sid, r in sorted(results.items(), key=lambda kv: (kv[1]['median1'] if kv[1]['median1'] is not None else 999)):
    print(sid, r['name'], '| deps', r['n_departures'], '| <=1xfer min/med', r['min1'], r['median1'], '| <=2xfer', r['min2'], r['median2'], '|', r['legs'])
