import sys, csv, time
gtfs = sys.argv[1]
rail = {"Red","Blue","Brn","G","Org","P","Pink","Y"}
trip_ids = {}
with open(f"{gtfs}/trips.txt", newline='', encoding='utf-8-sig') as f:
    r = csv.DictReader(f)
    n=0
    for row in r:
        n+=1
        if row['route_id'] in rail:
            trip_ids[row['trip_id']] = row
print('trips total', n, 'rail trips', len(trip_ids), flush=True)
with open(f"{gtfs}/rail_trips.txt", 'w', newline='', encoding='utf-8') as out:
    w = csv.DictWriter(out, fieldnames=list(next(iter(trip_ids.values())).keys()))
    w.writeheader()
    for row in trip_ids.values(): w.writerow(row)
t0=time.time(); kept=0; total=0
with open(f"{gtfs}/stop_times.txt", newline='', encoding='utf-8-sig') as f, open(f"{gtfs}/rail_stop_times.txt",'w',newline='',encoding='utf-8') as out:
    header = f.readline()
    out.write(header)
    cols = header.strip().split(',')
    ti = cols.index('trip_id')
    for line in f:
        total+=1
        tid = line.split(',',1)[0] if ti==0 else line.split(',')[ti]
        if tid in trip_ids:
            out.write(line); kept+=1
print('stop_times total', total, 'rail rows kept', kept, 'secs', round(time.time()-t0,1), flush=True)
