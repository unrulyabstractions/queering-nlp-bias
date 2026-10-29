"""Cross-check community_areas.csv against the Wikipedia race/ethnicity table.
Rows are matched by name (not by Wikipedia's No. column, which has typos) and the six
counts are compared with CMAP's raw counts as well as the four percentage columns."""
import sys, re, csv, json
from html.parser import HTMLParser
src = open(sys.argv[1], encoding='utf-8').read()
rows = {r['name']: r for r in csv.DictReader(open(sys.argv[2], newline='', encoding='utf-8')) if r['number']}
cmap = {f['attributes']['GEOG']: f['attributes'] for f in json.load(open(sys.argv[3]))['features']}
class TP(HTMLParser):
    def __init__(self):
        super().__init__(); self.tables=[]; self.cur=None; self.row=None; self.cell=None; self.depth=0
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=='table' and 'wikitable' in (a.get('class') or ''): self.cur=[]; self.depth=1
        elif self.cur is not None:
            if tag=='table': self.depth+=1
            elif tag=='tr' and self.depth==1: self.row=[]
            elif tag in ('td','th') and self.row is not None: self.cell=[]
    def handle_endtag(self, tag):
        if self.cur is None: return
        if tag=='table':
            self.depth-=1
            if self.depth==0: self.tables.append(self.cur); self.cur=None
        elif tag in ('td','th') and self.cell is not None:
            self.row.append(re.sub(r'\s+',' ',''.join(self.cell)).strip()); self.cell=None
        elif tag=='tr' and self.row is not None: self.cur.append(self.row); self.row=None
    def handle_data(self, d):
        if self.cell is not None: self.cell.append(d)
p = TP(); p.feed(src)
race = p.tables[1]
print('header:', race[0])
def num(x): return float(x.replace(',', '')) if x not in ('', '-') else None
wiki = {}
for r in race[1:]:
    if len(r) >= 13 and r[1] and r[1] != 'Chicago':
        nm = re.sub(r'\[.*?\]', '', r[1]).strip()
        nm = {'Loop': 'The Loop'}.get(nm, nm)   # CMAP spelling
        wiki[nm] = {'no': r[0], 'tot': num(r[2]), 'white': num(r[3]), 'hisp': num(r[4]), 'black': num(r[5]), 'asian': num(r[6]), 'other': num(r[7]),
                    'pw': num(r[8]), 'ph': num(r[9]), 'pb': num(r[10]), 'pa': num(r[11])}
print('wikipedia rows:', len(wiki))
unmatched = [n for n in wiki if n not in rows]
print('wikipedia names not in csv:', unmatched)
exact = 0; defects = []
for n, w in wiki.items():
    if n not in rows: continue
    r = rows[n]; c = cmap[n]
    cnt_ok = all(abs(w[k] - round(c[K])) < 1 for k, K in [('tot','TOT_POP'),('white','WHITE'),('hisp','HISP'),('black','BLACK'),('asian','ASIAN'),('other','OTHER')])
    pct_ok = all(abs(w[k] - float(r[K])) <= 0.1 for k, K in [('ph','pct_hispanic'),('pw','pct_white_nh'),('pb','pct_black_nh'),('pa','pct_asian_nh')])
    no_ok = w['no'].strip() == r['number']
    if cnt_ok and pct_ok and no_ok: exact += 1
    else:
        defects.append((n, 'No.' if not no_ok else '', 'counts' if not cnt_ok else '', 'pct' if not pct_ok else '', w,
                        {k: round(c[k]) for k in ['TOT_POP','WHITE','HISP','BLACK','ASIAN','OTHER']}, {k: r[k] for k in ['number','pct_hispanic','pct_white_nh','pct_black_nh','pct_asian_nh']}))
print(f'rows matching CMAP on all six counts, all four percentages and the No. column: {exact} of {len(wiki)}')
for d in defects: print('DEFECT', d)
