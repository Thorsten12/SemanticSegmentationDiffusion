#!/usr/bin/env python
import argparse, csv, statistics

def mean(xs): return sum(xs)/len(xs) if xs else float('nan')

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv_path'); a=p.parse_args()
    rows=list(csv.DictReader(open(a.csv_path)))
    for r in rows:
        for k in ('gt_area_fraction','proposal_dice','final_dice','iou'): r[k]=float(r[k])
        r['gt_touches_border']=int(r['gt_touches_border'])
    small=[r for r in rows if r['gt_area_fraction']<.25]
    mid=[r for r in rows if .25<=r['gt_area_fraction']<.55]
    large=[r for r in rows if r['gt_area_fraction']>=.55]
    border=[r for r in rows if r['gt_touches_border']]
    non=[r for r in rows if not r['gt_touches_border']]
    for name,g in [('all',rows),('small',small),('mid',mid),('large',large),('border',border),('nonborder',non)]:
        print(f"{name:9s} n={len(g):3d} proposal={mean([x['proposal_dice'] for x in g]):.4f} final={mean([x['final_dice'] for x in g]):.4f}")
    worst=sorted(rows,key=lambda x:x['final_dice'])[:10]
    print('\nworst 10:')
    for r in worst: print(int(r['index']),f"area={r['gt_area_fraction']:.3f}",f"border={r['gt_touches_border']}",f"proposal={r['proposal_dice']:.3f}",f"final={r['final_dice']:.3f}")
if __name__=='__main__': main()
