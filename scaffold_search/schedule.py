"""Random interleaved run schedule (waves of K that mix arms).

  python3.12 -m scaffold_search.schedule --name main --arms A0S=4,A1=4,A2=4,A3=4 --k 3 --seed 20261118 \
      [--prefix p] [--out schedule.json]

Run ids are <prefix><nn> in schedule order. Rejection sampling over random permutations: every
wave of K must contain at least 2 arms (when more than one arm is scheduled), and no arm may fill
a whole wave. The seed, the attempt count and the result are saved, so the schedule is
reproducible and can be committed to the `results` branch before any run starts.
"""
import argparse, json, random, sys, time


def make(arms, k, seed, prefix, max_tries=100000):
    runs = [a for a, n in arms.items() for _ in range(n)]
    rng = random.Random(seed)
    multi = len(arms) > 1
    for attempt in range(1, max_tries + 1):
        order = runs[:]
        rng.shuffle(order)
        waves = [order[i:i + k] for i in range(0, len(order), k)]
        if not multi or all(len(set(w)) >= min(2, len(w)) for w in waves):
            break
    else:
        raise SystemExit('no schedule satisfies the mixing rule')
    n = 0; out = []
    for w in waves:
        row = []
        for a in w:
            n += 1
            row.append({'run_id': f'{prefix}{n:02d}', 'arm': a})
        out.append(row)
    return {'seed': seed, 'k': k, 'arms': arms, 'attempts': attempt, 'waves': out,
            'order': [r for w in out for r in w], 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def parse_arms(s):
    out = {}
    for part in s.split(','):
        a, n = part.split('=')
        out[a.strip()] = int(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--arms', required=True)
    ap.add_argument('--k', type=int, default=3); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--prefix', default=None); ap.add_argument('--out')
    a = ap.parse_args()
    s = make(parse_arms(a.arms), a.k, a.seed, a.prefix or a.name[:1])
    s['name'] = a.name
    text = json.dumps(s, indent=1)
    if a.out:
        open(a.out, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    sys.exit(main())
