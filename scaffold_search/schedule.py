"""Random interleaved run schedule (waves of K that mix arms).

  python3.12 -m scaffold_search.schedule --name main --arms A0S=4,A1=4,A2=4,A3=4 --k 3 --seed 20261118 \
      [--prefix p] [--out schedule.json]

Run ids are <prefix><nn> in schedule order. Rejection sampling over random permutations: every
wave of K must contain at least 2 arms (when more than one arm is scheduled), and no arm may fill
a whole wave. The seed, the attempt count and the result are saved, so the schedule is
reproducible and can be committed to the `results` branch before any run starts.

--latin (TP4 protocol, K = number of arms = number of 4-GPU slots): every wave holds each arm once, and
wave position = slot (orchestrate --slots). Waves come in blocks of K that form a random Latin square
(each slot runs each arm once per block); a last partial block uses the first rows of a fresh square,
so no slot runs an arm twice more often than another.
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


def make_latin(arms, k, seed, prefix):
    names = sorted(arms)
    n = set(arms.values())
    assert len(n) == 1 and k == len(names), '--latin needs equal n per arm and K = number of arms'
    n = n.pop()
    rng = random.Random(seed)
    waves = []
    while len(waves) < n:
        rows, cols, syms = list(range(k)), list(range(k)), names[:]
        rng.shuffle(rows); rng.shuffle(cols); rng.shuffle(syms)
        square = [[syms[(rows[r] + cols[c]) % k] for c in range(k)] for r in range(k)]
        waves += square[:n - len(waves)]
    out, i = [], 0
    for w in waves:
        row = []
        for a in w:
            i += 1
            row.append({'run_id': f'{prefix}{i:02d}', 'arm': a})
        out.append(row)
    return {'seed': seed, 'k': k, 'arms': arms, 'latin': True, 'waves': out,
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
    ap.add_argument('--latin', action='store_true', help='blocked Latin-square schedule (see above)')
    a = ap.parse_args()
    s = (make_latin if a.latin else make)(parse_arms(a.arms), a.k, a.seed, a.prefix or a.name[:1])
    s['name'] = a.name
    text = json.dumps(s, indent=1)
    if a.out:
        open(a.out, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    sys.exit(main())
