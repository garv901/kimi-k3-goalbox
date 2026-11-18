"""flag_audit: list every preprocessor switch in the candidate's CUDA/C++ sources, its state in the SHIPPED build,
and flag experiment code that NOTES.md calls off / reverted / no gain / slower but that is still compiled in.

  ./flag_audit                 audit the workspace (exit 1 if there are findings)
  ./flag_audit --flags "-DX"   audit as if the build used these -D/-U flags (overrides what is detected)
  ./flag_audit --env           also honour build-flag environment variables set in this shell (the grader does not)
  ./flag_audit --all           print every conditional, not only the experiment-like ones
  ./flag_audit --self-test

Shipped flags are read from the Python build code: string constants in an nvcc / cpp_extension command list
(lists containing 'nvcc', '-gencode', '-arch', or extra_cuda_cflags=...), including the DEFAULT of any
os.environ.get(...) used inside that list (the grader runs without your shell's environment).
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SRC_EXT = {'.cu', '.cuh', '.h', '.hpp', '.hh', '.cpp', '.cc', '.c', '.inl'}
SKIP_DIRS = {'.git', 'reports', 'build', '__pycache__', 'reference', 'megakernel_skills', '.a1', '.claude', '.ss',
             'node_modules', '.cache', 'TASK.d'}
OVERLAY_FILES = {'phase_table.cuh', 'timeline_probe.cuh'}  # tool headers, not candidate switches
NEG = re.compile(r"no gain|not faster|no improvement|no speed ?up|no benefit|no effect|didn'?t help|did not help|"
                 r"slower|worse|regress|revert|backed out|rolled back|abandon|dropped|disabled|turned off|"
                 r"switched off|\boff\b|neutral|not used|unused|dead code|no win|lost|hurts?", re.I)
STOP = set('''the and for with from into this that when then than else only once each per all any are was were not
off out use used using via its it is be to of in on at by or an as if no yes do does did can may just more less
same new old one two three four tile tiles row rows col cols block blocks warp warps thread threads kernel phase
phases const auto void int float uint32 uint64 uint8 size_t unsigned return true false bool static inline device
global forceinline shared template typename struct pragma unroll define endif ifdef ifndef include ptr idx len num
data buffer value values first last next prev count base step steps loop sum max min mean tried attempt ms us gb
tb slower faster gain reverted neutral wrong'''.split())


def stem(w: str) -> str:
    w = w.lower()
    for suf in ('ing', 'es', 'ed', 's'):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def words(text: str) -> list[str]:
    out = []
    for tok in re.findall(r'[A-Za-z][A-Za-z0-9]*', re.sub(r'([a-z])([A-Z])', r'\1 \2', text.replace('_', ' '))):
        out.append(stem(tok))
    return out


# ------------------------------------------------------------------------------------------ build flags
def detect_flags(ws: Path, honour_env: bool) -> tuple[dict, list[str]]:
    """Return ({macro: value or None for -U}, provenance notes) from the Python build code."""
    defs, notes = {}, []
    for py in sorted(ws.glob('*.py')) + sorted(p for p in ws.glob('*/*.py') if p.parts[-2] not in SKIP_DIRS):
        try:
            tree = ast.parse(py.read_text(errors='replace'))
        except SyntaxError:
            continue
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = node.value.value

        def strval(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                return n.value
            if isinstance(n, ast.Name) and n.id in consts:
                return consts[n.id]
            return None

        def env_get(n):
            # os.environ.get('X', D) / os.getenv('X', D) -> (X, D)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('get', 'getenv') and n.args:
                name = strval(n.args[0])
                if name is None:
                    return None
                d = strval(n.args[1]) if len(n.args) > 1 else ''
                return name, d if d is not None else ''
            return None

        lists = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)):
                s = [strval(e) for e in node.elts]
                if any(x and (x.endswith('nvcc') or x.startswith('-gencode') or x.startswith('-arch') or x == '-O3'
                              or x.startswith('--generate-code')) for x in s):
                    lists.append(node)
            if isinstance(node, ast.keyword) and node.arg in ('extra_cuda_cflags', 'extra_cflags', 'extra_compile_args'):
                lists.append(node.value)
        for L in lists:
            for n in ast.walk(L):
                toks = []
                eg = env_get(n)
                if eg:
                    name, d = eg
                    val = os.environ.get(name) if honour_env and name in os.environ else d
                    src = 'env' if honour_env and name in os.environ else 'default of env ' + name
                    toks = [(t, f'{py.name}: {src}') for t in shlex.split(val)]
                else:
                    s = strval(n)
                    if s is not None:
                        toks = [(t, f'{py.name}: literal') for t in shlex.split(s) if t.startswith(('-D', '-U'))] if ' ' in s \
                            else [(s, f'{py.name}: literal')]
                for t, src in toks:
                    if t.startswith('-D') and len(t) > 2:
                        k, _, v = t[2:].partition('=')
                        defs[k] = v if _ else '1'
                        notes.append(f'{t}  ({src}, line {getattr(n, "lineno", "?")})')
                    elif t.startswith('-U') and len(t) > 2:
                        defs[t[2:]] = None
                        notes.append(f'{t}  ({src})')
    return defs, list(dict.fromkeys(notes))


def build_sources(ws: Path) -> list[Path] | None:
    """Source files named in the Python build code (e.g. 'mk.cu') plus their local includes; None if none found."""
    found = []
    for py in sorted(ws.glob('*.py')):
        t = py.read_text(errors='replace')
        if not re.search(r'nvcc|cpp_extension|load_inline|CUDAExtension', t):
            continue
        for name in re.findall(r"['\"]([\w./-]+\.(?:cu|cuh|cpp|cc|h|hpp))['\"]", t):
            p = (py.parent / name)
            if p.is_file() and p.resolve() not in found and p.name not in OVERLAY_FILES:
                found.append(p.resolve())
    if not found:
        return None
    todo = list(found)
    while todo:
        f = todo.pop()
        for inc in re.findall(r'^\s*#\s*include\s*"([^"]+)"', f.read_text(errors='replace'), re.M):
            q = (f.parent / inc).resolve()
            if q.is_file() and q not in found and q.name not in OVERLAY_FILES:
                found.append(q)
                todo.append(q)
    return sorted(found)


def parse_flags(s: str) -> dict:
    d = {}
    for t in shlex.split(s):
        if t.startswith('-D'):
            k, _, v = t[2:].partition('=')
            d[k] = v if _ else '1'
        elif t.startswith('-U'):
            d[t[2:]] = None
    return d


# ------------------------------------------------------------------------------------------ preprocessor
def logical_lines(text: str):
    """(line number, logical line) with backslash continuations joined and /* */ comments blanked."""
    text = re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text, flags=re.S)
    phys = text.split('\n')
    i = 0
    while i < len(phys):
        start, s = i + 1, phys[i]
        while s.endswith('\\') and i + 1 < len(phys):
            i += 1
            s = s[:-1] + ' ' + phys[i]
        yield start, s
        i += 1


def eval_cond(expr: str, macros: dict):
    e = re.sub(r'//.*', '', expr)
    e = re.sub(r'defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)',
               lambda m: ' 1 ' if (m.group(1) or m.group(2)) in macros and macros[m.group(1) or m.group(2)] is not None else ' 0 ', e)
    for _ in range(8):
        new = re.sub(r'\b([A-Za-z_]\w*)\b', lambda m: f' ({macros[m.group(1)]}) ' if macros.get(m.group(1)) not in (None, '')
                     and m.group(1) not in ('and', 'or', 'not') else (' 1 ' if macros.get(m.group(1)) == '' else m.group(0)), e)
        if new == e:
            break
        e = new
    e = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)[uUlL]+\b', r'\1', e)
    e = re.sub(r'\b[A-Za-z_]\w*\b', ' 0 ', e)          # unknown identifiers are 0 in #if
    e = e.replace('&&', ' and ').replace('||', ' or ')
    e = re.sub(r'!(?!=)', ' not ', e)
    e = re.sub(r'(?<![/])/(?![/])', '//', e)
    try:
        return bool(eval(e, {'__builtins__': {}}, {}))
    except Exception:
        return None


PLATFORM = re.compile(r'^(__\w+|_WIN32|_MSC_VER|NDEBUG|CUDA_VERSION|CUDART_VERSION|TORCH_\w+|C10_\w+|USE_ROCM)$')


class Audit:
    def __init__(self, ws: Path, files: list[Path], flags: dict):
        self.ws, self.files, self.flags = ws, files, flags
        self.groups = []            # conditional groups
        self.seen_inc = set()
        self.all_lines = []         # (file, text) for term frequencies
        self.defs = {}              # function name -> (file, line index) of definition
        self.text = {}
        for f in files:
            t = f.read_text(errors='replace')
            self.text[f] = t.split('\n')
            self.all_lines += t.split('\n')
            for i, l in enumerate(self.text[f]):
                m = re.match(r'\s*(?:template\s*<[^>]*>\s*)?(?:static\s+|inline\s+|__device__\s+|__host__\s+|__global__\s+|__forceinline__\s+|constexpr\s+)+[\w:<>,\s\*&]*?\b(\w+)\s*\(', l)
                if m:
                    self.defs.setdefault(m.group(1), (f, i))

    def run(self):
        included = set()
        for f in self.files:
            for _, l in logical_lines(f.read_text(errors='replace')):
                m = re.match(r'\s*#\s*include\s*"([^"]+)"', l)
                if m:
                    included.add((f.parent / m.group(1)).resolve())
        roots = [f for f in self.files if f.resolve() not in included] or self.files
        for f in roots:
            self.process(f, dict(self.flags_defined()), [])
        for f in self.files:  # headers never reached from a root
            if f.resolve() not in self.seen_inc and f not in roots:
                self.process(f, dict(self.flags_defined()), [])

    def flags_defined(self):
        return {k: v for k, v in self.flags.items()}

    def process(self, f: Path, macros: dict, chain: list):
        if f.resolve() in chain:
            return
        self.seen_inc.add(f.resolve())
        stack = []  # each: dict(group, parent_active, taken, cur_active)
        active = True
        lines = list(logical_lines(f.read_text(errors='replace')))
        for idx, (ln, s) in enumerate(lines):
            m = re.match(r'\s*#\s*(\w+)\s*(.*)', s)
            if not m:
                if stack:
                    stack[-1]['group']['branches'][-1]['body'].append(s)
                continue
            d, rest = m.group(1), m.group(2).strip()
            if d in ('if', 'ifdef', 'ifndef'):
                if d == 'ifdef':
                    nm = rest.split()[0]
                    val = nm in macros and macros[nm] is not None
                    names = [nm]
                elif d == 'ifndef':
                    nm = rest.split()[0]
                    val = not (nm in macros and macros[nm] is not None)
                    names = [nm]
                else:
                    val = eval_cond(rest, macros)
                    names = sorted(set(re.findall(r'\b[A-Za-z_]\w*\b', re.sub(r'\bdefined\b', '', rest))))
                g = dict(file=str(f.relative_to(self.ws)) if f.is_relative_to(self.ws) else str(f), line=ln,
                         directive=f'#{d} {rest}'.strip(), macros=names, branches=[], parent_active=active)
                g['branches'].append(dict(line=ln, cond=f'#{d} {rest}', active=bool(active and val), unknown=val is None, body=[]))
                stack.append(dict(group=g, taken=bool(val), parent=active))
                active = bool(active and val)
                self.groups.append(g)
            elif d in ('elif', 'else') and stack:
                top = stack[-1]
                if d == 'elif':
                    val = eval_cond(rest, macros)
                    top['group']['macros'] = sorted(set(top['group']['macros']) | set(re.findall(r'\b[A-Za-z_]\w*\b', re.sub(r'\bdefined\b', '', rest))))
                else:
                    val = True
                br_active = top['parent'] and not top['taken'] and bool(val)
                top['taken'] = top['taken'] or bool(val)
                top['group']['branches'].append(dict(line=ln, cond=f'#{d} {rest}'.strip(), active=br_active, unknown=val is None, body=[]))
                active = br_active
            elif d == 'endif' and stack:
                top = stack.pop()
                active = top['parent']
                if stack:
                    stack[-1]['group']['branches'][-1]['body'].append(s)
            else:
                if stack:
                    stack[-1]['group']['branches'][-1]['body'].append(s)
                if not active:
                    continue
                if d == 'define':
                    mm = re.match(r'(\w+)(\([^)]*\))?\s*(.*)', rest)
                    if mm:
                        macros[mm.group(1)] = mm.group(3).strip() if not mm.group(2) else '1'
                elif d == 'undef':
                    macros[rest.split()[0]] = None
                elif d == 'include':
                    mm = re.match(r'"([^"]+)"', rest)
                    if mm:
                        inc = (f.parent / mm.group(1))
                        if inc.exists() and inc.name not in OVERLAY_FILES:
                            self.process(inc, macros, chain + [f.resolve()])

    # ---------------------------------------------------------------------- classification + notes matching
    def classify(self, g):
        b0 = g['branches'][0]
        body = [l for l in b0['body'] if l.strip() and not l.strip().startswith('//')]
        if all(PLATFORM.match(m) for m in g['macros']) and g['macros']:
            return 'platform'
        if g['directive'].startswith('#ifndef') and body and re.match(r'\s*#\s*define\s+' + re.escape(g['macros'][0]) + r'\b', body[0]):
            if re.match(r'\s*#\s*define\s+\w+\s*(//.*)?$', body[0]):
                return 'include-guard'
            if all(re.match(r'\s*#\s*define\b', l) for l in body):
                return 'default'
        if len(g['branches']) > 1:
            return 'variant'
        return 'optional'

    def term_profile(self, g, br):
        terms = collections.Counter()
        text = '\n'.join(br['body'])
        for m in g['macros']:
            terms.update(words(m))
        terms.update(words(text))
        for fn in set(re.findall(r'\b([A-Za-z_]\w*)\s*\(', text)):
            if fn in self.defs:
                f, i = self.defs[fn]
                L = self.text[f]
                ctx = '\n'.join(l for l in L[max(0, i - 3): i] if '//' in l) + '\n' + '\n'.join(L[i: i + 12])
                terms.update(words(ctx))
        return terms

    def notes_clauses(self):
        out = []
        for name in ('NOTES.md',):
            p = self.ws / name
            if not p.exists():
                continue
            for ln, line in enumerate(p.read_text(errors='replace').split('\n'), 1):
                for clause in re.split(r';|\|', line):
                    if NEG.search(clause) and len(clause.strip()) > 3:
                        out.append(dict(line=ln, text=clause.strip()))
        return out

    def doc_freq(self):
        df = collections.Counter()
        for l in self.all_lines:
            df.update(set(words(l)))
        return df, max(1, len(self.all_lines))


def audit(ws: Path, flags_override: str | None, honour_env: bool, files: list[Path] | None):
    if files is None:
        built = build_sources(ws)
        if built:
            files = built
    if files is None:
        files = sorted(p for p in ws.rglob('*') if p.suffix in SRC_EXT and p.is_file()
                       and not (set(p.relative_to(ws).parts[:-1]) & SKIP_DIRS) and p.name not in OVERLAY_FILES
                       and not any(part.startswith('.') for part in p.relative_to(ws).parts[:-1]))
    if flags_override is not None:
        flags, prov = parse_flags(flags_override), [f'--flags {flags_override!r}']
    else:
        flags, prov = detect_flags(ws, honour_env)
    A = Audit(ws, files, flags)
    A.run()
    clauses = A.notes_clauses()
    df, nlines = A.doc_freq()
    switches, findings, seen = [], [], {}
    for g in A.groups:
        kind = A.classify(g)
        state = ', '.join(('ON' if b['active'] else ('?' if b['unknown'] else 'off')) + f" ({b['cond']}:{b['line']})"
                          for b in g['branches'])
        vals = {m: (A.flags.get(m, '<src/undefined>') if m in A.flags else 'undefined/src') for m in g['macros']}
        sw = dict(file=g['file'], line=g['line'], directive=g['directive'], macros=g['macros'], kind=kind,
                  parent_active=g['parent_active'], branches=[dict(line=b['line'], cond=b['cond'], active=b['active'],
                                                                     unknown=b['unknown']) for b in g['branches']],
                  flag_values={m: A.flags.get(m) for m in g['macros'] if m in A.flags}, state=state, notes=[])
        if kind in ('optional', 'variant') and g['parent_active']:
            for b in g['branches']:
                if not b['active'] or (kind == 'variant' and b is not g['branches'][0] and False):
                    continue
                if kind == 'variant':
                    # a variant's active branch is "the" implementation; it is only suspicious when its guard macro is
                    # an explicit opt-in (#ifdef X / #if X with X defined)
                    if not (b['cond'].startswith('#ifdef') or (b['cond'].startswith('#if ') and any(m in A.flags for m in g['macros']))):
                        continue
                prof = A.term_profile(g, b)
                key = (tuple(g['macros']), re.sub(r'\s+', '', '\n'.join(b['body']))[:4000])
                import math
                cand = []
                for c in clauses:
                    cw = set(words(c['text']))
                    lit = [m for m in g['macros'] if re.search(r'\b' + re.escape(m) + r'\b', c['text'])]
                    shared = sorted(t for t in cw & set(prof) if t not in STOP and len(t) >= 4 and df[t] <= max(5, 0.015 * nlines))
                    score = 100.0 * len(lit) + sum(math.log(nlines / max(1, df[t])) for t in shared)
                    if lit or shared:
                        cand.append((score, dict(notes_line=c['line'], clause=c['text'][:200], macro_named=lit,
                                                 shared_terms=shared, score=round(score, 2))))
                if cand:
                    best = max(x[0] for x in cand)
                    sw['notes'] = [d for sc, d in sorted(cand, key=lambda x: -x[0]) if sc >= 0.5 * best]
                if sw['notes']:
                    if key in seen:
                        seen[key]['also_at'].append(f"{g['file']}:{g['line']}")
                    else:
                        sw['also_at'] = []
                        seen[key] = sw
                        findings.append(sw)
        switches.append(sw)
    return dict(files=[str(f.relative_to(ws)) for f in files], flags=flags, flag_sources=prov, switches=switches,
                findings=findings)


def print_report(R, show_all: bool):
    P = print
    P(f"flag_audit: {len(R['files'])} source files, {len(R['switches'])} conditionals")
    P('shipped build flags: ' + (' '.join(f'-D{k}' + (f'={v}' if v not in ('1', None) else '') if v is not None else f'-U{k}'
                                          for k, v in R['flags'].items()) or '(none detected)'))
    for s in R['flag_sources']:
        P(f'  from {s}')
    P('')
    P(f'{"file:line":28s} {"kind":13s} {"directive":38s} shipped state')
    for s in R['switches']:
        if not show_all and s['kind'] in ('include-guard', 'platform'):
            continue
        if not show_all and not s['parent_active']:
            continue
        st = ' / '.join(('ON' if b['active'] else ('?' if b['unknown'] else 'off')) for b in s['branches'])
        if s['kind'] == 'default':
            st = 'default value used' if s['branches'][0]['active'] else f"overridden ({s['flag_values']})"
        P(f"{(s['file'] + ':' + str(s['line']))[:28]:28s} {s['kind']:13s} {s['directive'][:38]:38s} {st}")
    P('')
    if R['findings']:
        P(f"FINDINGS ({len(R['findings'])}): code that NOTES.md describes as off/reverted/no gain/slower is COMPILED IN the shipped build")
        for s in R['findings']:
            P(f"  {s['file']}:{s['line']} {s['directive']}  [{s['kind']}]" + (f"  (same block also at {', '.join(s['also_at'][:4])})" if s.get('also_at') else ''))
            for n in s['notes'][:4]:
                why = ('names ' + ','.join(n['macro_named'])) if n['macro_named'] else ('shares terms ' + ','.join(n['shared_terms'][:6]))
                P(f"     NOTES.md:{n['notes_line']}: \"{n['clause'][:110]}\" ({why})")
        P('Fix: remove the dead path, or make the shipped build define/undefine the switch, then re-check; if the match '
          'is spurious, say so in NOTES.md (e.g. "PF path: kept on purpose") and ignore it.')
    else:
        P('no findings: nothing NOTES.md calls off/reverted/no gain is compiled into the shipped build')


def self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        (ws / 'k.cu').write_text('''#include "p.h"
#ifndef UNROLL
#define UNROLL 2
#endif
// L2 prefetch of upcoming rows
__device__ __forceinline__ void pf_rows(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
__global__ void k(const float* x) {
#ifndef NO_PF
  pf_rows(x);
#endif
#ifdef FAST_DECODE
  int a = 1;
#else
  int a = 2;
#endif
#if USE_SPLITK && !defined(NO_PF)
  int b = 3;
#endif
}
''')
        (ws / 'p.h').write_text('#ifndef P_H\n#define P_H\n#define USE_SPLITK 1\n#endif\n')
        (ws / 'NOTES.md').write_text('| - | (tried) L2 prefetch of upcoming tiles: no gain; split-K is fine |\n')
        build = "import os\ncmd = ['/usr/local/cuda/bin/nvcc', '-O3', *os.environ.get('MKF', '%s').split(), 'k.cu']\n"
        (ws / 'host.py').write_text(build % '')
        R = audit(ws, None, False, None)
        kinds = {s['directive']: s['kind'] for s in R['switches']}
        assert kinds['#ifndef UNROLL'] == 'default', kinds
        assert kinds['#ifndef P_H'] == 'include-guard', kinds
        assert kinds['#ifdef FAST_DECODE'] == 'variant', kinds
        assert kinds['#ifndef NO_PF'] == 'optional', kinds
        sw = {s['directive']: s for s in R['switches']}
        assert sw['#if USE_SPLITK && !defined(NO_PF)']['branches'][0]['active'] is True
        assert len(R['findings']) == 1 and R['findings'][0]['directive'] == '#ifndef NO_PF', R['findings']
        assert 'prefetch' in R['findings'][0]['notes'][0]['shared_terms'], R['findings'][0]['notes']
        (ws / 'host.py').write_text(build % '-DNO_PF')
        R = audit(ws, None, False, None)
        assert R['flags'] == {'NO_PF': '1'}, R['flags']
        assert not R['findings'], R['findings']
        sw = {s['directive']: s for s in R['switches']}
        assert sw['#if USE_SPLITK && !defined(NO_PF)']['branches'][0]['active'] is False
        R = audit(ws, '', False, None)
        assert len(R['findings']) == 1
        print_report(R, False)
    print('flag_audit self-test: PASS')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--flags', help='audit with these nvcc -D/-U flags instead of the detected ones')
    ap.add_argument('--env', action='store_true', help='honour build-flag env vars of this shell')
    ap.add_argument('--all', action='store_true', help='also list include guards, platform checks and inactive nests')
    ap.add_argument('--files', nargs='*', type=Path, help='only these source files')
    ap.add_argument('--dir', type=Path, help='workspace root to audit (default: this workspace)')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    import common
    ws = (a.dir or common.WS).resolve()
    files = [f.resolve() for f in a.files] if a.files else None
    R = audit(ws, a.flags, a.env, files)
    print_report(R, a.all)
    out = ws / 'reports'
    if os.access(ws, os.W_OK):
        out.mkdir(exist_ok=True)
        p = out / f'flag_audit-{common.run_id()}.json'
        p.write_text(json.dumps(R, indent=1))
        print(f'wrote {p}')
    return 1 if R['findings'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
