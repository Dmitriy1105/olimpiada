import sys
from array import array


def build_components(rows, w, h):
    """Label 4-connected free components using horizontal runs + DSU."""
    parent = array('i')
    row_runs = []
    prev_s = prev_e = prev_id = None

    for row in rows:
        starts = array('H')
        ends = array('H')
        ids = array('i')
        pos = 0
        while True:
            s = row.find(b'.', pos)
            if s < 0:
                break
            wall = row.find(b'#', s)
            e = w if wall < 0 else wall
            node = len(parent)
            parent.append(-1)
            starts.append(s)
            ends.append(e - 1)
            ids.append(node)
            pos = e

        if prev_s is not None and starts:
            i = j = 0
            pn = len(prev_s)
            cn = len(starts)
            while i < pn and j < cn:
                ps = prev_s[i]
                pe = prev_e[i]
                cs = starts[j]
                ce = ends[j]
                if pe < cs:
                    i += 1
                elif ce < ps:
                    j += 1
                else:
                    a = prev_id[i]
                    b = ids[j]
                    while parent[a] >= 0:
                        p = parent[a]
                        if parent[p] >= 0:
                            parent[a] = parent[p]
                        a = p
                    while parent[b] >= 0:
                        p = parent[b]
                        if parent[p] >= 0:
                            parent[b] = parent[p]
                        b = p
                    if a != b:
                        if parent[a] > parent[b]:
                            a, b = b, a
                        parent[a] += parent[b]
                        parent[b] = a
                    if pe <= ce:
                        i += 1
                    if ce <= pe:
                        j += 1

        row_runs.append((starts, ends, ids))
        prev_s, prev_e, prev_id = starts, ends, ids

    size = w * h
    comp = array('i', [0]) * size

    def find(x):
        while parent[x] >= 0:
            p = parent[x]
            if parent[p] >= 0:
                parent[x] = parent[p]
            x = p
        return x

    for y, (starts, ends, ids) in enumerate(row_runs):
        base = y * w
        for j in range(len(starts)):
            root = find(ids[j]) + 1
            left = base + starts[j]
            right = base + ends[j] + 1
            length = right - left
            if length == 1:
                comp[left] = root
            elif length <= 4:
                for p in range(left, right):
                    comp[p] = root
            else:
                comp[left:right] = array('i', [root]) * length

    return comp


def parse_query(line):
    a, b, c, d = line.split()
    return int(a), int(b), int(c), int(d)


def solve():
    read = sys.stdin.buffer.readline
    first = read()
    if not first:
        return
    w, h = map(int, first.split())

    rows = [read().rstrip(b'\r\n') for _ in range(h)]

    # Transposed obstacle map: bytearray.find checks vertical segments in C.
    columns = [bytearray(h) for _ in range(w)]
    for y, row in enumerate(rows):
        pos = 0
        while True:
            x = row.find(b'#', pos)
            if x < 0:
                break
            columns[x][y] = 1
            pos = x + 1

    comp = build_components(rows, w, h)
    size = w * h

    # Reusable arrays for exact bidirectional BFS. Stamps avoid clearing the
    # full arrays after every query.
    dist_a = array('I', [0]) * size
    dist_b = array('I', [0]) * size
    seen_a = bytearray(size)
    seen_b = bytearray(size)
    stamp = 0

    # Cache only difficult pairs; memory remains bounded.
    cache = {}
    cache_limit = 8192

    def exact_bidir(s, t):
        nonlocal stamp
        stamp += 1
        if stamp == 256:
            seen_a[:] = b'\x00' * size
            seen_b[:] = b'\x00' * size
            stamp = 1

        sa = seen_a
        sb = seen_b
        da = dist_a
        db = dist_b
        st = stamp

        sa[s] = st
        sb[t] = st
        da[s] = 0
        db[t] = 0
        qa = [s]
        qb = [t]
        ia = ib = 0
        enda = endb = 1
        deptha = depthb = 0
        best = 1 << 30

        while ia < len(qa) and ib < len(qb):
            if enda - ia <= endb - ib:
                nd = deptha + 1
                for qi in range(ia, enda):
                    u = qa[qi]
                    x = u % w
                    if u >= w:
                        v = u - w
                        if comp[v] and sa[v] != st:
                            sa[v] = st
                            da[v] = nd
                            qa.append(v)
                            if sb[v] == st:
                                z = nd + db[v]
                                if z < best:
                                    best = z
                    if u + w < size:
                        v = u + w
                        if comp[v] and sa[v] != st:
                            sa[v] = st
                            da[v] = nd
                            qa.append(v)
                            if sb[v] == st:
                                z = nd + db[v]
                                if z < best:
                                    best = z
                    if x:
                        v = u - 1
                        if comp[v] and sa[v] != st:
                            sa[v] = st
                            da[v] = nd
                            qa.append(v)
                            if sb[v] == st:
                                z = nd + db[v]
                                if z < best:
                                    best = z
                    if x + 1 < w:
                        v = u + 1
                        if comp[v] and sa[v] != st:
                            sa[v] = st
                            da[v] = nd
                            qa.append(v)
                            if sb[v] == st:
                                z = nd + db[v]
                                if z < best:
                                    best = z
                ia = enda
                enda = len(qa)
                deptha = nd
            else:
                nd = depthb + 1
                for qi in range(ib, endb):
                    u = qb[qi]
                    x = u % w
                    if u >= w:
                        v = u - w
                        if comp[v] and sb[v] != st:
                            sb[v] = st
                            db[v] = nd
                            qb.append(v)
                            if sa[v] == st:
                                z = nd + da[v]
                                if z < best:
                                    best = z
                    if u + w < size:
                        v = u + w
                        if comp[v] and sb[v] != st:
                            sb[v] = st
                            db[v] = nd
                            qb.append(v)
                            if sa[v] == st:
                                z = nd + da[v]
                                if z < best:
                                    best = z
                    if x:
                        v = u - 1
                        if comp[v] and sb[v] != st:
                            sb[v] = st
                            db[v] = nd
                            qb.append(v)
                            if sa[v] == st:
                                z = nd + da[v]
                                if z < best:
                                    best = z
                    if x + 1 < w:
                        v = u + 1
                        if comp[v] and sb[v] != st:
                            sb[v] = st
                            db[v] = nd
                            qb.append(v)
                            if sa[v] == st:
                                z = nd + da[v]
                                if z < best:
                                    best = z
                ib = endb
                endb = len(qb)
                depthb = nd

            if best <= deptha + depthb:
                return best

        return best if best < (1 << 30) else -1

    out = sys.stdout.buffer
    while True:
        line = read()
        if not line:
            break
        if not line.strip():
            continue

        sx, sy, tx, ty = parse_query(line)
        s = sy * w + sx
        t = ty * w + tx

        if s == t:
            ans = 0
        else:
            man = abs(sx - tx) + abs(sy - ty)
            xlo, xhi = (sx, tx) if sx <= tx else (tx, sx)
            ylo, yhi = (sy, ty) if sy <= ty else (ty, sy)

            # If either L-shaped route is clear, Manhattan distance is exact.
            clear1 = (
                rows[sy].find(b'#', xlo, xhi + 1) < 0
                and columns[tx].find(b'\x01', ylo, yhi + 1) < 0
            )
            if clear1:
                ans = man
            else:
                clear2 = (
                    columns[sx].find(b'\x01', ylo, yhi + 1) < 0
                    and rows[ty].find(b'#', xlo, xhi + 1) < 0
                )
                if clear2:
                    ans = man
                elif comp[s] != comp[t]:
                    ans = -1
                else:
                    key = (s << 22) | t if s < t else (t << 22) | s
                    cached = cache.get(key)
                    if cached is not None:
                        ans = cached
                    else:
                        ans = exact_bidir(s, t)
                        if len(cache) >= cache_limit:
                            cache.clear()
                        cache[key] = ans

        out.write(str(ans).encode())
        out.write(b'\n')
        out.flush()


if __name__ == '__main__':
    solve()
