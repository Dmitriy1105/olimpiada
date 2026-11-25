import sys
import heapq
from array import array


def build_components(rows, w, h):
    """Connected components of free cells via horizontal runs and DSU."""
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
            while i < len(prev_s) and j < len(starts):
                ps, pe = prev_s[i], prev_e[i]
                cs, ce = starts[j], ends[j]
                if pe < cs:
                    i += 1
                elif ce < ps:
                    j += 1
                else:
                    a, b = prev_id[i], ids[j]
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

    comp = array('i', [0]) * (w * h)

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


def solve():
    read = sys.stdin.buffer.readline
    first = read()
    if not first:
        return
    w, h = map(int, first.split())
    rows = [read().rstrip(b'\r\n') for _ in range(h)]
    wall_count = sum(row.count(b'#') for row in rows)
    use_astar = wall_count * 200 >= w * h

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
    stride = w + 2
    free_pad = None
    padded_size = 0
    dist_a = dist_b = seen_a = seen_b = None
    stamp = 0

    def ensure_bfs_workspace():
        nonlocal free_pad, padded_size, dist_a, dist_b, seen_a, seen_b
        if free_pad is not None:
            return
        free_pad = bytearray((h + 2) * stride)
        trans_free = bytes.maketrans(b'.#', b'\x01\x00')
        for yy, row in enumerate(rows):
            base = (yy + 1) * stride + 1
            free_pad[base:base + w] = row.translate(trans_free)
        padded_size = len(free_pad)
        dist_a = array('I', [0]) * padded_size
        seen_a = bytearray(padded_size)
        if not use_astar:
            dist_b = array('I', [0]) * padded_size
            seen_b = bytearray(padded_size)

    cache = {}
    cache_limit = 8192
    free_rows = None
    free_cols = None
    row_trans = bytes.maketrans(b'.#', b'10')
    col_trans = bytes.maketrans(b'\x00\x01', b'01')

    def exact_astar(s, t, sx, sy, tx, ty):
        nonlocal stamp
        ensure_bfs_workspace()
        stamp += 1
        if stamp == 256:
            seen_a[:] = b'\x00' * padded_size
            stamp = 1

        sp = (sy + 1) * stride + sx + 1
        tp = (ty + 1) * stride + tx + 1
        st = stamp
        sa, da, free = seen_a, dist_a, free_pad
        sa[sp] = st
        da[sp] = 0
        man = abs(sx - tx) + abs(sy - ty)
        heap = [(man, 0, sp)]

        while heap:
            fscore, neg_g, u = heapq.heappop(heap)
            g = -neg_g
            if sa[u] != st or da[u] != g:
                continue
            if u == tp:
                return g

            x = u % stride - 1
            y = u // stride - 1
            hcur = fscore - g
            ng = g + 1

            v = u - 1
            if free[v] and (sa[v] != st or ng < da[v]):
                sa[v] = st
                da[v] = ng
                hh = hcur + (1 if x <= tx else -1)
                heapq.heappush(heap, (ng + hh, -ng, v))

            v = u + 1
            if free[v] and (sa[v] != st or ng < da[v]):
                sa[v] = st
                da[v] = ng
                hh = hcur + (-1 if x < tx else 1)
                heapq.heappush(heap, (ng + hh, -ng, v))

            v = u - stride
            if free[v] and (sa[v] != st or ng < da[v]):
                sa[v] = st
                da[v] = ng
                hh = hcur + (1 if y <= ty else -1)
                heapq.heappush(heap, (ng + hh, -ng, v))

            v = u + stride
            if free[v] and (sa[v] != st or ng < da[v]):
                sa[v] = st
                da[v] = ng
                hh = hcur + (-1 if y < ty else 1)
                heapq.heappush(heap, (ng + hh, -ng, v))

        return -1

    def monotone_exists(sx, sy, tx, ty, dx, dy):
        nonlocal free_rows, free_cols
        row_cost = (dy + 1) * dx.bit_length()
        col_cost = (dx + 1) * dy.bit_length()

        if row_cost <= col_cost:
            if row_cost > 4096:
                return False
            if free_rows is None:
                free_rows = [int(row.translate(row_trans), 2) for row in rows]
            xlo = min(sx, tx)
            xhi = max(sx, tx)
            width = dx + 1
            rect = ((1 << width) - 1) << (w - 1 - xhi)
            target_bit = 1 << (w - 1 - tx)
            reachable = 1 << (w - 1 - sx)
            move_right = sx < tx
            ystep = 1 if sy < ty else -1
            y = sy
            while True:
                mask = free_rows[y] & rect
                r = reachable & mask
                shift = 1
                if move_right:
                    good = mask << 1
                    while shift < width:
                        r |= (r & good) >> shift
                        good &= good << shift
                        shift <<= 1
                else:
                    good = mask >> 1
                    while shift < width:
                        r |= (r & good) << shift
                        good &= good >> shift
                        shift <<= 1
                if y == ty:
                    return bool(r & target_bit)
                reachable = r
                y += ystep

        if col_cost > 4096:
            return False
        if free_cols is None:
            free_cols = [int(bytes(col).translate(col_trans), 2) for col in columns]
        ylo = min(sy, ty)
        yhi = max(sy, ty)
        height = dy + 1
        rect = ((1 << height) - 1) << (h - 1 - yhi)
        target_bit = 1 << (h - 1 - ty)
        reachable = 1 << (h - 1 - sy)
        move_down = sy < ty
        xstep = 1 if sx < tx else -1
        x = sx
        while True:
            mask = free_cols[x] & rect
            r = reachable & mask
            shift = 1
            if move_down:
                good = mask << 1
                while shift < height:
                    r |= (r & good) >> shift
                    good &= good << shift
                    shift <<= 1
            else:
                good = mask >> 1
                while shift < height:
                    r |= (r & good) << shift
                    good &= good >> shift
                    shift <<= 1
            if x == tx:
                return bool(r & target_bit)
            reachable = r
            x += xstep

    def exact_bidir(s, t):
        nonlocal stamp
        ensure_bfs_workspace()
        stamp += 1
        if stamp == 256:
            seen_a[:] = b'\x00' * padded_size
            if seen_b is not None:
                seen_b[:] = b'\x00' * padded_size
            stamp = 1

        s = (s // w + 1) * stride + (s % w) + 1
        t = (t // w + 1) * stride + (t % w) + 1
        sa, sb, da, db, st = seen_a, seen_b, dist_a, dist_b, stamp
        sa[s] = st
        sb[t] = st
        da[s] = db[t] = 0
        qa, qb = [s], [t]
        ia = ib = 0
        enda = endb = 1
        deptha = depthb = 0
        best = 1 << 30
        f = free_pad
        off = stride

        while ia < len(qa) and ib < len(qb):
            if enda - ia <= endb - ib:
                nd = deptha + 1
                for qi in range(ia, enda):
                    u = qa[qi]
                    v = u - 1
                    if f[v] and sa[v] != st:
                        sa[v] = st; da[v] = nd; qa.append(v)
                        if sb[v] == st:
                            z = nd + db[v]
                            if z < best: best = z
                    v = u + 1
                    if f[v] and sa[v] != st:
                        sa[v] = st; da[v] = nd; qa.append(v)
                        if sb[v] == st:
                            z = nd + db[v]
                            if z < best: best = z
                    v = u - off
                    if f[v] and sa[v] != st:
                        sa[v] = st; da[v] = nd; qa.append(v)
                        if sb[v] == st:
                            z = nd + db[v]
                            if z < best: best = z
                    v = u + off
                    if f[v] and sa[v] != st:
                        sa[v] = st; da[v] = nd; qa.append(v)
                        if sb[v] == st:
                            z = nd + db[v]
                            if z < best: best = z
                ia = enda
                enda = len(qa)
                deptha = nd
            else:
                nd = depthb + 1
                for qi in range(ib, endb):
                    u = qb[qi]
                    v = u - 1
                    if f[v] and sb[v] != st:
                        sb[v] = st; db[v] = nd; qb.append(v)
                        if sa[v] == st:
                            z = nd + da[v]
                            if z < best: best = z
                    v = u + 1
                    if f[v] and sb[v] != st:
                        sb[v] = st; db[v] = nd; qb.append(v)
                        if sa[v] == st:
                            z = nd + da[v]
                            if z < best: best = z
                    v = u - off
                    if f[v] and sb[v] != st:
                        sb[v] = st; db[v] = nd; qb.append(v)
                        if sa[v] == st:
                            z = nd + da[v]
                            if z < best: best = z
                    v = u + off
                    if f[v] and sb[v] != st:
                        sb[v] = st; db[v] = nd; qb.append(v)
                        if sa[v] == st:
                            z = nd + da[v]
                            if z < best: best = z
                ib = endb
                endb = len(qb)
                depthb = nd

            if best <= deptha + depthb:
                return best
        return best if best < (1 << 30) else -1

    out = sys.stdout.buffer
    fast_bytes = [str(i).encode() + b'\n' for i in range(w + h)]
    minus_one = b'-1\n'

    while True:
        line = read()
        if not line:
            break
        if not line.strip():
            continue
        sx, sy, tx, ty = map(int, line.split())
        s = sy * w + sx
        t = ty * w + tx

        if s == t:
            ans = 0
        elif comp[s] != comp[t]:
            ans = -1
        else:
            dx = abs(sx - tx)
            dy = abs(sy - ty)
            man = dx + dy
            xlo, xhi = (sx, tx) if sx <= tx else (tx, sx)
            ylo, yhi = (sy, ty) if sy <= ty else (ty, sy)

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
                elif dx and dy and min((dy + 1) * dx.bit_length(), (dx + 1) * dy.bit_length()) <= 4096 and monotone_exists(sx, sy, tx, ty, dx, dy):
                    ans = man
                else:
                    key = (s << 22) | t if s < t else (t << 22) | s
                    cached = cache.get(key)
                    if cached is not None:
                        ans = cached
                    else:
                        ans = exact_astar(s, t, sx, sy, tx, ty) if use_astar else exact_bidir(s, t)
                        if len(cache) >= cache_limit:
                            cache.clear()
                        cache[key] = ans

        out.write(minus_one if ans < 0 else (fast_bytes[ans] if ans < len(fast_bytes) else str(ans).encode() + b'\n'))
        out.flush()


if __name__ == '__main__':
    solve()
