import sys
import os
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

    row_obs = [
        int(row[::-1].translate(bytes.maketrans(b'.#', b'01')), 2)
        for row in rows
    ]
    col_obs = [
        int(bytes(col)[::-1].translate(bytes.maketrans(b'\x00\x01', b'01')), 2)
        for col in columns
    ]
    full_w = (1 << w) - 1
    full_h = (1 << h) - 1
    right_edge_w = 1 << (w - 1)
    bottom_edge_h = 1 << (h - 1)

    forced_east = [0] * h
    forced_west = [0] * h
    for y in range(h):
        cur_free = full_w ^ row_obs[y]
        if y:
            up_obs = row_obs[y - 1]
            up_free = full_w ^ up_obs
        else:
            up_obs = full_w
            up_free = 0
        if y + 1 < h:
            down_obs = row_obs[y + 1]
            down_free = full_w ^ down_obs
        else:
            down_obs = full_w
            down_free = 0

        behind_left_up = ((up_obs << 1) | 1) & full_w
        behind_left_down = ((down_obs << 1) | 1) & full_w
        behind_right_up = (up_obs >> 1) | right_edge_w
        behind_right_down = (down_obs >> 1) | right_edge_w

        forced_east[y] = cur_free & (
            (behind_left_up & up_free) |
            (behind_left_down & down_free)
        )
        forced_west[y] = cur_free & (
            (behind_right_up & up_free) |
            (behind_right_down & down_free)
        )

    forced_south = [0] * w
    forced_north = [0] * w
    for x in range(w):
        cur_free = full_h ^ col_obs[x]
        if x:
            left_obs = col_obs[x - 1]
            left_free = full_h ^ left_obs
        else:
            left_obs = full_h
            left_free = 0
        if x + 1 < w:
            right_obs = col_obs[x + 1]
            right_free = full_h ^ right_obs
        else:
            right_obs = full_h
            right_free = 0

        behind_up_left = ((left_obs << 1) | 1) & full_h
        behind_up_right = ((right_obs << 1) | 1) & full_h
        behind_down_left = (left_obs >> 1) | bottom_edge_h
        behind_down_right = (right_obs >> 1) | bottom_edge_h

        forced_south[x] = cur_free & (
            (behind_up_left & left_free) |
            (behind_up_right & right_free)
        )
        forced_north[x] = cur_free & (
            (behind_down_left & left_free) |
            (behind_down_right & right_free)
        )

    sys.setrecursionlimit(10000)

    def exact_jps(sx, sy, tx, ty):
        target = (tx, ty)

        @functools.lru_cache(maxsize=4096)
        def jump(x, y, dx, dy):
            start_x, start_y = x, y
            while True:
                nx = x + dx
                ny = y + dy

                if nx < 0 or nx >= w or ny < 0 or ny >= h or rows[ny][nx] == 35:
                    if (x != start_x or y != start_y):
                        if dx:
                            side_free = (
                                (y > 0 and rows[y - 1][x] == 46) or
                                (y + 1 < h and rows[y + 1][x] == 46)
                            )
                        else:
                            side_free = (
                                (x > 0 and rows[y][x - 1] == 46) or
                                (x + 1 < w and rows[y][x + 1] == 46)
                            )
                        if side_free:
                            return (x, y)
                    return None

                if nx == tx and ny == ty:
                    return target

                if dx == 1:
                    if (forced_east[ny] >> nx) & 1:
                        return (nx, ny)
                elif dx == -1:
                    if (forced_west[ny] >> nx) & 1:
                        return (nx, ny)
                elif dy == 1:
                    if (forced_south[nx] >> ny) & 1:
                        return (nx, ny)
                else:
                    if (forced_north[nx] >> ny) & 1:
                        return (nx, ny)

                if dx:
                    pd = 1 if ty > ny else (-1 if ty < ny else 0)
                    if pd and jump(nx, ny, 0, pd) is not None:
                        return (nx, ny)
                    if nx == tx:
                        return (nx, ny)
                else:
                    pd = 1 if tx > nx else (-1 if tx < nx else 0)
                    if pd and jump(nx, ny, pd, 0) is not None:
                        return (nx, ny)
                    if ny == ty:
                        return (nx, ny)

                x, y = nx, ny

        start = (sx, sy)
        heap = [(abs(sx - tx) + abs(sy - ty), 0, start)]
        dist = {start: 0}

        while heap:
            _, neg_g, point = heapq.heappop(heap)
            g = -neg_g
            if dist.get(point) != g:
                continue
            x, y = point
            if point == target:
                return g

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = jump(x, y, dx, dy)
                if nxt is None:
                    continue
                nx, ny = nxt
                ng = g + abs(nx - x) + abs(ny - y)
                if ng < dist.get(nxt, 1 << 60):
                    dist[nxt] = ng
                    hval = abs(nx - tx) + abs(ny - ty)
                    heapq.heappush(heap, (ng + hval, -ng, nxt))

        return -1

    cache = {}
    cache_limit = 8192
    free_rows = None
    free_cols = None
    row_trans = bytes.maketrans(b'.#', b'10')
    col_trans = bytes.maketrans(b'\x00\x01', b'10')

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

    write_answer = os.write
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
                        ans = exact_jps(sx, sy, tx, ty)
                        if ans == -1:
                            ans = exact_astar(s, t, sx, sy, tx, ty)
                        if len(cache) >= cache_limit:
                            cache.clear()
                        cache[key] = ans

        write_answer(1, minus_one if ans < 0 else (fast_bytes[ans] if ans < len(fast_bytes) else str(ans).encode() + b'\n'))


if __name__ == '__main__':
    solve()
