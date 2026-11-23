// OASIS benchmark harness: replays an event stream (insertions + kNN/radius/range queries) and, for every
// query, times all seven exact strategies (min of R runs), checks that they return the same answer, and records
// the features used by OASIS, UnIS-style, AI+R-style and learned-planner selectors.
// Build: g++ -O3 -march=native -std=c++17 -o bench bench.cpp
// Run:   ./bench points.bin stream.ev out.csv [R=3]
#include <bits/stdc++.h>
using namespace std;
using Clk = chrono::steady_clock;

static int D;
static vector<double> PT;
static inline const double* pt(int i) { return &PT[(size_t)i * D]; }
static inline double dist2(const double* a, const double* b) {
    double s = 0;
    for (int j = 0; j < D; j++) { double t = a[j] - b[j]; s += t * t; }
    return s;
}
struct Box { double lo[3], hi[3]; };
static inline void box_init(Box& b) { for (int j = 0; j < 3; j++) { b.lo[j] = 1e300; b.hi[j] = -1e300; } }
static inline void box_add(Box& b, const double* p) {
    for (int j = 0; j < D; j++) { if (p[j] < b.lo[j]) b.lo[j] = p[j]; if (p[j] > b.hi[j]) b.hi[j] = p[j]; }
}
static inline void box_merge(Box& b, const Box& o) {
    for (int j = 0; j < D; j++) { b.lo[j] = min(b.lo[j], o.lo[j]); b.hi[j] = max(b.hi[j], o.hi[j]); }
}
static inline double mind2(const double* q, const Box& b) {
    double s = 0;
    for (int j = 0; j < D; j++) {
        double t = 0;
        if (q[j] < b.lo[j]) t = b.lo[j] - q[j]; else if (q[j] > b.hi[j]) t = q[j] - b.hi[j];
        s += t * t;
    }
    return s;
}
static inline bool inter(const Box& a, const Box& b) {
    for (int j = 0; j < D; j++) if (a.hi[j] < b.lo[j] || a.lo[j] > b.hi[j]) return false;
    return true;
}
static inline bool inside(const double* p, const Box& b) {
    for (int j = 0; j < D; j++) if (p[j] < b.lo[j] || p[j] > b.hi[j]) return false;
    return true;
}
static inline double volume(const Box& b) {
    double a = 1;
    for (int j = 0; j < D; j++) a *= max(0.0, b.hi[j] - b.lo[j]);
    return a;
}

// ---------------------------------------------------------------- result containers
struct KH {  // max-heap of the k best (squared distance, id)
    vector<pair<double, int>> h; int k = 1;
    void init(int k_) { h.clear(); k = k_; }
    inline double kth() const { return (int)h.size() < k ? 1e300 : h.front().first; }
    inline void add(double d, int id) {
        if ((int)h.size() < k) { h.emplace_back(d, id); push_heap(h.begin(), h.end()); }
        else if (d < h.front().first) { pop_heap(h.begin(), h.end()); h.back() = {d, id}; push_heap(h.begin(), h.end()); }
    }
};
static KH H;
static vector<int> OUT;
using PQE = pair<double, int>;
static auto PQCMP = [](const PQE& a, const PQE& b) { return a.first > b.first; };

// ---------------------------------------------------------------- native KD-tree
struct KN { int l = -1, r = -1, dim = 0, depth = 0; double val = 0; Box bb; double c[3] = {0, 0, 0}; double rad = 0; vector<int> ids; };
struct KDT {
    vector<KN> nd; const int C = 32;
    void geom(int n, const vector<int>& ids, int lo, int hi) {
        Box bb; box_init(bb);
        for (int i = lo; i < hi; i++) box_add(bb, pt(ids[i]));
        double c[3] = {0, 0, 0}, r = 0;
        for (int j = 0; j < D; j++) c[j] = (bb.lo[j] + bb.hi[j]) / 2;
        for (int i = lo; i < hi; i++) r = max(r, dist2(c, pt(ids[i])));
        nd[n].bb = bb; memcpy(nd[n].c, c, sizeof c); nd[n].rad = sqrt(r);
    }
    static int widest(const Box& bb) {
        int dim = 0; double best = -1;
        for (int j = 0; j < D; j++) { double e = bb.hi[j] - bb.lo[j]; if (e > best) { best = e; dim = j; } }
        return dim;
    }
    int build(vector<int>& ids, int lo, int hi, int depth) {
        int me = nd.size(); nd.emplace_back(); nd[me].depth = depth;
        geom(me, ids, lo, hi);
        if (hi - lo <= C) { nd[me].ids.assign(ids.begin() + lo, ids.begin() + hi); return me; }
        int dim = widest(nd[me].bb), mid = (lo + hi) / 2;
        nth_element(ids.begin() + lo, ids.begin() + mid, ids.begin() + hi, [&](int a, int b) { return pt(a)[dim] < pt(b)[dim]; });
        double val = pt(ids[mid])[dim];
        int L = build(ids, lo, mid, depth + 1), R = build(ids, mid, hi, depth + 1);
        nd[me].dim = dim; nd[me].val = val; nd[me].l = L; nd[me].r = R;
        return me;
    }
    void insert(int id) {
        const double* p = pt(id); int n = 0;
        while (true) {
            KN& x = nd[n]; box_add(x.bb, p);
            double dd = sqrt(dist2(x.c, p)); if (dd > x.rad) x.rad = dd;
            if (x.l < 0) break;
            n = (p[x.dim] < x.val) ? x.l : x.r;
        }
        nd[n].ids.push_back(id);
        if ((int)nd[n].ids.size() > 2 * C) split(n);
    }
    void split(int n) {  // local leaf split: the tree becomes unbalanced after many insertions
        vector<int> ids = nd[n].ids; int depth = nd[n].depth;
        int dim = widest(nd[n].bb), mid = ids.size() / 2;
        nth_element(ids.begin(), ids.begin() + mid, ids.end(), [&](int a, int b) { return pt(a)[dim] < pt(b)[dim]; });
        double val = pt(ids[mid])[dim];
        int L = nd.size(); nd.emplace_back(); int R = nd.size(); nd.emplace_back();
        nd[L].ids.assign(ids.begin(), ids.begin() + mid); nd[R].ids.assign(ids.begin() + mid, ids.end());
        nd[L].depth = nd[R].depth = depth + 1;
        geom(L, nd[L].ids, 0, nd[L].ids.size()); geom(R, nd[R].ids, 0, nd[R].ids.size());
        nd[n].ids.clear(); nd[n].ids.shrink_to_fit();
        nd[n].l = L; nd[n].r = R; nd[n].dim = dim; nd[n].val = val;
    }
    inline double lbM(const double* q, int n) const { return mind2(q, nd[n].bb); }
    inline double lbB(const double* q, int n) const { double t = sqrt(dist2(q, nd[n].c)) - nd[n].rad; return t > 0 ? t * t : 0; }
    template <bool BALL> inline double lb(const double* q, int n) const { return BALL ? lbB(q, n) : lbM(q, n); }
    int descend(const double* q) const { int n = 0; while (nd[n].l >= 0) n = (q[nd[n].dim] < nd[n].val) ? nd[n].l : nd[n].r; return n; }
    inline void scan(const double* q, int n) { for (int id : nd[n].ids) H.add(dist2(q, pt(id)), id); }

    template <bool BALL> void knn_dfs(const double* q) {
        int L = descend(q); scan(q, L);
        static vector<int> st; st.clear(); st.push_back(0);
        while (!st.empty()) {
            int n = st.back(); st.pop_back();
            if (n == L || lb<BALL>(q, n) >= H.kth()) continue;
            if (nd[n].l < 0) { scan(q, n); continue; }
            int nr = (q[nd[n].dim] < nd[n].val) ? nd[n].l : nd[n].r, fr = (nr == nd[n].l) ? nd[n].r : nd[n].l;
            st.push_back(fr); st.push_back(nr);
        }
    }
    template <bool BALL> void knn_bfs(const double* q) {
        int L = descend(q); scan(q, L);
        static vector<int> qu; qu.clear(); qu.push_back(0); size_t h = 0;
        while (h < qu.size()) {
            int n = qu[h++];
            if (n == L || lb<BALL>(q, n) >= H.kth()) continue;
            if (nd[n].l < 0) { scan(q, n); continue; }
            qu.push_back(nd[n].l); qu.push_back(nd[n].r);
        }
    }
    void knn_bf(const double* q) {
        static vector<PQE> pq; pq.clear(); pq.emplace_back(0.0, 0);
        while (!pq.empty()) {
            pop_heap(pq.begin(), pq.end(), PQCMP); PQE e = pq.back(); pq.pop_back();
            if (e.first >= H.kth()) break;
            int n = e.second;
            if (nd[n].l < 0) { scan(q, n); continue; }
            for (int c : {nd[n].l, nd[n].r}) { double d = lbM(q, c); if (d < H.kth()) { pq.emplace_back(d, c); push_heap(pq.begin(), pq.end(), PQCMP); } }
        }
    }
    template <bool BALL> void rad_dfs(const double* q, double r) {
        double r2 = r * r; static vector<int> st; st.clear(); st.push_back(0);
        while (!st.empty()) {
            int n = st.back(); st.pop_back();
            if (lb<BALL>(q, n) > r2) continue;
            if (nd[n].l < 0) { for (int id : nd[n].ids) if (dist2(q, pt(id)) <= r2) OUT.push_back(id); continue; }
            st.push_back(nd[n].r); st.push_back(nd[n].l);
        }
    }
    template <bool BALL> void rad_bfs(const double* q, double r) {
        double r2 = r * r; static vector<int> qu; qu.clear(); qu.push_back(0); size_t h = 0;
        while (h < qu.size()) {
            int n = qu[h++];
            if (lb<BALL>(q, n) > r2) continue;
            if (nd[n].l < 0) { for (int id : nd[n].ids) if (dist2(q, pt(id)) <= r2) OUT.push_back(id); continue; }
            qu.push_back(nd[n].l); qu.push_back(nd[n].r);
        }
    }
    void rad_bf(const double* q, double r) {
        double r2 = r * r; static vector<PQE> pq; pq.clear(); pq.emplace_back(0.0, 0);
        while (!pq.empty()) {
            pop_heap(pq.begin(), pq.end(), PQCMP); PQE e = pq.back(); pq.pop_back();
            if (e.first > r2) break;
            int n = e.second;
            if (nd[n].l < 0) { for (int id : nd[n].ids) if (dist2(q, pt(id)) <= r2) OUT.push_back(id); continue; }
            for (int c : {nd[n].l, nd[n].r}) { double d = lbM(q, c); if (d <= r2) { pq.emplace_back(d, c); push_heap(pq.begin(), pq.end(), PQCMP); } }
        }
    }
    template <bool BALL> inline bool hit(int n, const Box& B) const {
        return BALL ? mind2(nd[n].c, B) <= nd[n].rad * nd[n].rad : inter(nd[n].bb, B);
    }
    template <bool BALL> void rng_dfs(const Box& B) {
        static vector<int> st; st.clear(); st.push_back(0);
        while (!st.empty()) {
            int n = st.back(); st.pop_back();
            if (!hit<BALL>(n, B)) continue;
            if (nd[n].l < 0) { for (int id : nd[n].ids) if (inside(pt(id), B)) OUT.push_back(id); continue; }
            st.push_back(nd[n].r); st.push_back(nd[n].l);
        }
    }
    template <bool BALL> void rng_bfs(const Box& B) {
        static vector<int> qu; qu.clear(); qu.push_back(0); size_t h = 0;
        while (h < qu.size()) {
            int n = qu[h++];
            if (!hit<BALL>(n, B)) continue;
            if (nd[n].l < 0) { for (int id : nd[n].ids) if (inside(pt(id), B)) OUT.push_back(id); continue; }
            qu.push_back(nd[n].l); qu.push_back(nd[n].r);
        }
    }
    void rng_bf(const double* q, const Box& B) {
        static vector<PQE> pq; pq.clear(); pq.emplace_back(0.0, 0);
        while (!pq.empty()) {
            pop_heap(pq.begin(), pq.end(), PQCMP); PQE e = pq.back(); pq.pop_back();
            int n = e.second;
            if (nd[n].l < 0) { for (int id : nd[n].ids) if (inside(pt(id), B)) OUT.push_back(id); continue; }
            for (int c : {nd[n].l, nd[n].r}) if (inter(nd[c].bb, B)) { pq.emplace_back(lbM(q, c), c); push_heap(pq.begin(), pq.end(), PQCMP); }
        }
    }
};

// ---------------------------------------------------------------- R-tree (STR bulk load, insertion with split)
struct RN { Box bb; bool leaf = true; int parent = -1; vector<int> ch; };
struct RT {
    vector<RN> nd; int root = 0; const int M = 32;
    void recompute(int n) {
        box_init(nd[n].bb);
        if (nd[n].leaf) for (int id : nd[n].ch) box_add(nd[n].bb, pt(id));
        else for (int c : nd[n].ch) box_merge(nd[n].bb, nd[c].bb);
    }
    // sort-tile-recursive grouping of items by their centres
    void tile(vector<int>& it, int lo, int hi, int dim, const function<double(int, int)>& key, vector<pair<int, int>>& groups) {
        int n = hi - lo;
        if (n <= M) { groups.emplace_back(lo, hi); return; }
        sort(it.begin() + lo, it.begin() + hi, [&](int a, int b) { return key(a, dim) < key(b, dim); });
        if (dim == D - 1) { for (int s = lo; s < hi; s += M) groups.emplace_back(s, min(hi, s + M)); return; }
        double P = ceil((double)n / M); int S = max(1, (int)ceil(pow(P, 1.0 / (D - dim))));
        int sz = (int)ceil((double)n / S);
        for (int s = lo; s < hi; s += sz) tile(it, s, min(hi, s + sz), dim + 1, key, groups);
    }
    void build(vector<int> ids) {
        vector<pair<int, int>> g;
        tile(ids, 0, ids.size(), 0, [&](int a, int d) { return pt(a)[d]; }, g);
        vector<int> level;
        for (auto& x : g) { int n = nd.size(); nd.emplace_back(); nd[n].ch.assign(ids.begin() + x.first, ids.begin() + x.second); recompute(n); level.push_back(n); }
        while (level.size() > 1) {
            vector<pair<int, int>> gg;
            tile(level, 0, level.size(), 0, [&](int a, int d) { return (nd[a].bb.lo[d] + nd[a].bb.hi[d]) / 2; }, gg);
            vector<int> up;
            for (auto& x : gg) {
                int n = nd.size(); nd.emplace_back(); nd[n].leaf = false;
                nd[n].ch.assign(level.begin() + x.first, level.begin() + x.second);
                for (int c : nd[n].ch) nd[c].parent = n;
                recompute(n); up.push_back(n);
            }
            level.swap(up);
        }
        root = level[0]; nd[root].parent = -1;
    }
    void insert(int id) {
        const double* p = pt(id); int n = root;
        while (!nd[n].leaf) {
            int best = -1; double be = 1e300, ba = 1e300;
            for (int c : nd[n].ch) {
                Box b = nd[c].bb; double a0 = volume(b); box_add(b, p); double e = volume(b) - a0;
                if (e < be || (e == be && a0 < ba)) { be = e; ba = a0; best = c; }
            }
            box_add(nd[n].bb, p); n = best;
        }
        box_add(nd[n].bb, p); nd[n].ch.push_back(id);
        if ((int)nd[n].ch.size() > M) split(n);
    }
    void split(int n) {
        Box bb = nd[n].bb; int ax = KDT::widest(bb);
        vector<int> e = nd[n].ch; bool leaf = nd[n].leaf;
        auto key = [&](int a) { return leaf ? pt(a)[ax] : (nd[a].bb.lo[ax] + nd[a].bb.hi[ax]) / 2; };
        sort(e.begin(), e.end(), [&](int a, int b) { return key(a) < key(b); });
        int half = e.size() / 2;
        int m = nd.size(); nd.emplace_back(); nd[m].leaf = leaf;
        nd[n].ch.assign(e.begin(), e.begin() + half); nd[m].ch.assign(e.begin() + half, e.end());
        if (!leaf) for (int c : nd[m].ch) nd[c].parent = m;
        recompute(n); recompute(m);
        int p = nd[n].parent;
        if (p < 0) {
            int r = nd.size(); nd.emplace_back(); nd[r].leaf = false; nd[r].ch = {n, m};
            nd[n].parent = nd[m].parent = r; recompute(r); root = r;
        } else {
            nd[m].parent = p; nd[p].ch.push_back(m);
            if ((int)nd[p].ch.size() > M) split(p);
        }
    }
    void knn_bf(const double* q) {
        static vector<PQE> pq; pq.clear(); pq.emplace_back(0.0, root);
        while (!pq.empty()) {
            pop_heap(pq.begin(), pq.end(), PQCMP); PQE e = pq.back(); pq.pop_back();
            if (e.first >= H.kth()) break;
            int n = e.second;
            if (nd[n].leaf) { for (int id : nd[n].ch) H.add(dist2(q, pt(id)), id); continue; }
            for (int c : nd[n].ch) { double d = mind2(q, nd[c].bb); if (d < H.kth()) { pq.emplace_back(d, c); push_heap(pq.begin(), pq.end(), PQCMP); } }
        }
    }
    void rad(const double* q, double r) {
        double r2 = r * r; static vector<int> st; st.clear(); st.push_back(root);
        while (!st.empty()) {
            int n = st.back(); st.pop_back();
            if (mind2(q, nd[n].bb) > r2) continue;
            if (nd[n].leaf) { for (int id : nd[n].ch) if (dist2(q, pt(id)) <= r2) OUT.push_back(id); continue; }
            for (int c : nd[n].ch) st.push_back(c);
        }
    }
    void rng(const Box& B) {
        static vector<int> st; st.clear(); st.push_back(root);
        while (!st.empty()) {
            int n = st.back(); st.pop_back();
            if (!inter(nd[n].bb, B)) continue;
            if (nd[n].leaf) { for (int id : nd[n].ch) if (inside(pt(id), B)) OUT.push_back(id); continue; }
            for (int c : nd[n].ch) st.push_back(c);
        }
    }
};

// ---------------------------------------------------------------- uniform grid (cell lists over the operating region)
struct Grid {
    int g = 1; Box bb; double w[3] = {1, 1, 1}; vector<vector<int>> cell; double dens = 1;
    void build(int n0, int ntotal) {   // the grid covers the region of all points the device will collect
        box_init(bb); for (int i = 0; i < ntotal; i++) box_add(bb, pt(i));
        for (int j = 0; j < D; j++) { double e = (bb.hi[j] - bb.lo[j]) * 1e-6 + 1e-12; bb.lo[j] -= e; bb.hi[j] += e; }
        g = max(1, (int)ceil(pow(ntotal / 8.0, 1.0 / D)));
        for (int j = 0; j < D; j++) w[j] = (bb.hi[j] - bb.lo[j]) / g;
        cell.assign((size_t)pow(g, D), {});
        for (int i = 0; i < n0; i++) add(i);
        dens = n0 / volume(bb);
    }
    inline int ci(double v, int j) const { int c = (int)floor((v - bb.lo[j]) / w[j]); return c < 0 ? 0 : (c >= g ? g - 1 : c); }
    inline size_t id(const int* c) const { size_t k = 0; for (int j = 0; j < D; j++) k = k * g + c[j]; return k; }
    void add(int i) { int c[3]; for (int j = 0; j < D; j++) c[j] = ci(pt(i)[j], j); cell[id(c)].push_back(i); }
    // visit the cells overlapping [lo, hi]; full = the cell lies inside the query box (range queries only)
    template <class F> void cells(const double* lo, const double* hi, F f) const {
        int a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
        for (int j = 0; j < D; j++) { a[j] = ci(lo[j], j); b[j] = ci(hi[j], j); }
        int c[3];
        for (c[0] = a[0]; c[0] <= b[0]; c[0]++)
            for (c[1] = (D > 1 ? a[1] : 0); c[1] <= (D > 1 ? b[1] : 0); c[1]++)
                for (c[2] = (D > 2 ? a[2] : 0); c[2] <= (D > 2 ? b[2] : 0); c[2]++) f(c);
    }
    void rad(const double* q, double r) {
        double lo[3], hi[3], r2 = r * r;
        for (int j = 0; j < D; j++) { lo[j] = q[j] - r; hi[j] = q[j] + r; }
        cells(lo, hi, [&](const int* c) { for (int id_ : cell[id(c)]) if (dist2(q, pt(id_)) <= r2) OUT.push_back(id_); });
    }
    void rng(const Box& B) {
        cells(B.lo, B.hi, [&](const int* c) {
            bool full = true;
            for (int j = 0; j < D; j++) {
                double cl = bb.lo[j] + c[j] * w[j], ch = cl + w[j];
                if (cl < B.lo[j] || ch > B.hi[j] || c[j] == 0 || c[j] == g - 1) { full = false; break; }
            }
            const auto& v = cell[id(c)];
            if (full) OUT.insert(OUT.end(), v.begin(), v.end());
            else for (int id_ : v) if (inside(pt(id_), B)) OUT.push_back(id_);
        });
    }
    // exact kNN by an expanding radius: start from the radius that covers about k points at the local density,
    // collect all points within the radius, and double it until at least k points are inside
    void knn(const double* q, int k) {
        int c0[3]; for (int j = 0; j < D; j++) c0[j] = ci(q[j], j);
        double cv = 1; for (int j = 0; j < D; j++) cv *= w[j];
        double ld = cell[id(c0)].size() / cv; if (ld <= 0) ld = dens;
        double unit = (D == 2) ? M_PI : 4.0 / 3 * M_PI;
        double r = pow(k / (ld * unit), 1.0 / D);
        double out2 = mind2(q, bb); if (out2 > 0) r += sqrt(out2);
        while (true) {
            H.init(k); double r2 = r * r, lo[3], hi[3]; long long within = 0;
            for (int j = 0; j < D; j++) { lo[j] = q[j] - r; hi[j] = q[j] + r; }
            cells(lo, hi, [&](const int* c) { for (int id_ : cell[id(c)]) { double d = dist2(q, pt(id_)); if (d <= r2) { within++; H.add(d, id_); } } });
            bool all = true; for (int j = 0; j < D; j++) if (lo[j] > bb.lo[j] || hi[j] < bb.hi[j]) all = false;
            if (within >= k || all) {
                if (within < k) {  // fewer than k points in total inside the radius: take every point (whole grid covered)
                    H.init(k); for (auto& v : cell) for (int id_ : v) H.add(dist2(q, pt(id_)), id_);
                }
                return;
            }
            r *= 2;
        }
    }
};

// ---------------------------------------------------------------- query features (incrementally maintained)
struct Feat {
    int g = 32; Box gb, data; vector<int> cnt; int nmax = 1; long long n = 0;
    double sum[3] = {0, 0, 0}, sxx[3][3] = {{0}}; double mean[3] = {0, 0, 0}, axis[3] = {1, 0, 0}, extent = 1;
    long long since = 0;
    int cell_of(const double* z, bool clamp, bool& out) const {
        int idx = 0; out = false;
        for (int j = 0; j < D; j++) {
            int c = (int)floor((z[j] - gb.lo[j]) / (gb.hi[j] - gb.lo[j]) * g);
            if (c < 0 || c >= g) { out = true; c = c < 0 ? 0 : g - 1; }
            idx = idx * g + c;
        }
        if (out && !clamp) return -1;
        return idx;
    }
    void init(int n0) {
        g = (D == 2) ? 32 : 16;
        box_init(data);
        for (int i = 0; i < n0; i++) box_add(data, pt(i));
        gb = data;
        for (int j = 0; j < D; j++) { double e = (gb.hi[j] - gb.lo[j]) * .01 + 1e-9; gb.lo[j] -= e; gb.hi[j] += e; }
        cnt.assign((size_t)pow(g, D), 0);
        for (int i = 0; i < n0; i++) add(pt(i), false);
        refresh();
    }
    void add(const double* p, bool live = true) {
        bool out; int c = cell_of(p, true, out); if (++cnt[c] > nmax) nmax = cnt[c];
        if (live) box_add(data, p);
        n++;
        for (int a = 0; a < D; a++) { sum[a] += p[a]; for (int b = 0; b < D; b++) sxx[a][b] += p[a] * p[b]; }
        if (live && ++since >= 1000) { refresh(); since = 0; }
    }
    void refresh() {  // principal axis by power iteration on the covariance matrix
        double C[3][3];
        for (int a = 0; a < D; a++) mean[a] = sum[a] / n;
        for (int a = 0; a < D; a++) for (int b = 0; b < D; b++) C[a][b] = sxx[a][b] / n - mean[a] * mean[b];
        double v[3] = {1, .7, .3};
        for (int it = 0; it < 30; it++) {
            double w[3] = {0, 0, 0}, s = 0;
            for (int a = 0; a < D; a++) for (int b = 0; b < D; b++) w[a] += C[a][b] * v[b];
            for (int a = 0; a < D; a++) s += w[a] * w[a];
            s = sqrt(s) + 1e-300; for (int a = 0; a < D; a++) v[a] = w[a] / s;
        }
        for (int a = 0; a < D; a++) axis[a] = v[a];
        extent = 0; for (int j = 0; j < D; j++) extent = max(extent, data.hi[j] - data.lo[j]);
    }
    // x = (1, l, l^2, h, min(1,10h), rho, l*rho, o)
    void compute(const double* z, double ell, double x[8], int& cellcnt) const {
        bool out; int c = cell_of(z, false, out);
        cellcnt = (c < 0) ? 0 : cnt[c];
        double rho = (c < 0) ? 0 : log(1.0 + cellcnt) / log(1.0 + nmax);
        double d[3], proj = 0, h2 = 0;
        for (int a = 0; a < D; a++) { d[a] = z[a] - mean[a]; proj += d[a] * axis[a]; }
        for (int a = 0; a < D; a++) { double t = d[a] - proj * axis[a]; h2 += t * t; }
        double h = sqrt(h2) / extent, o = sqrt(mind2(z, data)) / extent;
        x[0] = 1; x[1] = ell; x[2] = ell * ell; x[3] = h; x[4] = min(1.0, 10 * h); x[5] = rho; x[6] = ell * rho; x[7] = o;
    }
    double cellvol() const { double v = 1; for (int j = 0; j < D; j++) v *= (gb.hi[j] - gb.lo[j]) / g; return v; }
};

// ---------------------------------------------------------------- events
#pragma pack(push, 1)
struct Ev { int32_t kind, qtype; int64_t pid; double z[3]; double param; double half[3]; };
#pragma pack(pop)
static_assert(sizeof(Ev) == 72, "event size");

static KDT KD; static RT RTR; static Feat F; static Grid GR;
static const int NS = 7;
static const char* SNAME[NS] = {"R-BF", "KD-BF", "KD-DFS-MBR", "KD-BFS-MBR", "KD-DFS-Ball", "KD-BFS-Ball", "Grid"};
static const double KMAX = 1000, RMAX = .08;

// run strategy s once; returns a signature of the answer (count, and kth distance for kNN)
static pair<long long, double> run(int s, const Ev& e, const Box& B) {
    const double* q = e.z;
    if (e.qtype == 0) {
        H.init((int)e.param);
        switch (s) {
            case 0: RTR.knn_bf(q); break; case 1: KD.knn_bf(q); break;
            case 2: KD.knn_dfs<false>(q); break; case 3: KD.knn_bfs<false>(q); break;
            case 4: KD.knn_dfs<true>(q); break; case 5: KD.knn_bfs<true>(q); break; default: GR.knn(q, (int)e.param); break;
        }
        return {(long long)H.h.size(), H.h.empty() ? 0 : H.h.front().first};
    }
    OUT.clear();
    if (e.qtype == 1) {
        switch (s) {
            case 0: RTR.rad(q, e.param); break; case 1: KD.rad_bf(q, e.param); break;
            case 2: KD.rad_dfs<false>(q, e.param); break; case 3: KD.rad_bfs<false>(q, e.param); break;
            case 4: KD.rad_dfs<true>(q, e.param); break; case 5: KD.rad_bfs<true>(q, e.param); break; default: GR.rad(q, e.param); break;
        }
    } else {
        switch (s) {
            case 0: RTR.rng(B); break; case 1: KD.rng_bf(q, B); break;
            case 2: KD.rng_dfs<false>(B); break; case 3: KD.rng_bfs<false>(B); break;
            case 4: KD.rng_dfs<true>(B); break; case 5: KD.rng_bfs<true>(B); break; default: GR.rng(B); break;
        }
    }
    return {(long long)OUT.size(), 0.0};
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: bench points.bin stream.ev out.csv [R]\n"); return 1; }
    int R = argc > 4 ? atoi(argv[4]) : 3;
    FILE* fp = fopen(argv[1], "rb"); int64_t n; int32_t d;
    if (fread(&n, 8, 1, fp) != 1 || fread(&d, 4, 1, fp) != 1) return 2;
    D = d; PT.resize((size_t)n * D);
    if (fread(PT.data(), 8, PT.size(), fp) != PT.size()) return 3;
    fclose(fp);
    FILE* fe = fopen(argv[2], "rb"); int64_t ne, n0, nuse; int32_t dd;
    if (fread(&ne, 8, 1, fe) != 1 || fread(&n0, 8, 1, fe) != 1 || fread(&nuse, 8, 1, fe) != 1 || fread(&dd, 4, 1, fe) != 1) return 4;
    vector<Ev> ev(ne);
    if (fread(ev.data(), sizeof(Ev), ne, fe) != (size_t)ne) return 5;
    fclose(fe);

    auto t0 = Clk::now();
    vector<int> ids(n0); iota(ids.begin(), ids.end(), 0);
    KD.nd.reserve(4 * n / KD.C + 16); KD.build(ids, 0, n0, 0);
    RTR.build(ids); F.init(n0); GR.build(n0, n);
    fprintf(stderr, "built %lld points in %.2fs\n", (long long)n0, chrono::duration<double>(Clk::now() - t0).count());

    FILE* fo = fopen(argv[3], "w");
    fprintf(fo, "qid,qtype,param,ell");
    for (int s = 0; s < NS; s++) fprintf(fo, ",t%d", s);
    fprintf(fo, ",ans,ok,feat_ns");
    for (int i = 0; i < 8; i++) fprintf(fo, ",x%d", i);
    fprintf(fo, ",path_ns,depth,leafsz,leafvol,leafdist,cellcnt,estans,z0,z1,z2,ninserted\n");

    long long qid = 0, inserted = n0, bad = 0;
    for (const Ev& e : ev) {
        if (e.kind == 0) { KD.insert((int)e.pid); RTR.insert((int)e.pid); GR.add((int)e.pid); F.add(pt((int)e.pid)); inserted++; continue; }
        Box B; box_init(B);
        double ell = 0, meanhalf = 0;
        if (e.qtype == 2) {
            for (int j = 0; j < D; j++) { B.lo[j] = e.z[j] - e.half[j]; B.hi[j] = e.z[j] + e.half[j]; meanhalf += e.half[j] / D; }
            ell = meanhalf / RMAX;
        } else if (e.qtype == 1) ell = e.param / RMAX;
        else ell = log(1 + e.param) / log(1 + KMAX);

        // OASIS features (timed)
        double x[8]; int cellcnt;
        auto a = Clk::now(); F.compute(e.z, ell, x, cellcnt); auto b = Clk::now();
        double feat_ns = chrono::duration<double, nano>(b - a).count();
        // UnIS-style index-path features (timed): descend to the leaf of the query
        a = Clk::now();
        int L = KD.descend(e.z); const KN& lf = KD.nd[L];
        double ld = 1e300; for (int j = 0; j < D; j++) ld = min(ld, min(e.z[j] - lf.bb.lo[j], lf.bb.hi[j] - e.z[j]));
        b = Clk::now();
        double path_ns = chrono::duration<double, nano>(b - a).count();
        double leafvol = volume(lf.bb);
        double dens = cellcnt / F.cellvol(), estans;
        if (e.qtype == 0) estans = e.param;
        else if (e.qtype == 1) estans = dens * (D == 2 ? M_PI * e.param * e.param : 4.0 / 3 * M_PI * pow(e.param, 3));
        else estans = dens * volume(B);

        // correctness: all strategies must agree
        auto ref = run(0, e, B); bool ok = true;
        for (int s = 1; s < NS; s++) {
            auto o = run(s, e, B);
            if (o.first != ref.first || fabs(o.second - ref.second) > 1e-12 * (1 + ref.second)) ok = false;
        }
        if (!ok) bad++;
        double best[NS]; for (int s = 0; s < NS; s++) best[s] = 1e300;
        for (int r = 0; r < R; r++)
            for (int i = 0; i < NS; i++) {
                int s = (i + (int)qid + r) % NS;
                auto c0 = Clk::now(); run(s, e, B); auto c1 = Clk::now();
                best[s] = min(best[s], chrono::duration<double, nano>(c1 - c0).count());
            }
        fprintf(fo, "%lld,%d,%.6g,%.6g", qid, e.qtype, e.qtype == 2 ? meanhalf : e.param, ell);
        for (int s = 0; s < NS; s++) fprintf(fo, ",%.1f", best[s]);
        fprintf(fo, ",%lld,%d,%.1f", ref.first, ok ? 1 : 0, feat_ns);
        for (int i = 0; i < 8; i++) fprintf(fo, ",%.6g", x[i]);
        fprintf(fo, ",%.1f,%d,%zu,%.6g,%.6g,%d,%.6g,%.6g,%.6g,%.6g,%lld\n", path_ns, lf.depth, lf.ids.size(), leafvol, ld, cellcnt,
                estans, e.z[0], e.z[1], D > 2 ? e.z[2] : 0.0, inserted);
        qid++;
    }
    fclose(fo);
    fprintf(stderr, "queries %lld, mismatched answers %lld, total %.1fs\n", qid, bad, chrono::duration<double>(Clk::now() - t0).count());
    return 0;
}
