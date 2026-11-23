// Per-query cost of OASIS (2026-11-13 revision) in C++: two clock reads (every search is timed), the basic quantities
// (answer estimate from a summed-area table, shape, density of the cell), the closest class, the cell experience, the
// candidates and their temperatures, the draw, the time limit, and the update of the experience (decay inside a window).
// Decisions use the measured search times, so the code paths match the replay in oasis2.py.
// Build: g++ -O3 -march=native -std=c++17 -o selbench2 selbench2.cpp
// Usage: selbench2 ctx2 times.bin types.bin query.bin dim tau K r N0 T0 gamma b lam w cf features limited recency
//   query.bin columns: z0 z1 z2 param h0 h1 h2   (h: half sides of a range query; coordinates in the unit box)
// Matrices: int64 rows, int64 cols, then rows*cols float64. Prints mean nanoseconds per query.
#include <bits/stdc++.h>
using namespace std;
using Clk = chrono::steady_clock;

struct Mat { int64_t r = 0, c = 0; vector<double> v; double at(int64_t i, int64_t j) const { return v[i * c + j]; } };
static Mat readmat(const char* f) {
    Mat m; FILE* fp = fopen(f, "rb");
    if (!fp || fread(&m.r, 8, 1, fp) != 1 || fread(&m.c, 8, 1, fp) != 1) { fprintf(stderr, "bad %s\n", f); exit(1); }
    m.v.resize(m.r * m.c);
    if (fread(m.v.data(), 8, m.v.size(), fp) != m.v.size()) { fprintf(stderr, "short %s\n", f); exit(1); }
    fclose(fp); return m;
}
static volatile double SINK = 0;
static const int M = 7;

// Short polynomial log2 and exp2 (relative error about 1e-6); the libm log10/pow calls took most of the time per query.
static inline double flog2(double x) {                  // x > 0
    union { double d; uint64_t u; } v = {x};
    int e = (int)((v.u >> 52) & 0x7ff) - 1023;
    v.u = (v.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;   // mantissa in [1, 2)
    double m = v.d, t = (m - 1) / (m + 1), t2 = t * t;
    double ln = t * (2.0 + t2 * (2.0 / 3 + t2 * (2.0 / 5 + t2 * (2.0 / 7 + t2 * (2.0 / 9)))));
    return e + ln * 1.4426950408889634;
}
static inline double fexp2(double x) {
    if (x < -1000) return 0.0;
    if (x > 1000) return 1e300;
    double fi = floor(x), f = x - fi;
    double p = 1 + f * (0.6931471805599453 + f * (0.2402265069591007 + f * (0.0555041086648216 + f * (0.0096181291076285 +
               f * (0.0013333558146428 + f * 0.0001540353039338)))));
    union { double d; uint64_t u; } v; v.u = (uint64_t)((int64_t)fi + 1023) << 52;
    return p * v.d;
}
static inline double flog10(double x) { return flog2(x) * 0.30102999566398120; }
static inline double flog1p(double x) { return flog2(1.0 + x) * 0.6931471805599453; }
static inline double fexpm(double x) { return fexp2(-x * 1.4426950408889634); }   // e^-x

struct Cell { double S[M], W[M]; int c[M]; };
struct Entry { int a, cell; double s, u; };
struct Cls {
    int type = 0; double ctr[3] = {0, 0, 0}; long long n = 0;
    double S[M], W[M]; int c[M];
    vector<int> slot; vector<Cell> cells;
    vector<Entry> ring; int head = 0, fill = 0;
    long long t = 0, base = 0; double uc = 1; int ord[M];
    Cls() { memset(S, 0, sizeof S); memset(W, 0, sizeof W); memset(c, 0, sizeof c); for (int a = 0; a < M; a++) ord[a] = a; }
};

int ctx2(const Mat& T, const Mat& Y, const Mat& Z, int D, double tau, int K, int r, double N0, double T0, double gamma,
         double b, double lam, int w, int cf, bool features, bool limited, bool recency) {
    cf = max(cf, 1);
    if (!recency) { lam = 1.0; w = 0; }
    const int G = D == 2 ? 64 : 24, G1 = G + 1, GC = (G + cf - 1) / cf;
    int NCC = 1; for (int j = 0; j < D; j++) NCC *= GC;
    int tot = 1, ncell = 1; for (int j = 0; j < D; j++) { tot *= G1; ncell *= G; }
    vector<double> sat(tot), cnt(ncell); mt19937 rg(5); double acc = 0;
    for (auto& v : cnt) v = rg() % 64;
    for (auto& v : sat) { acc += rg() % 16; v = acc; }
    double navg = 0; for (double v : cnt) navg += v; navg /= ncell;
    const long long lim = lam < 1 ? (long long)(300 / -log(lam)) : 0;
    vector<Cls> cls; cls.reserve(3 * K + 3); vector<int> bytype[3];
    mt19937_64 rng(11); uniform_real_distribution<double> U01(0, 1);
    auto add = [&](Cls& C, int a, int cs, double s) {
        if (lam < 1 && C.t - C.base > lim) {                    // renormalise the scaled sums (rare)
            double f = 1.0 / C.uc;
            for (int k = 0; k < M; k++) { C.S[k] *= f; C.W[k] *= f; }
            for (auto& e : C.cells) for (int k = 0; k < M; k++) { e.S[k] *= f; e.W[k] *= f; }
            for (auto& e : C.ring) e.u *= f;
            C.uc = 1; C.base = C.t;
        }
        double u = C.uc;
        C.S[a] += u * s; C.W[a] += u; C.c[a]++;
        if (cs >= 0) { Cell& e = C.cells[cs]; e.S[a] += u * s; e.W[a] += u; e.c[a]++; }
        if (w > 0) {
            if ((int)C.ring.size() < w) C.ring.resize(w);
            if (C.fill == w) {
                Entry o = C.ring[C.head];
                C.S[o.a] -= o.u * o.s; C.W[o.a] -= o.u; if (--C.c[o.a] == 0) { C.S[o.a] = 0; C.W[o.a] = 0; }
                if (o.cell >= 0) { Cell& e = C.cells[o.cell]; e.S[o.a] -= o.u * o.s; e.W[o.a] -= o.u; if (--e.c[o.a] == 0) { e.S[o.a] = 0; e.W[o.a] = 0; } }
            } else C.fill++;
            C.ring[C.head] = {a, cs, s, u}; C.head = (C.head + 1) % w;
        }
        C.t++; if (lam < 1) C.uc /= lam;
    };
    int64_t Q = T.r; double lo = 0, wc = 1.0 / G;
    auto t0 = Clk::now();
    for (int64_t q = 0; q < Q; q++) {
        auto s0 = Clk::now();                                   // the search starts (timed)
        int ty = (int)Y.at(q, 0); const double* z = &Z.v[q * Z.c];
        // ---- basic quantities
        double est, shape = 0;
        if (ty == 0) est = z[3];
        else {
            double h[3]; for (int j = 0; j < D; j++) h[j] = ty == 1 ? z[3] : z[4 + j];
            int ia[3], ib[3]; double fr = 1;
            for (int j = 0; j < D; j++) {
                double a = (z[j] - h[j] - lo) / wc, bb = (z[j] + h[j] - lo) / wc;
                a = a < 0 ? 0 : a > G ? G : a; bb = bb < 0 ? 0 : bb > G ? G : bb;
                ia[j] = (int)floor(a); ib[j] = (int)ceil(bb); fr *= max(bb - a, 1e-9) / max(ib[j] - ia[j], 1);
            }
            double e = 0;
            for (int c = 0; c < (1 << D); c++) {
                int id = 0, sg = 1;
                for (int j = 0; j < D; j++) { int v = (c >> j & 1) ? ib[j] : (sg = -sg, ia[j]); id = id * G1 + v; }
                e += sg * sat[id];
            }
            est = e * min(1.0, fr) * (ty == 1 ? (D == 2 ? M_PI / 4 : M_PI / 6) : 1.0);
            if (ty == 2) { double mx = 0, mn = 1e300; for (int j = 0; j < D; j++) { mx = max(mx, h[j]); mn = min(mn, h[j]); } shape = flog10(max(mx / max(mn, 1e-300), 1.0)); }
        }
        long long flat = 0; int cflat = 0;
        for (int j = 0; j < D; j++) { int ci = (int)((z[j] - lo) / wc); ci = ci < 0 ? 0 : ci >= G ? G - 1 : ci; flat = flat * G + ci; cflat = cflat * GC + ci / cf; }
        double x[3] = {flog10(1 + max(est, 0.0)), shape, flog10((1 + cnt[flat]) / (1 + navg))};
        // ---- common feature: the closest class of the same type
        int cid = -1, cs = -1;
        if (features) {
            int best = -1; double bd = 1e300;
            for (int j : bytype[ty]) { double d = 0; for (int k = 0; k < 3; k++) { double t2 = x[k] - cls[j].ctr[k]; d += t2 * t2; } if (d < bd) { bd = d; best = j; } }
            if (best < 0 || (bd > tau && (int)bytype[ty].size() < K)) {
                cls.emplace_back(); Cls& C = cls.back(); C.type = ty; C.n = 1; for (int k = 0; k < 3; k++) C.ctr[k] = x[k];
                cid = (int)cls.size() - 1; bytype[ty].push_back(cid);
                if (best >= 0) {                                // warm start: try the plans in the order of the closest class
                    Cls& Bc = cls[best]; double mu[M];
                    for (int a = 0; a < M; a++) mu[a] = Bc.c[a] > 0 ? Bc.S[a] / Bc.W[a] : 1e300;
                    stable_sort(cls[cid].ord, cls[cid].ord + M, [&](int p1, int p2) { return mu[p1] < mu[p2]; });
                }
            } else { Cls& C = cls[best]; C.n++; for (int k = 0; k < 3; k++) C.ctr[k] += (x[k] - C.ctr[k]) / C.n; cid = best; }
            Cls& Cc = cls[cid];                                 // cell of the coarse grid: direct slot, no hashing
            if (Cc.slot.empty()) Cc.slot.assign(NCC, -1);
            if (Cc.slot[cflat] < 0) { Cc.slot[cflat] = (int)Cc.cells.size(); Cc.cells.emplace_back(); memset(&Cc.cells.back(), 0, sizeof(Cell)); }
            cs = Cc.slot[cflat];
        } else {
            if (bytype[ty].empty()) { cls.emplace_back(); cls.back().type = ty; bytype[ty].push_back((int)cls.size() - 1); }
            cid = bytype[ty][0];
        }
        Cls& C = cls[cid];
        double scale = lam < 1 ? 1.0 / (C.uc * lam) : 1.0;
        double muc[M], mix[M], Ni[M]; int order[M], ntried = 0, untried = -1;
        for (int a = 0; a < M; a++) {
            muc[a] = C.c[a] > 0 ? C.S[a] / C.W[a] : 1e300; order[a] = a;
            if (C.c[a] > 0) ntried++;
            if (cs >= 0) { const Cell& e = C.cells[cs]; double ni = e.c[a] > 0 ? e.W[a] * scale : 0, mi = e.c[a] > 0 ? e.S[a] / e.W[a] : 0;
                           Ni[a] = ni; mix[a] = C.c[a] > 0 ? (ni * mi + N0 * muc[a]) / (ni + N0) : 1e300; }
            else { Ni[a] = C.c[a] > 0 ? C.W[a] * scale : 0; mix[a] = muc[a]; }
        }
        for (int i = 0; i < M; i++) if (C.c[C.ord[i]] == 0) { untried = C.ord[i]; break; }
        int nc = min(limited ? r : M, ntried);
        partial_sort(order, order + nc, order + M, [&](int p1, int p2) { return muc[p1] < muc[p2]; });
        int astar = -1; double bm = 1e300;
        for (int i = 0; i < nc; i++) if (mix[order[i]] < bm) { bm = mix[order[i]]; astar = order[i]; }
        int pick;
        if (untried >= 0) pick = untried;
        else if (nc == 1) pick = order[0];
        else {
            double nsum = 0; for (int a = 0; a < M; a++) nsum += C.c[a] > 0 ? C.W[a] * scale : 0;
            double wts[M], sw = 0, Tg = T0 / (1 + flog1p(nsum));
            for (int i = 0; i < nc; i++) {
                int a = order[i]; double Ta = limited ? T0 * (1 + gamma * fexpm(Ni[a])) / (1 + flog1p(Ni[a])) : Tg;
                wts[i] = fexp2(flog2(bm / mix[a]) / max(Ta, 1e-12)); sw += wts[i];
            }
            double u = U01(rng) * sw; pick = order[nc - 1];
            for (int i = 0; i < nc; i++) { u -= wts[i]; if (u <= 0) { pick = order[i]; break; } }
        }
        double tv = T.at(q, pick) / 1000.0; bool stop = false;
        if (astar >= 0 && pick != astar && limited && tv > b * bm) stop = true;
        auto s1 = Clk::now();                                   // the search ends (timed)
        SINK += chrono::duration<double, nano>(s1 - s0).count() * 1e-12;
        if (stop) { add(C, pick, cs, b * bm); add(C, astar, cs, T.at(q, astar) / 1000.0); }
        else add(C, pick, cs, tv);
        SINK += pick;
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / Q);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    string mode = argv[1];
    if (mode == "ctx2" && argc >= 19)
        return ctx2(readmat(argv[2]), readmat(argv[3]), readmat(argv[4]), atoi(argv[5]), atof(argv[6]), atoi(argv[7]), atoi(argv[8]),
                    atof(argv[9]), atof(argv[10]), atof(argv[11]), atof(argv[12]), atof(argv[13]), atoi(argv[14]), atoi(argv[15]),
                    atoi(argv[16]) != 0, atoi(argv[17]) != 0, atoi(argv[18]) != 0);
    fprintf(stderr, "usage: selbench2 ctx2 times types query dim tau K r N0 T0 gamma b lam w cf features limited recency\n");
    return 1;
}
