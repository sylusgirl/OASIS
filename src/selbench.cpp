// Measures the per-query cost of each selector in C++ (selection + model update), on the real feature stream.
// Build: g++ -O3 -march=native -std=c++17 -o selbench selbench.cpp
// Modes:
//   selbench online <rule> feats.bin times.bin types.bin alpha lambda window   (rule: lcb ts eps squarecb ucb1)
//   selbench trees  model.txt feats.bin        (tree ensemble: RF/DT classifier, or per-arm regressors)
//   selbench knn    train.bin feats.bin k
//   selbench mlp    mlp.txt feats.bin
//   selbench ctx    times.bin types.bin query.bin alpha bound pace window revive dim   (OASIS final rule)
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

// ---------------------------------------------------------------- online linear selectors (d = 8, M = 6 per type)
static const int Dm = 8, M = 7;
struct Arm { double Ai[Dm][Dm], b[Dm], th[Dm]; int n = 0; double sum = 0; };
static void arm_init(Arm& a, double lam) {
    for (int i = 0; i < Dm; i++) { for (int j = 0; j < Dm; j++) a.Ai[i][j] = (i == j) ? 1.0 / lam : 0; a.b[i] = 0; a.th[i] = 0; }
}
static void arm_update(Arm& a, const double* x, double c, double sign) {  // sign +1 add, -1 remove (Sherman-Morrison)
    double u[Dm], den = 0;
    for (int i = 0; i < Dm; i++) { u[i] = 0; for (int j = 0; j < Dm; j++) u[i] += a.Ai[i][j] * x[j]; den += x[i] * u[i]; }
    den = 1 + sign * den;
    for (int i = 0; i < Dm; i++) for (int j = 0; j < Dm; j++) a.Ai[i][j] -= sign * u[i] * u[j] / den;
    for (int i = 0; i < Dm; i++) a.b[i] += sign * c * x[i];
    for (int i = 0; i < Dm; i++) { a.th[i] = 0; for (int j = 0; j < Dm; j++) a.th[i] += a.Ai[i][j] * a.b[j]; }
    a.n += (int)sign; a.sum += sign * c;
}
static double quad(const Arm& a, const double* x) {
    double s = 0;
    for (int i = 0; i < Dm; i++) { double t = 0; for (int j = 0; j < Dm; j++) t += a.Ai[i][j] * x[j]; s += x[i] * t; }
    return s;
}
static double dot(const double* a, const double* b) { double s = 0; for (int i = 0; i < Dm; i++) s += a[i] * b[i]; return s; }

// context bucket of the tabular rule: size (4) x density (4) x outside (2)
static inline int bucket(const double* x) {
    int lb = min(3, (int)(x[1] * 4)), db = min(3, (int)(x[5] * 4)), of = x[7] > 0 ? 1 : 0;
    return lb * 8 + db * 2 + of;
}

int online(const string& rule, const Mat& X, const Mat& T, const Mat& Y, double alpha, double lam, int window, bool logt) {
    vector<array<Arm, M>> arms(3);
    for (auto& g : arms) for (auto& a : g) arm_init(a, lam);
    static double tc[3][32][M], ts[3][32][M];
    memset(tc, 0, sizeof tc); memset(ts, 0, sizeof ts);
    mt19937_64 rng(7); normal_distribution<double> N01; uniform_real_distribution<double> U01;
    deque<tuple<int, int, int>> win[3];   // (row, type, arm)
    int64_t Q = X.r; long long tcount[3] = {0, 0, 0};
    auto t0 = Clk::now();
    for (int64_t q = 0; q < Q; q++) {
        int ty = (int)Y.at(q, 0); const double* x = &X.v[q * X.c];
        auto& g = arms[ty]; int pick = 0; long long tt = ++tcount[ty];
        if (rule == "tab") {
            int k = bucket(x); double best = 1e300; pick = -1;
            for (int a = 0; a < M; a++) if (tc[ty][k][a] == 0) { pick = a; break; }
            if (pick < 0)
                for (int a = 0; a < M; a++) {
                    double s = ts[ty][k][a] / tc[ty][k][a] - alpha * .5 / sqrt(tc[ty][k][a]);
                    if (s < best) { best = s; pick = a; }
                }
            double c = logt ? log(T.at(q, pick) / 1000.0) : T.at(q, pick) / 1000.0;
            tc[ty][k][pick] += 1; ts[ty][k][pick] += c;
            if (window > 0) {
                win[ty].emplace_back((int)q, k, pick);
                if ((int)win[ty].size() > window) {
                    auto [r, k2, a2] = win[ty].front(); win[ty].pop_front();
                    double c2 = logt ? log(T.at(r, a2) / 1000.0) : T.at(r, a2) / 1000.0;
                    tc[ty][k2][a2] -= 1; ts[ty][k2][a2] -= c2;
                }
            }
            SINK += pick;
            continue;
        }
        if (tt <= 4 * M) pick = (int)((tt - 1) % M);
        else if (rule == "lcb") {
            double best = 1e300;
            for (int a = 0; a < M; a++) { double s = dot(g[a].th, x) - alpha * sqrt(max(0.0, quad(g[a], x))); if (s < best) { best = s; pick = a; } }
        } else if (rule == "ts") {  // sample theta ~ N(th, alpha^2 Ai) via Cholesky of Ai
            double best = 1e300;
            for (int a = 0; a < M; a++) {
                double L[Dm][Dm] = {{0}};
                for (int i = 0; i < Dm; i++) for (int j = 0; j <= i; j++) {
                    double s = g[a].Ai[i][j]; for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
                    L[i][j] = (i == j) ? sqrt(max(s, 1e-12)) : s / L[j][j];
                }
                double z[Dm], th[Dm]; for (int i = 0; i < Dm; i++) z[i] = N01(rng);
                for (int i = 0; i < Dm; i++) { th[i] = g[a].th[i]; for (int k = 0; k <= i; k++) th[i] += alpha * L[i][k] * z[k]; }
                double s = dot(th, x); if (s < best) { best = s; pick = a; }
            }
        } else if (rule == "eps") {
            if (U01(rng) < .05) pick = (int)(rng() % M);
            else { double best = 1e300; for (int a = 0; a < M; a++) { double s = dot(g[a].th, x); if (s < best) { best = s; pick = a; } } }
        } else if (rule == "squarecb") {
            double yh[M], mn = 1e300; int am = 0;
            for (int a = 0; a < M; a++) { yh[a] = dot(g[a].th, x); if (yh[a] < mn) { mn = yh[a]; am = a; } }
            double p[M], ps = 0, gam = 200;
            for (int a = 0; a < M; a++) if (a != am) { p[a] = 1.0 / (M + gam * (yh[a] - mn)); ps += p[a]; }
            p[am] = 1 - ps; double u = U01(rng), c = 0; pick = am;
            for (int a = 0; a < M; a++) { c += p[a]; if (u <= c) { pick = a; break; } }
        } else {  // ucb1 on mean cost, no features
            double best = 1e300;
            for (int a = 0; a < M; a++) { double s = g[a].sum / max(1, g[a].n) - alpha * sqrt(2 * log((double)tt) / max(1, g[a].n)); if (s < best) { best = s; pick = a; } }
        }
        double c = logt ? log(T.at(q, pick) / 1000.0) : T.at(q, pick) / 1000.0;
        arm_update(g[pick], x, c, +1);
        if (window > 0) {
            win[ty].emplace_back((int)q, ty, pick);
            if ((int)win[ty].size() > window) {
                auto [r, t2, a2] = win[ty].front(); win[ty].pop_front();
                double c2 = logt ? log(T.at(r, a2) / 1000.0) : T.at(r, a2) / 1000.0;
                arm_update(arms[t2][a2], &X.v[r * X.c], c2, -1);
            }
        }
        SINK += pick;
    }
    double ns = chrono::duration<double, nano>(Clk::now() - t0).count() / Q;
    printf("%.2f\n", ns);
    return 0;
}

// lazy OASIS: per query only the context bucket is computed from the raw query (grid cell, size, outside flag);
// cached greedy decisions are reused; every r-th query of a type explores by LCB and updates the model.
int lazy(const Mat& X, const Mat& T, const Mat& Y, const Mat& Z, double alpha, double lam, int window, int r, int expire) {
    vector<array<Arm, M>> arms(3);
    for (auto& g : arms) for (auto& a : g) arm_init(a, lam);
    const int G = 32; vector<int> grid(G * G * G); mt19937 rg(3); for (auto& c : grid) c = rg() % 2000;
    int nmax = 2000; int th1 = nmax / 16, th2 = nmax / 4, th3 = nmax / 2;    // density bucket thresholds on cell counts
    static int cstamp[3][32], carm[3][32];
    for (int t = 0; t < 3; t++) for (int k = 0; k < 32; k++) cstamp[t][k] = -1000000000;
    deque<pair<int, int>> win[3]; long long seen[3] = {0, 0, 0};
    int64_t Q = X.r;
    auto t0 = Clk::now();
    for (int64_t q = 0; q < Q; q++) {
        int ty = (int)Y.at(q, 0); auto& g = arms[ty]; long long n = ++seen[ty];
        const double* z = &Z.v[q * Z.c];
        // context bucket from the raw query
        int idx = 0; bool out = false;
        for (int j = 0; j < 3; j++) { int c = (int)(z[j] * G); if (c < 0 || c >= G) { out = true; c = c < 0 ? 0 : G - 1; } idx = idx * G + c; }
        int cnt = grid[idx];
        int db = cnt < th1 ? 0 : cnt < th2 ? 1 : cnt < th3 ? 2 : 3;
        double p = z[3]; int lb = ty == 0 ? (p < 5 ? 0 : p < 50 ? 1 : p < 500 ? 2 : 3) : (p < .02 ? 0 : p < .04 ? 1 : p < .06 ? 2 : 3);
        int key = lb * 8 + db * 2 + (out ? 1 : 0);
        const double* x = &X.v[q * X.c];
        int pick;
        bool fb = false;
        if (n <= M) { pick = (int)(n - 1); fb = true; }
        else if (n % r == 0) {
            double best = 1e300; pick = 0;
            for (int a = 0; a < M; a++) { double s = dot(g[a].th, x) - alpha * sqrt(max(0.0, quad(g[a], x))); if (s < best) { best = s; pick = a; } }
            fb = true;
        } else if (n - cstamp[ty][key] < expire) pick = carm[ty][key];
        else {
            double best = 1e300; pick = 0;
            for (int a = 0; a < M; a++) { double s = dot(g[a].th, x); if (s < best) { best = s; pick = a; } }
            cstamp[ty][key] = (int)n; carm[ty][key] = pick;
        }
        if (fb) {
            arm_update(g[pick], x, log(T.at(q, pick) / 1000.0), +1);
            if (window > 0) {
                win[ty].emplace_back((int)q, pick);
                if ((int)win[ty].size() > window) {
                    auto [r0, a2] = win[ty].front(); win[ty].pop_front();
                    arm_update(arms[ty][a2], &X.v[r0 * X.c], log(T.at(r0, a2) / 1000.0), -1);
                }
            }
        }
        SINK += pick;
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / Q);
    return 0;
}

// bucketed OASIS: per query the context bucket is computed from the raw query; cached decisions are reused; every
// r-th query of a type is a feedback query that explores (untried first, then LCB), updates bucket statistics and may
// eliminate a strategy that is `elim` times slower than the best one in its coarse bucket.
int bucketed(const Mat& X, const Mat& T, const Mat& Y, const Mat& Z, double alpha, int r, int expire, double elim, int window) {
    const int G = 32; vector<int> grid(G * G * G); mt19937 rg(3); for (auto& c : grid) c = rg() % 2000;
    int th1 = 125, th2 = 500, th3 = 1000;
    static double cc[18][M], sc[18][M], cf[144][M], sf[144][M]; static bool dead[18][M]; static int cst[144], carm[144];
    memset(cc, 0, sizeof cc); memset(sc, 0, sizeof sc); memset(cf, 0, sizeof cf); memset(sf, 0, sizeof sf); memset(dead, 0, sizeof dead);
    for (int i = 0; i < 144; i++) cst[i] = -1000000000;
    struct W { int c, f, a; double v; }; deque<W> win[3]; long long seen[3] = {0, 0, 0};
    double le = log(elim);
    int64_t Q = X.r;
    auto t0 = Clk::now();
    for (int64_t q = 0; q < Q; q++) {
        int ty = (int)Y.at(q, 0); long long n = ++seen[ty];
        const double* z = &Z.v[q * Z.c];
        int idx = 0; bool out = false;
        for (int j = 0; j < 3; j++) { int c = (int)(z[j] * G); if (c < 0 || c >= G) { out = true; c = c < 0 ? 0 : G - 1; } idx = idx * G + c; }
        int cnt = grid[idx], db = cnt < th1 ? 0 : cnt < th2 ? 1 : cnt < th3 ? 2 : 3;
        double p = z[3]; int sb = 0;
        if (ty == 0) sb = p < 5 ? 0 : p < 50 ? 1 : p < 500 ? 2 : 3;
        else {   // summed-area-table estimate of the answers: four corner reads
            int i0 = idx % (G * G), i1 = (idx + 37) % (G * G), i2 = (idx + 991) % (G * G), i3 = (idx + 4093) % (G * G);
            double e = (double)grid[i0] - grid[i1] - grid[i2] + grid[i3] + z[4];
            sb = min(5, (int)floor(log10(1 + fabs(e))));
        }
        int c0 = ty * 6 + sb, f0 = (c0 * 4 + db) * 2 + (out ? 1 : 0);
        int pick = -1;
        if (n % r == 0 || n <= M) {
            for (int a = 0; a < M; a++) if (!dead[c0][a] && cc[c0][a] == 0) { pick = a; break; }
            if (pick < 0) {
                double best = 1e300;
                for (int a = 0; a < M; a++) {   // exploration at the coarse level only
                    if (dead[c0][a] || cc[c0][a] <= 0) continue;
                    double s = sc[c0][a] / cc[c0][a] * (1 - alpha / sqrt(cc[c0][a])); if (s < best) { best = s; pick = a; }
                }
                if (pick < 0) pick = 0;
            }
            int wa = -1; double wm = 1e300, bm = 1e300;
            for (int a = 0; a < M; a++) if (!dead[c0][a] && cc[c0][a] > 0 && sc[c0][a] / cc[c0][a] < wm) { wm = sc[c0][a] / cc[c0][a]; wa = a; }
            int live = 0; for (int a = 0; a < M; a++) if (!dead[c0][a]) live++;
            double tp = T.at(q, pick) / 1000.0;
            if (wa >= 0 && pick != wa && tp > 2.0 * wm) {   // bounded exploration: abort the probe, re-run the winner
                if (live > 2) dead[c0][pick] = true;
                pick = wa;
            }
            double v = T.at(q, pick) / 1000.0;
            cc[c0][pick] += 1; sc[c0][pick] += v; cf[f0][pick] += 1; sf[f0][pick] += v;
            for (int a = 0; a < M; a++) if (cc[c0][a] > 0) bm = min(bm, sc[c0][a] / cc[c0][a]);
            if (cc[c0][pick] >= 3 && sc[c0][pick] / cc[c0][pick] > bm * elim && live > 2) dead[c0][pick] = true;   // this coarse bucket only
            if (window > 0) {
                win[ty].push_back({c0, f0, pick, v});
                if ((int)win[ty].size() > window) {
                    W w = win[ty].front(); win[ty].pop_front();
                    cc[w.c][w.a] -= 1; sc[w.c][w.a] -= w.v; cf[w.f][w.a] -= 1; sf[w.f][w.a] -= w.v;
                }
            }
        } else if (n - cst[f0] < expire && !dead[c0][carm[f0]]) pick = carm[f0];
        else {
            double best = 1e300;   // conservative greedy: coarse winner unless the fine bucket is clearly faster
            for (int a = 0; a < M; a++) {
                if (dead[c0][a] || cc[c0][a] <= 0) continue;
                double m = sc[c0][a] / cc[c0][a]; if (m < best) { best = m; pick = a; }
            }
            if (pick < 0) pick = 0;
            if (cf[f0][pick] >= 5) {
                double ref = sf[f0][pick] / cf[f0][pick], bf = 1e300; int b2 = pick;
                for (int a = 0; a < M; a++) if (!dead[c0][a] && cf[f0][a] >= 5) { double m = sf[f0][a] / cf[f0][a]; if (m < bf) { bf = m; b2 = a; } }
                if (bf < ref * 0.9) pick = b2;
            }
            cst[f0] = (int)n; carm[f0] = pick;
        }
        SINK += pick;
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / Q);
    return 0;
}

// OASIS (final rule): per query two clock reads (every search is timed), the answer estimate of a radius/range query
// from a summed-area table (2^d corner reads), the context, the choice of the strategy and the update of the context
// statistics with a sliding window. Probe aborts use the measured search times.
int ctxrule(const Mat& T, const Mat& Y, const Mat& Z, double alpha, double bound, int pace, int window, int revive, int D) {
    const int G = D == 2 ? 64 : 24; const int G1 = G + 1;
    int tot = 1; for (int j = 0; j < D; j++) tot *= G1;
    vector<double> sat(tot); mt19937 rg(5); double acc = 0; for (auto& v : sat) { acc += rg() % 16; v = acc; }
    const int NK = 24;
    static double cnt[NK][M], sm[NK][M]; static long long dead[NK][M], visits[NK], probes[NK]; static int order[NK][M]; static bool started[NK];
    memset(cnt, 0, sizeof cnt); memset(sm, 0, sizeof sm); memset(visits, 0, sizeof visits); memset(probes, 0, sizeof probes); memset(started, 0, sizeof started);
    for (int k = 0; k < NK; k++) for (int a = 0; a < M; a++) dead[k][a] = -1000000000LL;
    int W = max(window, 1);
    vector<vector<pair<int, double>>> ring(NK, vector<pair<int, double>>(W)); vector<int> head(NK, 0), fill(NK, 0);
    int64_t Q = T.r; double lo = 0, w = 1.0 / G;
    auto t0 = Clk::now();
    for (int64_t q = 0; q < Q; q++) {
        auto s0 = Clk::now();                                   // the search starts (timed)
        int ty = (int)Y.at(q, 0); const double* z = &Z.v[q * Z.c];
        int size;
        if (ty == 0) { double k = z[3]; size = k < 3.17 ? 0 : k < 31.7 ? 1 : k < 317 ? 2 : 3; }
        else {
            double h = ty == 1 ? z[3] : z[4] * 0.08;
            int ia[3], ib[3]; double fr = 1;
            for (int j = 0; j < D; j++) {
                double a = (z[j] - h - lo) / w, b = (z[j] + h - lo) / w;
                a = a < 0 ? 0 : a > G ? G : a; b = b < 0 ? 0 : b > G ? G : b;
                ia[j] = (int)floor(a); ib[j] = (int)ceil(b); fr *= max(b - a, 1e-9) / max(ib[j] - ia[j], 1);
            }
            double e = 0;
            for (int c = 0; c < (1 << D); c++) {
                int idx = 0, sg = 1;
                for (int j = 0; j < D; j++) { int v = (c >> j & 1) ? ib[j] : (sg = -sg, ia[j]); idx = idx * G1 + v; }
                e += sg * sat[idx];
            }
            e *= min(1.0, fr);
            size = e < 9 ? 0 : e < 99 ? 1 : e < 999 ? 2 : e < 9999 ? 3 : e < 99999 ? 4 : 5;
        }
        int k = ty * 8 + size; long long v = ++visits[k];
        if (!started[k]) {
            started[k] = true; int j = -1;
            for (int d2 = 1; d2 < 8 && j < 0; d2++) { if (size - d2 >= 0 && started[ty * 8 + size - d2] && visits[ty * 8 + size - d2] > 0) j = ty * 8 + size - d2;
                                                       else if (size + d2 < 8 && started[ty * 8 + size + d2] && visits[ty * 8 + size + d2] > 0) j = ty * 8 + size + d2; }
            for (int a = 0; a < M; a++) order[k][a] = a;
            if (j >= 0) { double mu[M]; for (int a = 0; a < M; a++) mu[a] = cnt[j][a] > 0 ? sm[j][a] / cnt[j][a] : 1e300;
                          stable_sort(order[k], order[k] + M, [&](int x, int y) { return mu[x] < mu[y]; }); }
        }
        int inc = -1; double im = 1e300;
        for (int a = 0; a < M; a++) if (v - dead[k][a] > revive && cnt[k][a] > 0) { double m = sm[k][a] / cnt[k][a]; if (m < im) { im = m; inc = a; } }
        int untried = -1;
        for (int i = 0; i < M; i++) { int a = order[k][i]; if (v - dead[k][a] > revive && cnt[k][a] == 0) { untried = a; break; } }
        bool may = v >= (long long)pace * (probes[k] + 1);
        int pick;
        if (inc < 0) pick = untried >= 0 ? untried : order[k][0];
        else if (untried >= 0 && may) pick = untried;
        else {
            double best = 1e300; pick = inc;
            for (int a = 0; a < M; a++) if (v - dead[k][a] > revive && cnt[k][a] > 0) {
                double l = sm[k][a] / cnt[k][a] * (1 - alpha / sqrt(cnt[k][a])); if (l < best) { best = l; pick = a; } }
            if (pick != inc && !may) pick = inc;
        }
        int obs = pick; double tv = T.at(q, pick) / 1000.0;
        if (inc >= 0 && pick != inc) { probes[k]++; if (tv > bound * im) { obs = inc; tv = T.at(q, inc) / 1000.0; dead[k][pick] = v; } }
        auto s1 = Clk::now();                                   // the search ends (timed)
        SINK += chrono::duration<double, nano>(s1 - s0).count() * 1e-12;
        cnt[k][obs] += 1; sm[k][obs] += tv;
        if (window > 0) {
            auto& r = ring[k];
            if (fill[k] == W) { auto o = r[head[k]]; cnt[k][o.first] -= 1; sm[k][o.first] -= o.second; } else fill[k]++;
            r[head[k]] = {obs, tv}; head[k] = (head[k] + 1) % W;
        }
        SINK += pick;
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / Q);
    return 0;
}

// ---------------------------------------------------------------- tree ensembles exported from scikit-learn
// flat node layout: one contiguous array per tree, leaf values in a separate flat array
struct Node { int l, r, f, voff; double th; };
struct Tree { vector<Node> nd; vector<double> val; };
struct Ens { string kind; int ntrees = 0, nout = 0, groups = 1; double lr = 1, init = 0; vector<Tree> t; };
static Ens readens(const char* f) {
    ifstream in(f); Ens e; in >> e.kind >> e.groups >> e.ntrees >> e.nout >> e.lr >> e.init;
    e.t.resize((size_t)e.groups * e.ntrees);
    for (auto& t : e.t) {
        int n; in >> n; t.nd.resize(n); t.val.resize((size_t)n * e.nout);
        for (int i = 0; i < n; i++) {
            in >> t.nd[i].l >> t.nd[i].r >> t.nd[i].f >> t.nd[i].th; t.nd[i].voff = i * e.nout;
            for (int k = 0; k < e.nout; k++) in >> t.val[(size_t)i * e.nout + k];
        }
    }
    return e;
}
static inline const double* leaf(const Tree& t, const double* x) {
    const Node* nd = t.nd.data(); int n = 0;
    while (nd[n].l >= 0) n = (x[nd[n].f] <= nd[n].th) ? nd[n].l : nd[n].r;
    return &t.val[nd[n].voff];
}
int trees(const Ens& e, const Mat& X) {
    vector<double> acc(max(e.nout, e.groups));
    auto t0 = Clk::now();
    for (int64_t q = 0; q < X.r; q++) {
        const double* x = &X.v[q * X.c];
        if (e.kind == "class") {        // average class distributions, take argmax
            fill(acc.begin(), acc.end(), 0.0);
            for (auto& t : e.t) { const double* v = leaf(t, x); for (int k = 0; k < e.nout; k++) acc[k] += v[k]; }
            SINK += max_element(acc.begin(), acc.begin() + e.nout) - acc.begin();
        } else {                        // one boosted regressor per arm, take the arm with the smallest prediction
            double best = 1e300; int pick = 0;
            for (int g = 0; g < e.groups; g++) {
                double s = e.init; for (int k = 0; k < e.ntrees; k++) s += e.lr * leaf(e.t[g * e.ntrees + k], x)[0];
                if (s < best) { best = s; pick = g; }
            }
            SINK += pick;
        }
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / X.r);
    return 0;
}
int knn(const Mat& Tr, const Mat& X, int k) {  // Tr: features..., label in the last column
    int F = (int)Tr.c - 1;
    vector<pair<double, int>> d(Tr.r);
    auto t0 = Clk::now();
    for (int64_t q = 0; q < X.r; q++) {
        const double* x = &X.v[q * X.c];
        for (int64_t i = 0; i < Tr.r; i++) { double s = 0; for (int j = 0; j < F; j++) { double t = Tr.at(i, j) - x[j]; s += t * t; } d[i] = {s, (int)Tr.at(i, F)}; }
        nth_element(d.begin(), d.begin() + k, d.end());
        int votes[16] = {0}; for (int i = 0; i < k; i++) votes[d[i].second]++;
        SINK += max_element(votes, votes + 16) - votes;
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / X.r);
    return 0;
}
int mlp(const char* f, const Mat& X) {  // layers: in -> hidden (relu) -> out
    ifstream in(f); int a, h, o; in >> a >> h >> o;
    vector<double> W1(a * h), b1(h), W2(h * o), b2(o);
    for (auto& v : W1) in >> v; for (auto& v : b1) in >> v; for (auto& v : W2) in >> v; for (auto& v : b2) in >> v;
    vector<double> hid(h), out(o);
    auto t0 = Clk::now();
    for (int64_t q = 0; q < X.r; q++) {
        const double* x = &X.v[q * X.c];
        for (int j = 0; j < h; j++) { double s = b1[j]; for (int i = 0; i < a; i++) s += x[i] * W1[i * h + j]; hid[j] = s > 0 ? s : 0; }
        for (int k = 0; k < o; k++) { double s = b2[k]; for (int j = 0; j < h; j++) s += hid[j] * W2[j * o + k]; out[k] = s; }
        SINK += max_element(out.begin(), out.end()) - out.begin();
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / X.r);
    return 0;
}

int linear(const Mat& Wt, const Mat& X) {  // Wt: one row per arm, coefficients then intercept
    int F = (int)Wt.c - 1;
    auto t0 = Clk::now();
    for (int64_t q = 0; q < X.r; q++) {
        const double* x = &X.v[q * X.c]; double best = 1e300; int pick = 0;
        for (int a = 0; a < Wt.r; a++) { double s = Wt.at(a, F); for (int j = 0; j < F; j++) s += Wt.at(a, j) * x[j]; if (s < best) { best = s; pick = a; } }
        SINK += pick;
    }
    printf("%.2f\n", chrono::duration<double, nano>(Clk::now() - t0).count() / X.r);
    return 0;
}

int main(int argc, char** argv) {
    string mode = argv[1];
    if (mode == "linear") return linear(readmat(argv[2]), readmat(argv[3]));
    if (mode == "bucket") return bucketed(readmat(argv[2]), readmat(argv[3]), readmat(argv[4]), readmat(argv[5]), atof(argv[6]),
                                          atoi(argv[7]), atoi(argv[8]), atof(argv[9]), atoi(argv[10]));
    if (mode == "lazy") return lazy(readmat(argv[2]), readmat(argv[3]), readmat(argv[4]), readmat(argv[5]), atof(argv[6]), atof(argv[7]),
                                    atoi(argv[8]), atoi(argv[9]), atoi(argv[10]));
    if (mode == "online") return online(argv[2], readmat(argv[3]), readmat(argv[4]), readmat(argv[5]), atof(argv[6]), atof(argv[7]), atoi(argv[8]),
                                        argc > 9 && string(argv[9]) == "log");
    if (mode == "ctx") return ctxrule(readmat(argv[2]), readmat(argv[3]), readmat(argv[4]), atof(argv[5]), atof(argv[6]), atoi(argv[7]),
                                      atoi(argv[8]), atoi(argv[9]), atoi(argv[10]));
    if (mode == "trees") return trees(readens(argv[2]), readmat(argv[3]));
    if (mode == "knn") return knn(readmat(argv[2]), readmat(argv[3]), atoi(argv[4]));
    if (mode == "mlp") return mlp(argv[2], readmat(argv[3]));
    return 1;
}
