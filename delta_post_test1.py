import sys, json, math
src = sys.argv[1] if len(sys.argv) > 1 else 'out1/set1_rows.json'
dst = sys.argv[2] if len(sys.argv) > 2 else 'out_delta/test1_deltahat.md'
rows = [r for r in json.load(open(src)) if r['aux'] == 'paper']
byN = {}
for r in rows: byN.setdefault(r['N'], {})[r['i']] = r
lines = ["| N | i | p=i-1 | gap λ_ph<λ̲_i | δ̄ (a priori) | δ̂ (Thm 1(iv)) | η̃² with δ̄ | I^lb with δ̄ | η̃² with δ̂ | I^lb with δ̂ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for N in sorted(byN):
    for i in sorted(byN[N]):
        r = byN[N][i]; p = i - 1
        if p == 0:
            dhat = 0.0; gap = True
        else:
            low = r['lam_i_low']
            gap = all(byN[N][k]['lam_ih'] < low for k in range(1, p + 1))
            dhat = math.sqrt(sum((1 - byN[N][k]['lam_ih'] / low) ** -2 * byN[N][k]['eta_res'] ** 2 / byN[N][k]['lam_ih']
                                 for k in range(1, p + 1))) if gap else float('inf')
        lo, up, ab, R2 = r['lam_i_low'], r['lam_ih'], r['alpha'], r['R2']
        if dhat < 1:
            et = max(-up * ab ** 2 + 0.5 * (math.sqrt(lo ** 2 + 4 * lo * (1 - dhat ** 2) * R2) - lo), 0.0)
        else:
            et = 0.0
        Ilb = (r['lam_ih'] - r['lam_i']) / et if et > 0 else float('inf')
        # consistency: dhat = dbar reproduces et_new (same formula)
        et_chk = max(-up * ab ** 2 + 0.5 * (math.sqrt(lo ** 2 + 4 * lo * (1 - r['dbar'] ** 2) * R2) - lo), 0.0) if r['dbar'] < 1 else 0.0
        assert abs(et_chk - r['et_new']) <= 1e-10 * max(1.0, r['et_new']), (N, i, et_chk, r['et_new'])
        f = lambda x: 'void' if x == float('inf') else f'{x:.3f}'
        lines.append(f"| {N} | {i} | {p} | {gap} | {r['dbar']:.4f} | {f(dhat) if dhat<1e9 else 'no gap'} | {r['et_new']:.4e} | {f(r['Ilb_new'])} | {et:.4e} | {f(Ilb)} |")
        print(lines[-1])
open(dst, 'w', encoding='utf-8').write("\n".join(lines) + "\n")
print("written", dst)
