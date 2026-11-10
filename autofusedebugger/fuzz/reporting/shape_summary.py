"""Shape-mode grouping for valid timings; do not compare unrelated graphs as pairs."""
import math
import statistics


def shape_summary(cases):
    rows=[]
    for mode in ('static','dynamic'):
        group=[c for c in cases if c.get('shape_mode','static')==mode]
        valid=[c for c in group if c.get('performance_status') in ('pass','fail')]
        row={'mode':mode,'total':len(group),'pass':sum(c.get('performance_status')=='pass' for c in group),
             'fail':sum(c.get('performance_status')=='fail' for c in group),'error':len(group)-len(valid)}
        row['pass_rate']=row['pass']/len(valid) if valid else None
        speeds=[c['ascendc_speedup'] for c in valid if c.get('ascendc_speedup') and math.isfinite(c['ascendc_speedup'])]
        row['geomean_speedup']=math.exp(statistics.mean(math.log(s) for s in speeds)) if speeds else None
        for backend in ('ascendc','triton-ascend'):
            times=[c['backends'][backend]['mean_kernel_us'] for c in valid]
            row[backend]=statistics.median(times) if times else None
        rows.append(row)
    return rows


def render_shape_summary(cases):
    def fmt(value): return '—' if value is None else f'{value:.3f}'
    body=''.join(f"<tr><td>{r['mode']}</td><td>{r['total']}</td><td>{r['pass']}</td><td>{r['fail']}</td><td>{r['error']}</td>"
        f"<td>{'—' if r['pass_rate'] is None else format(r['pass_rate']*100,'.1f')+'%'}</td>"
        f"<td>{fmt(r['ascendc'])}</td><td>{fmt(r['triton-ascend'])}</td><td>{fmt(r['geomean_speedup'])}</td></tr>" for r in shape_summary(cases))
    return '<h3>静态 / 动态 shape 分组统计</h3><p>通过率分母为有效对比数；耗时取有效 case 的中位数（μs/步），加速比为 Triton / AscendC 的几何平均。两组是不同随机图，不能据此推断动态编译相对静态编译的开销。</p><div style="overflow-x:auto"><table><thead><tr><th>模式</th><th>总数</th><th>pass</th><th>fail</th><th>error</th><th>有效通过率</th><th>AscendC 中位数</th><th>Triton 中位数</th><th>几何平均加速比</th></tr></thead><tbody>'+body+'</tbody></table></div><br>'
