"""Raw acceptance; preserve independent equations and charge construction."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adaptive-fullopt'))
from policy import expected_policy,validate_policy
from prepared_analysis import nonnegative,detail,REBUILD,CACHE,LINEAR,SPARSE,RESOURCE


def analyze(text):
    rows=[json.loads(line[4:]) for line in text.splitlines() if line.startswith('ROW ')]
    assert rows and rows[0]['kind']=='divisor_start'
    start=rows[0];n=start['n'];gpu=start['backend']=='gpu';policy=expected_policy(n)
    assert start['schema']==1 and start['backend'] in ['cpu','gpu'] and n in [256,1024,65536]
    assert start['steps']==4 and start['measured_steps']==2 and start['construction_chunk_roots']==64
    assert start['prepared_roots']==start['readback_roots']==64 and start['row_batch']==16
    assert start['window']==policy['window'] and start['packet_cap']==policy['transform_packet_max_len']
    assert start['native_owner_limit_bytes']==2<<30 and start['undo_bytes']==policy['undo_bytes']
    kinds=['divisor_start','divisor_setup']+(['fullopt_readback_calibration'] if gpu else [])
    kinds+=['divisor_initial']*2+['divisor_pair']+(['divisor_update']*2+['divisor_pair'])*4+['divisor_dense']
    assert [r['kind'] for r in rows]==kinds and all(r['n']==n for r in rows)
    marker=f'FULL_DIVISOR_OK backend={start["backend"]} n={n} measured_steps=2'
    assert [x for x in text.splitlines() if x.startswith('FULL_DIVISOR_OK')]==[marker]
    assert nonnegative(rows[1]['ns'])>0
    affine=False
    if gpu:
        calibration=rows[2];affine=calibration['device_affine_readback_selected']
        validate_policy(n,dict(policy,bounded_g1_decode=True,device_affine_readback=affine),calibration)
    initials=[r for r in rows if r['kind']=='divisor_initial']
    assert [r['route'] for r in initials]==[0,1]
    for route,r in enumerate(initials):
        assert r['correct'] is True and r['affine'] is affine and r['prepared_roots']==64
        assert nonnegative(r['ns'])>0 and nonnegative(r['verify_ns'])>0
        assert 0<r['prepared_precompute_ns']<=r['ns']
        assert r['initial']=={'equations':3*n,'divisors_formed':0 if route else n,'hits':n if route else 0}
        table=r['table']
        # Pinned Rust1.89 x86_64 layout: rootFr32 + G2Affine136 =168;
        # owner/domain/powers/Vec/stats header368. These are requested bytes.
        table_bound=n*168+368+64*2048+4096
        if route:
            assert set(table)=={'roots','construction_divisors','retained_bytes','host_upper_bytes','precompute_ns'}
            assert table['roots']==table['construction_divisors']==n
            assert table['retained_bytes']==n*168+368 and table['host_upper_bytes']==table_bound
            assert nonnegative(table['precompute_ns'])>0
            assert table['precompute_ns']+r['prepared_precompute_ns']<=r['ns']
        else:assert table is None
        p=r['resource'];assert set(p)==RESOURCE|{'cache_divisor_host_bytes'}
        for value in p.values():nonnegative(value)
        assert p['workspace_reserve']==256<<20 and p['frozen_linear_bytes']==16128
        assert p['global_g2_host_bytes']==128*512+4096
        assert p['global_g2_workspace_bytes']==(64<<20)+128*1024
        assert p['cache_readback_host_bytes']==64*512+4096
        assert p['cache_preparation_host_bytes']==(64+8)*65536+64*512+4096
        assert p['cache_divisor_host_bytes']==(table_bound if route else 0)
        assert p['tile_len']==1<<((n.bit_length()+1)//2)
        if gpu:assert p['transaction_bytes']==1472*n
        assert 0<p['transaction_bytes']<=start['undo_bytes']
        upper=n*16384+sum(p[k] for k in ['workspace_reserve','frozen_linear_bytes','global_g2_host_bytes',
            'cache_preparation_host_bytes','cache_readback_host_bytes','cache_divisor_host_bytes'])
        assert p['upper_bytes']==upper<=2<<30 and 0<p['resident_bytes']<=upper
        for key in p:
            if key not in ['upper_bytes','cache_divisor_host_bytes']:assert p[key]==initials[0]['resource'][key]
        b=r['budget'];assert set(b)=={'q','total','max_chunk','large'}
        q,total,chunk,large=(int(b[k]) for k in ['q','total','max_chunk','large'])
        h=start['window']//2
        assert min(q,total,chunk,large)>0 and q>chunk
        assert q==max((total+h-1)//h+chunk+1,(large+h)//(h+1))
        assert h*(q-chunk)>=total and (h+1)*q>=large
    assert int(initials[1]['budget']['total'])>int(initials[0]['budget']['total'])
    pairs=[r for r in rows if r['kind']=='divisor_pair']
    assert [r['step'] for r in pairs]==[-1,0,1,2,3]
    for r in pairs:
        assert r['proof_equal'] is r['witness_equal'] is True
        if r['step']>=0:assert r['arithmetic_equal'] is True
    updates=[r for r in rows if r['kind']=='divisor_update']
    for step in range(4):
        first=(6 if gpu else 5)+step*3
        group=rows[first:first+2]
        assert all(r['step']==step for r in rows[first:first+3])
        order=[0,1] if (step//2+step)%2==0 else [1,0]
        assert [r['route'] for r in group]==order and [r['order'] for r in group]==[0,1]
        for r in group:
            route=r['route'];c=r['cache'];t=r['timings'];fallback=2*(n-64)
            assert r['correct'] is r['large_batch'] is r['swapped'] is True
            assert r['pending'] is r['restarted'] is False
            assert r['generation']==r['anchor_generation']==step+1 and r['changed']==16
            assert r['measured'] is (step>=2)
            assert int(r['allowance'])==16*int(initials[route]['budget']['q'])
            assert 0<int(r['rebuild_units'])<=int(r['allowance'])
            assert r['resident_bytes']==initials[route]['resource']['resident_bytes']
            assert r['frozen_linear_captured'] is r['foreground_linear_reused'] is r['global_g2_reused'] is True
            assert r['cache_reused_points']==n and r['global_g2_inputs']==(0 if step%2 else 8)
            assert (r['transform_packet_calls']>0) is (start['packet_cap'] is not None)
            assert r['sparse_cpu_g1_points']==0 if gpu else r['sparse_gpu_g1_points']==0
            assert r['sparse_gpu_g1_points']>0 if gpu else r['sparse_cpu_g1_points']>0
            assert c['equations']==c['miller_calls']==c['final_exponentiations']==2*n
            assert c['pairing_terms']==4*n and c['g2_prepare_builds']==fallback*5//4
            assert c['g2_prepare_builds']+c['g2_prepare_hits']==4*n
            assert c['divisors_formed']==(0 if route else fallback)
            assert c['divisor_cache_hits']==(fallback if route else 0)
            assert c['read_points']==2*n and c['read_batches']==2*n//64 and c['max_read_roots']==64
            assert nonnegative(r['ns'])>0 and nonnegative(r['verify_ns'])>0
            assert t['enabled'] is True and 0<t['total_ns']<=r['ns']
            assert sum(nonnegative(t[k]) for k in ['normalization_ns','constraints_ns','preparation_ns','publication_ns'])<=t['total_ns']
            inner=detail(t['rebuild_ns'],REBUILD)+sum(nonnegative(t[k]) for k in ['live_linear_ns','delta_proof_ns','rebase_ns'])
            assert inner<=t['preparation_ns']
            assert detail(t['cache_ns'],CACHE)<=t['rebuild_ns']['cache_check']
            assert detail(t['linear_ns'],LINEAR)<=t['rebuild_ns']['linear_msm']
            assert detail(t['delta_detail_ns'],SPARSE)<=t['delta_proof_ns']
            assert detail(t['rebase_detail_ns'],SPARSE)<=t['rebase_ns']
        for key in ['fr_elements','g1_elements','sparse_gpu_g1_points','sparse_cpu_g1_points','transform_packet_calls','transform_packet_elements']:
            assert group[0][key]==group[1][key],key
        byroute={r['route']:r for r in group}
        assert int(byroute[1]['rebuild_units'])>int(byroute[0]['rebuild_units'])
    assert rows[-1]['correct'] is True and rows[-1]['ns']>0
    routes=[]
    for route,initial in enumerate(initials):
        own=[r for r in updates if r['route']==route];measured=[r for r in own if r['measured']]
        table=initial['table']
        routes.append({'route':route,'initial_ns':initial['ns'],'precompute_ns':table['precompute_ns'] if table else 0,
            'construction_divisors':n if table else 0,'retained_bytes':table['retained_bytes'] if table else 0,
            'resource':initial['resource'],'budget':initial['budget'],'initial_check':initial['initial'],
            'two_update_ns':sum(r['ns'] for r in measured),'all_four_update_ns':sum(r['ns'] for r in own),
            'route_phase_sum_ns':rows[1]['ns']+initial['ns']+sum(r['ns'] for r in own),
            'external_reference_ns':initial['verify_ns']+sum(r['verify_ns'] for r in own),
            'divisors_formed_per_update':0 if route else 2*(n-64),
            'rebuild_ns':{k:sum(r['timings']['rebuild_ns'][k] for r in measured) for k in REBUILD},
            'cache_ns':{k:sum(r['timings']['cache_ns'][k] for r in measured) for k in CACHE}})
    return {'n':n,'backend':start['backend'],'measured_steps':2,'setup_ns':rows[1]['ns'],
        'cpu_dense_oracle_ns':rows[-1]['ns'],'two_update_control_over_cached':routes[0]['two_update_ns']/routes[1]['two_update_ns'],
        'phase_sum_control_over_cached':routes[0]['route_phase_sum_ns']/routes[1]['route_phase_sum_ns'],
        'routes':routes,'raw_rows':len(rows),'scope':'two measured update samples; phase sums are not independent E2E wall'}


if __name__=='__main__':
    print(json.dumps(analyze(Path(sys.argv[1]).read_text(encoding='utf-8')),indent=2))
