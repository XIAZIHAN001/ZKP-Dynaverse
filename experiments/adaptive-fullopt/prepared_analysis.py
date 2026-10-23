"""Audit paired raw events; nested timing scopes are never added twice."""
import json
from policy import expected_policy, validate_policy

REBUILD = ['materialize','linear_msm','fr_transform','g1_transform','polynomial','memory','cache_check','finish']
CACHE = ['read_points','g1_difference','g2_divisor','g2_prepare','miller','final_exponentiation','undo_store','unchanged_copy']
LINEAR = ['fixed_g1','global_s5_g2','frozen_copy','quotient_g1']
SPARSE = ['fixed_linear','geometry','interpolation_cauchy','local_arithmetic','g1_commit_opening','g2_commit','proof_validation']
RESOURCE = {'upper_bytes','resident_bytes','workspace_reserve','transaction_bytes','tile_len',
            'frozen_linear_bytes','global_g2_host_bytes','global_g2_workspace_bytes',
            'cache_preparation_host_bytes','cache_readback_host_bytes'}


def nonnegative(value):
    assert type(value) is int and value >= 0
    return value


def detail(values, names):
    assert set(values) == set(names)
    return sum(nonnegative(value) for value in values.values())


def analyze(text):
    rows = [json.loads(line[4:]) for line in text.splitlines() if line.startswith('ROW ')]
    assert rows and rows[0]['kind'] == 'prepared_start'
    start = rows[0]
    n, caps = start['n'], start['caps']
    policy = expected_policy(n)
    assert start['schema'] == 1 and start['steps'] == 4 and start['measured_steps'] == 2
    assert len(caps) >= 2 and caps[0] == 64 and caps == sorted(set(caps))
    assert all(cap in [64,256,1024,4096] for cap in caps)
    assert start['actual_caps'] == [min(n,cap) for cap in caps]
    assert len(set(start['actual_caps'])) == len(caps)
    assert start['window'] == policy['window'] and start['packet_cap'] == policy['transform_packet_max_len']
    assert start['undo_bytes'] == policy['undo_bytes'] and start['native_owner_limit_bytes'] == 2<<30
    assert start['readback_roots'] == start['calibration_points'] == 64 and start['row_batch'] == 16
    assert start['setup_seed'] == 2026102220+n
    expected_kinds = ['prepared_start','prepared_setup','fullopt_readback_calibration']
    expected_kinds += ['prepared_initial']*len(caps)+['prepared_pair']
    expected_kinds += (['prepared_update']*len(caps)+['prepared_pair'])*4+['prepared_dense']
    assert [r['kind'] for r in rows] == expected_kinds
    assert all(r['n'] == n for r in rows)
    marker = f'PREPARED_CACHE_OK n={n} routes={len(caps)} measured_steps=2'
    assert text.splitlines().count(marker) == 1
    assert [line for line in text.splitlines() if line.startswith('PREPARED_CACHE_OK')] == [marker]
    setup, calibration = rows[1:3]
    assert nonnegative(setup['ns']) > 0
    configuration = dict(policy, bounded_g1_decode=True,
                         device_affine_readback=calibration['device_affine_readback_selected'])
    validate_policy(n, configuration, calibration)
    initials = rows[3:3+len(caps)]
    assert [r['cap'] for r in initials] == caps
    baseline_plan = initials[0]['resource']
    for initial, cap in zip(initials,caps):
        assert initial['correct'] is True and initial['affine'] is calibration['device_affine_readback_selected']
        assert nonnegative(initial['ns']) > 0 and nonnegative(initial['verify_ns']) > 0
        roots = min(n,cap)
        host = (roots+8)*65536+roots*512+4096
        assert initial['roots'] == roots and initial['prepared_entries'] == roots+1
        assert initial['host_upper_bytes'] == host and 0 < initial['retained_bytes'] <= host
        assert 0 < initial['precompute_ns'] <= initial['ns']
        p = initial['resource']
        assert set(p) == RESOURCE
        assert p['upper_bytes'] <= 2<<30 and p['transaction_bytes'] <= start['undo_bytes']
        assert p['cache_preparation_host_bytes'] == host and p['cache_readback_host_bytes'] > 0
        assert initial['resident_bytes'] == p['resident_bytes']
        assert 0 < p['resident_bytes'] <= p['upper_bytes']
        assert p['workspace_reserve'] == 256<<20
        # FixedLinear's pinned x86_64 layout, unchanged in this experiment.
        assert p['frozen_linear_bytes'] == 16128
        assert p['global_g2_host_bytes'] == min(n,128)*512+4096
        assert p['global_g2_workspace_bytes'] == (64<<20)+min(n,128)*1024
        assert p['cache_readback_host_bytes'] == 64*512+4096
        assert p['transaction_bytes'] == 1472*n
        assert p['tile_len'] == 1<<((n.bit_length()+1)//2)
        expected_upper = n*16384 + sum(p[key] for key in ['workspace_reserve','frozen_linear_bytes',
            'global_g2_host_bytes','cache_preparation_host_bytes','cache_readback_host_bytes'])
        assert p['upper_bytes'] == expected_upper
        for key in p:
            nonnegative(p[key])
            if key not in ['upper_bytes','cache_preparation_host_bytes']:
                assert p[key] == baseline_plan[key]
        assert p['upper_bytes']-host == baseline_plan['upper_bytes']-baseline_plan['cache_preparation_host_bytes']
        assert initial['budget'] == initials[0]['budget']
        budget = initial['budget']
        assert set(budget) == {'q','total','max_chunk','large'}
        q,total,chunk,large = (int(budget[k]) for k in ['q','total','max_chunk','large'])
        assert min(q,total,chunk,large) > 0
        h = start['window']//2
        assert q == max((total+h-1)//h+chunk+1,(large+h)//(h+1))
        assert q > chunk and h*(q-chunk) >= total and (h+1)*q >= large
    pairs = [r for r in rows if r['kind'] == 'prepared_pair']
    assert [r['step'] for r in pairs] == [-1,0,1,2,3]
    for pair in pairs:
        assert pair['proof_equal'] is pair['witness_equal'] is True
        if pair['step'] >= 0: assert pair['work_equal'] is True
    updates = [r for r in rows if r['kind'] == 'prepared_update']
    for step in range(4):
        first = 4+len(caps)+step*(len(caps)+1)
        assert all(r['step'] == step for r in rows[first:first+len(caps)+1])
        group = [r for r in updates if r['step'] == step]
        order = caps if (step//2+step)%2 == 0 else caps[::-1]
        assert [r['cap'] for r in group] == order
        assert [r['order'] for r in group] == list(range(len(caps)))
        for r in group:
            cap = r['cap']
            assert r['correct'] is r['large_batch'] is r['swapped'] is True
            assert r['pending'] is r['restarted'] is False
            assert r['changed'] == 16 and r['generation'] == r['anchor_generation'] == step+1
            assert r['measured'] is (step >= 2)
            assert r['family'] == ('addition' if step%2 else 'multiplier')
            assert 0 < int(r['rebuild_units']) <= int(r['allowance'])
            assert int(r['allowance']) == r['changed']*int(initials[0]['budget']['q'])
            assert r['frozen_linear_captured'] is r['foreground_linear_reused'] is r['global_g2_reused'] is True
            assert r['global_g2_inputs'] == (0 if step%2 else 8)
            assert r['cache_reused_points'] == n and r['sparse_cpu_g1_points'] == 0
            assert r['sparse_gpu_g1_points'] > 0
            assert (r['transform_packet_calls'] > 0) is (start['packet_cap'] is not None)
            assert r['resident_bytes'] == initials[caps.index(cap)]['resident_bytes']
            assert nonnegative(r['ns']) > 0 and nonnegative(r['verify_ns']) > 0
            c = r['cache']
            assert c['equations'] == c['miller_calls'] == c['final_exponentiations'] == 2*n
            assert c['pairing_terms'] == 4*n
            fallback = 2*(n-min(n,cap))
            assert c['divisors_formed'] == fallback and c['g2_prepare_builds'] == fallback*5//4
            assert c['g2_prepare_builds']+c['g2_prepare_hits'] == 4*n
            assert c['read_points'] == 2*n and c['max_read_roots'] == 64
            assert c['read_batches'] == 2*n//64
            assert min(n,cap)+1 <= c['prepared_peak_entries'] <= min(n,cap)+6
            t = r['timings']
            assert t['enabled'] is True and 0 < t['total_ns'] <= r['ns']
            top = sum(nonnegative(t[k]) for k in ['normalization_ns','constraints_ns','preparation_ns','publication_ns'])
            assert top <= t['total_ns']
            rebuild = detail(t['rebuild_ns'], REBUILD)
            children = sum(nonnegative(t[k]) for k in ['live_linear_ns','delta_proof_ns','rebase_ns'])
            assert rebuild+children <= t['preparation_ns']
            assert detail(t['cache_ns'], CACHE) <= t['rebuild_ns']['cache_check']
            assert detail(t['linear_ns'], LINEAR) <= t['rebuild_ns']['linear_msm']
            assert detail(t['delta_detail_ns'], SPARSE) <= nonnegative(t['delta_proof_ns'])
            assert detail(t['rebase_detail_ns'], SPARSE) <= nonnegative(t['rebase_ns'])
        for key in ['fr_elements','g1_elements','rebuild_units','allowance','foreground_linear_work',
                    'cache_reused_points','sparse_gpu_g1_points','transform_packet_calls','transform_packet_elements']:
            assert len({r[key] for r in group}) == 1, key
    assert rows[-1]['correct'] is True and rows[-1]['ns'] > 0
    routes = []
    base_ns = sum(r['ns'] for r in updates if r['cap']==64 and r['measured'])
    for initial, cap in zip(initials,caps):
        measured = [r for r in updates if r['cap']==cap and r['measured']]
        elapsed = sum(r['ns'] for r in measured)
        routes.append({'cap':cap,'actual_roots':min(n,cap),'initial_ns':initial['ns'],
            'precompute_ns':initial['precompute_ns'],'retained_bytes':initial['retained_bytes'],
            'resource':initial['resource'],'two_update_ns':elapsed,'control_over_candidate':base_ns/elapsed,
            'divisors_formed_per_update':2*(n-min(n,cap)),
            'rebuild_ns':{key:sum(r['timings']['rebuild_ns'][key] for r in measured) for key in REBUILD},
            'cache_ns':{key:sum(r['timings']['cache_ns'][key] for r in measured) for key in CACHE}})
    return {'n':n,'measured_steps':2,'setup_ns':setup['ns'],'readback_calibration':calibration,
            'routes':routes,'raw_rows':len(rows),'scope':'two measured updates; nested scopes are separate'}


if __name__ == '__main__':
    import sys
    from pathlib import Path
    print(json.dumps(analyze(Path(sys.argv[1]).read_text(encoding='utf-8')),indent=2))
