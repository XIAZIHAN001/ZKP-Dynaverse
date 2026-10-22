"""Validate raw rows, paired timing, actual hits and policy before reporting."""
import json
from coverage import validate_coverage
from policy import validate_policy


def validate_performance(text,performance,sizes=(256,65536)):
    assert analyze_log(text,sizes)==performance,'derived performance differs from raw log'


def analyze_log(text,sizes=(256,65536)):
    lines=text.splitlines()
    assert lines.count(f'FULLOPT_OK cases={len(sizes)} measured_steps=2')==1
    rows=[json.loads(line[4:]) for line in lines if line.startswith('ROW ')]
    assert len(rows)==len(sizes)*11+(26 if 256 in sizes else 0)
    matrix=[]
    for case,n in enumerate(sizes):
        own=[r for r in rows if r.get('case')==case]
        setup=[r for r in own if r['kind']=='fullopt_setup']
        summary=[r for r in own if r['kind']=='fullopt_summary']
        calibrations=[r for r in rows if r['kind']=='fullopt_readback_calibration' and r['n']==n]
        assert len(setup)==len(summary)==len(calibrations)==1
        summary=summary[0];calibration=calibrations[0]
        updates=[r for r in own if r['kind']=='fullopt_update']
        features=[r for r in own if r['kind']=='fullopt_feature_probe']
        extra=[r for r in own if r['kind']=='fullopt_extra_audit']
        expected={(step,route) for step in range(4) for route in ['baseline','full_optimized']}
        assert len(updates)==8 and {(r['step'],r['route']) for r in updates}==expected
        assert len(features)==(24 if n==256 else 0)
        if n==256:
            assert {(r['step'],r['route']) for r in features}=={
                (step,route) for step in range(12) for route in ['baseline','full_optimized']}
        assert len(extra)==(2 if n==256 else 0)
        for row in own:
            assert row['n']==n
        for row in updates+features:
            assert row['correct'] is True and type(row['ns']) is int and row['ns']>0
            assert int(row['rebuild_units'])<=int(row['allowance'])
            assert row['sparse_cpu_g1_points']==0
            # A default paid-large rebuild is dense and may submit no sparse
            # work; the optimized foreground correction must reach the GPU.
            if row['route']=='full_optimized':assert row['sparse_gpu_g1_points']>0
        for row in updates:assert row['measured'] is (row['step']>=2)
        measured=[r for r in updates if r['measured']]
        totals={route:sum(r['ns'] for r in measured if r['route']==route)
                for route in ['baseline','full_optimized']}
        assert summary['baseline_update_ns']==totals['baseline']
        assert summary['optimized_update_ns']==totals['full_optimized']
        counts=summary['feature_counts'];cfg=summary['configuration']
        optimized=[r for r in updates+features if r['route']=='full_optimized']
        for key in ['linear_reused','frozen_linear_captured','foreground_linear_reused',
                    'global_g2_reused','support_geometry_reused','cache_reused_points',
                    'transform_packet_calls','sparse_gpu_g1_points','sparse_cpu_g1_points']:
            assert counts[key]==sum(int(r[key]) for r in optimized),key
        assert counts['cache_reused']==sum(r['cache_reused_points']>0 for r in optimized)
        observed={key:counts[key]>0 for key in ['linear_reused','frozen_linear_captured',
            'foreground_linear_reused','global_g2_reused','support_geometry_reused','cache_reused']}
        observed.update({key:counts[key] for key in ['transform_packet_calls',
            'sparse_gpu_g1_points','sparse_cpu_g1_points']})
        assert summary['baseline_cache_preparation'] is None
        assert summary['optimized_cache_preparation']['roots']==64
        for route in ['baseline','optimized']:
            plan=summary[route+'_resource']
            assert plan['upper_bytes']<=cfg['native_limit_bytes']
            assert plan['transaction_bytes']<=cfg['undo_bytes']
        assert summary['optimized_resource']['cache_readback_host_bytes']>0
        row={'case':case,'n':n,'setup_ns':setup[0]['ns'],
            'baseline_initial_ns':summary['baseline_initial_ns'],
            'optimized_initial_ns':summary['optimized_initial_ns'],
            'baseline_update_ns':totals['baseline'],'optimized_update_ns':totals['full_optimized'],
            'baseline_over_optimized':totals['baseline']/totals['full_optimized'],
            'feature_counts':counts,'observed':observed,'configuration':cfg,
            'extra_audit':{r['audit']:r for r in extra},'readback_calibration':calibration,
            'cache_preparation':{'baseline':None,'optimized':summary['optimized_cache_preparation']},
            'resource':{route:summary[route+'_resource'] for route in ['baseline','optimized']}}
        validate_coverage(row);validate_policy(n,cfg,calibration)
        matrix.append(row)
    return {'setup_mode':'TableBatchNormalize+GpuFrobenius63',
        'baseline_api':'IncrementalSession defaults + ForceGpu + same size-selected window',
        'optimized_api':'IncrementalSession size-selected bundle + bounded decode + calibrated readback',
        'cases':len(sizes),'measured_steps_per_route':2,'feature_audit_steps_n256':12,
        'matrix':matrix,'raw_rows':len(rows)}
