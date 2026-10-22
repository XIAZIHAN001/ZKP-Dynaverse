"""Acceptance model for the external Rust size-policy; no protocol defaults change."""


def expected_policy(n):
    assert isinstance(n,int) and 64<=n<=65536 and n&(n-1)==0
    window=6 if n<=512 else 8 if n<=4096 else 16
    undo=max(64<<20,1<<(1472*n-1).bit_length())
    return {'window':window,'transform_packet_max_len':4096 if n<=4096 else None,
            'undo_bytes':undo,'native_limit_bytes':2<<30,
            'rebuild_prepared_cache_roots':64,'rebuild_cache_readback_roots':64}


def validate_policy(n,configuration,calibration):
    for key,value in expected_policy(n).items():assert configuration[key]==value,key
    assert configuration['bounded_g1_decode'] is True
    assert calibration['n']==n and calibration['correct'] is True
    assert calibration['points']==64 and calibration['samples_per_route']==6
    assert calibration['cofactor_decode_points_per_call']==64
    assert calibration['full_decode_points_per_call']==0
    assert calibration['host_d2h_bytes']==64*96
    assert calibration['device_d2h_bytes']==64*64
    host=calibration['host_bounded_median_ns'];device=calibration['device_bounded_median_ns']
    assert host>0 and device>0
    selected=n>=1024 and device*100<host*97
    assert calibration['device_affine_readback_selected'] is selected
    assert configuration['device_affine_readback'] is selected
