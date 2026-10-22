"""One acceptance rule shared by analysis, review and final verification."""


def validate_coverage(row):
    observed = row['observed']
    for feature in ['frozen_linear_captured', 'foreground_linear_reused',
                    'global_g2_reused', 'cache_reused']:
        assert observed[feature] is True, feature
    if row['configuration']['transform_packet_max_len'] is not None:
        assert observed['transform_packet_calls'] > 0
    else:
        assert observed['transform_packet_calls'] == 0
    assert observed['sparse_gpu_g1_points'] > 0
    assert observed['sparse_cpu_g1_points'] == 0
    if row['n'] == 256:
        assert observed['support_geometry_reused'] is True
        extra = row['extra_audit']
        assert extra['geometry_only']['correct'] is True
        assert extra['geometry_only']['hits'] > 0
        assert extra['linear_only']['correct'] is True
        assert extra['linear_only']['linear_reused'] is True
