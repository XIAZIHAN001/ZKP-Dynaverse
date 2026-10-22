import copy
import unittest
from coverage import validate_coverage


def valid_row():
    return {'n':256, 'observed':{'linear_reused':False,
        'frozen_linear_captured':True,'foreground_linear_reused':True,
        'global_g2_reused':True,'support_geometry_reused':True,
        'cache_reused':True,'transform_packet_calls':12,
        'sparse_gpu_g1_points':70,'sparse_cpu_g1_points':0},
        'configuration':{'transform_packet_max_len':4096},
        'extra_audit':{'geometry_only':{'hits':5,'correct':True},
                       'linear_only':{'linear_reused':True,'correct':True}}}


class CoverageContract(unittest.TestCase):
    def test_independent_linear_hit_does_not_require_masked_bundle_path(self):
        validate_coverage(valid_row())

    def test_same_support_sequence_requires_bundle_geometry_reuse(self):
        row=valid_row();row['observed']['support_geometry_reused']=False
        with self.assertRaises(AssertionError):validate_coverage(row)

    def test_enabled_geometry_without_a_hit_is_rejected(self):
        row=valid_row(); row['extra_audit']['geometry_only']['hits']=0
        with self.assertRaises(AssertionError):validate_coverage(row)

    def test_bundle_hit_cannot_replace_missing_independent_audit(self):
        row=valid_row(); row['observed']['linear_reused']=True
        row['observed']['support_geometry_reused']=True
        del row['extra_audit']['linear_only']
        with self.assertRaises((AssertionError,KeyError)):validate_coverage(row)

    def test_cpu_fallback_is_rejected(self):
        row=valid_row(); row['observed']['sparse_cpu_g1_points']=1
        with self.assertRaises(AssertionError):validate_coverage(row)

    def test_large_transform_may_use_tiled_route_without_packet(self):
        row=valid_row(); row['n']=65536
        row['configuration']['transform_packet_max_len']=None
        row['observed']['transform_packet_calls']=0
        row['extra_audit']={}
        validate_coverage(row)


if __name__=='__main__':unittest.main()
