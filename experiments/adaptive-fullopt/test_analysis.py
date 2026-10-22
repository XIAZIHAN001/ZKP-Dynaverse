import copy
import json
from pathlib import Path
import unittest
from analysis import analyze_log,validate_performance

B=Path(__file__).resolve().parent


class RawEvidenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines=(B/'fixtures/n256.log').read_text(encoding='utf-8').splitlines()

    def mutate(self,kind,change):
        lines=list(self.lines)
        index=next(i for i,line in enumerate(lines) if line.startswith('ROW ')
                   and json.loads(line[4:])['kind']==kind)
        row=json.loads(lines[index][4:]);change(row)
        lines[index]='ROW '+json.dumps(row)
        return '\n'.join(lines)

    def test_real_completed_small_matrix_is_accepted(self):
        result=analyze_log('\n'.join(self.lines),(256,))
        self.assertEqual(result['matrix'][0]['extra_audit']['geometry_only']['hits'],5)
        self.assertTrue(result['matrix'][0]['extra_audit']['linear_only']['linear_reused'])

    def test_missing_paired_update_is_rejected(self):
        lines=list(self.lines)
        index=next(i for i,line in enumerate(lines) if '"kind":"fullopt_update"' in line)
        del lines[index]
        with self.assertRaises(AssertionError):analyze_log('\n'.join(lines),(256,))

    def test_declared_hits_cannot_exceed_actual_receipts(self):
        text=self.mutate('fullopt_summary',lambda r:r['feature_counts'].__setitem__('transform_packet_calls',999))
        with self.assertRaises(AssertionError):analyze_log(text,(256,))

    def test_false_correctness_is_rejected(self):
        text=self.mutate('fullopt_update',lambda r:r.__setitem__('correct',False))
        with self.assertRaises(AssertionError):analyze_log(text,(256,))

    def test_insufficient_undo_is_rejected(self):
        text=self.mutate('fullopt_summary',lambda r:r['configuration'].__setitem__('undo_bytes',1))
        with self.assertRaises(AssertionError):analyze_log(text,(256,))

    def test_readback_claim_requires_actual_decode_counts(self):
        text=self.mutate('fullopt_readback_calibration',lambda r:r.__setitem__('cofactor_decode_points_per_call',0))
        with self.assertRaises(AssertionError):analyze_log(text,(256,))

    def test_edited_performance_is_rejected_even_with_valid_observed_flags(self):
        text='\n'.join(self.lines);perf=analyze_log(text,(256,))
        perf['matrix'][0]['baseline_over_optimized']=999999
        with self.assertRaises(AssertionError):validate_performance(text,perf,(256,))


if __name__=='__main__':unittest.main()
