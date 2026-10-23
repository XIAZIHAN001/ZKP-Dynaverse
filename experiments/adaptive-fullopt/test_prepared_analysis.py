import json
from pathlib import Path
import unittest
from prepared_analysis import analyze


class PreparedEvidenceContract(unittest.TestCase):
    def test_completed_h200_tiled_partial_prefix_scan_is_accepted(self):
        text = (Path(__file__).parent / 'fixtures/prepared-h200-20261023.log').read_text(encoding='utf-8')
        result = analyze(text)
        self.assertEqual(result['n'], 65536)
        self.assertEqual([r['cap'] for r in result['routes']], [64,256,1024,4096])
        self.assertEqual(result['routes'][-1]['divisors_formed_per_update'], 122880)

    def setUp(self):
        self.lines = (Path(__file__).parent / 'fixtures/prepared-n256.log').read_text(encoding='utf-8').splitlines()

    def changed(self, kind, change):
        lines = list(self.lines)
        index = next(i for i, line in enumerate(lines)
                     if line.startswith('ROW ') and json.loads(line[4:])['kind'] == kind)
        row = json.loads(lines[index][4:])
        change(row)
        lines[index] = 'ROW ' + json.dumps(row)
        return '\n'.join(lines)

    def test_completed_real_paired_scan_is_accepted(self):
        result = analyze('\n'.join(self.lines))
        self.assertEqual(result['n'], 256)
        self.assertEqual(result['measured_steps'], 2)
        self.assertEqual(result['routes'][0]['cap'], 64)
        self.assertEqual(result['routes'][1]['cap'], 256)
        self.assertEqual(result['routes'][1]['divisors_formed_per_update'], 0)

    def test_completed_three_route_medium_scan_is_accepted(self):
        text = (Path(__file__).parent / 'fixtures/prepared-n1024.log').read_text(encoding='utf-8')
        result = analyze(text)
        self.assertEqual(result['n'], 1024)
        self.assertEqual([r['cap'] for r in result['routes']], [64,256,1024])
        self.assertEqual(result['routes'][2]['divisors_formed_per_update'], 0)

    def test_missing_update_or_terminal_is_rejected(self):
        index = next(i for i, line in enumerate(self.lines) if '"kind":"prepared_update"' in line)
        for lines in [self.lines[:index]+self.lines[index+1:], self.lines[:-1]]:
            with self.assertRaises(AssertionError): analyze('\n'.join(lines))

    def test_duplicate_pair_or_unknown_event_is_rejected(self):
        for line in [next(x for x in self.lines if '"kind":"prepared_pair"' in x), 'ROW {"kind":"unknown"}']:
            with self.assertRaises(AssertionError): analyze('\n'.join(self.lines+[line]))

    def test_incomplete_or_false_correctness_is_rejected(self):
        for key, value in [('correct',False),('pending',True),('swapped',False),('generation',999),('sparse_cpu_g1_points',1)]:
            with self.assertRaises(AssertionError):
                analyze(self.changed('prepared_update', lambda r:r.__setitem__(key,value)))

    def test_missing_real_hit_or_reduced_equations_is_rejected(self):
        for key in ['equations','g2_prepare_builds','divisors_formed','final_exponentiations']:
            with self.assertRaises(AssertionError):
                analyze(self.changed('prepared_update', lambda r:r['cache'].__setitem__(key,0)))

    def test_nested_timing_is_not_added_to_parent_and_cannot_exceed_it(self):
        text = self.changed('prepared_update',lambda r:r['timings']['cache_ns'].__setitem__('miller',r['ns']*2))
        with self.assertRaises(AssertionError): analyze(text)

    def test_disjoint_preparation_children_are_nonnegative_and_fit_together(self):
        for key in ['live_linear_ns','delta_proof_ns','rebase_ns']:
            for value in [-1,10**20]:
                text = self.changed('prepared_update',lambda r:r['timings'].__setitem__(key,value))
                with self.assertRaises(AssertionError): analyze(text)

    def test_over_budget_or_changed_work_is_rejected(self):
        for text in [self.changed('prepared_initial',lambda r:r['resource'].__setitem__('upper_bytes',3<<30)),
                     self.changed('prepared_initial',lambda r:r['resource'].__setitem__('transaction_bytes',1<<30)),
                     self.changed('prepared_update',lambda r:r.__setitem__('fr_elements',0)),
                     self.changed('prepared_update',lambda r:r.__setitem__('allowance','0'))]:
            with self.assertRaises(AssertionError): analyze(text)

    def test_common_resource_understatement_and_self_reported_quota_cannot_pass(self):
        for mode in ['resource','missing_resource','quota']:
            lines = []
            for line in self.lines:
                if line.startswith('ROW '):
                    r = json.loads(line[4:])
                    if r['kind'] == 'prepared_initial':
                        if mode == 'resource': r['resource']['upper_bytes'] = r['host_upper_bytes']
                        if mode == 'missing_resource': del r['resource']['workspace_reserve']
                        if mode == 'quota': r['budget']['q'] = '0'
                    if r['kind'] == 'prepared_update' and mode == 'quota': r['allowance'] = str(10**20)
                    line = 'ROW '+json.dumps(r)
                lines.append(line)
            with self.assertRaises(AssertionError): analyze('\n'.join(lines))

    def test_swapped_raw_step_groups_are_rejected(self):
        lines = list(self.lines)
        groups = [[i for i,line in enumerate(lines) if line.startswith('ROW ')
                   and json.loads(line[4:]).get('kind') == 'prepared_update'
                   and json.loads(line[4:]).get('step') == step] for step in [0,1]]
        for a,b in zip(*groups): lines[a],lines[b] = lines[b],lines[a]
        with self.assertRaises(AssertionError): analyze('\n'.join(lines))

    def test_calibration_and_readback_do_not_follow_prepared_capacity(self):
        for text in [self.changed('fullopt_readback_calibration',lambda r:r.__setitem__('points',256)),
                     self.changed('prepared_update',lambda r:r['cache'].__setitem__('max_read_roots',256))]:
            with self.assertRaises(AssertionError): analyze(text)

    def test_pair_proof_and_witness_are_required(self):
        with self.assertRaises(AssertionError):
            analyze(self.changed('prepared_pair',lambda r:r.__setitem__('witness_equal',False)))


if __name__ == '__main__': unittest.main()
