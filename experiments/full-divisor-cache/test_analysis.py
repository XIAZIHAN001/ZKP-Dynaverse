import json
from pathlib import Path
import unittest
from analysis import analyze


class FullDivisorEvidence(unittest.TestCase):
    def test_cpu_and_medium_gpu_fixtures_are_accepted(self):
        for backend,n in [('cpu',256),('cpu',1024),('gpu',1024)]:
            text=(Path(__file__).parent/f'fixtures/{backend}-n{n}.log').read_text(encoding='utf-8')
            result=analyze(text)
            self.assertEqual(result['backend'],backend)
            self.assertEqual(result['n'],n)

    def setUp(self):
        self.lines=(Path(__file__).parent/'fixtures/gpu-n256.log').read_text(encoding='utf-8').splitlines()

    def changed(self,kind,change):
        lines=list(self.lines)
        i=next(i for i,line in enumerate(lines) if line.startswith('ROW ') and json.loads(line[4:])['kind']==kind)
        row=json.loads(lines[i][4:]);change(row);lines[i]='ROW '+json.dumps(row)
        return '\n'.join(lines)

    def test_real_gpu_pair_is_accepted_and_reports_cost_separately(self):
        r=analyze('\n'.join(self.lines))
        self.assertEqual(r['n'],256)
        self.assertEqual(r['measured_steps'],2)
        self.assertGreater(r['routes'][1]['precompute_ns'],0)
        self.assertEqual(r['routes'][1]['construction_divisors'],256)
        self.assertEqual(r['routes'][1]['divisors_formed_per_update'],0)

    def test_missing_pair_update_or_terminal_is_rejected(self):
        for kind in ['divisor_update','divisor_pair','FULL_DIVISOR_OK']:
            i=next(i for i,line in enumerate(self.lines) if kind in line)
            with self.assertRaises(AssertionError):analyze('\n'.join(self.lines[:i]+self.lines[i+1:]))

    def test_false_correctness_and_unfinished_work_are_rejected(self):
        for k,v in [('correct',False),('swapped',False),('pending',True),('generation',99),('allowance','0')]:
            with self.assertRaises(AssertionError):analyze(self.changed('divisor_update',lambda r:r.__setitem__(k,v)))

    def test_lower_equations_or_missing_real_work_is_rejected(self):
        for k in ['equations','miller_calls','final_exponentiations','divisors_formed','g2_prepare_builds']:
            with self.assertRaises(AssertionError):analyze(self.changed('divisor_update',lambda r:r['cache'].__setitem__(k,0)))

    def test_hidden_table_construction_or_initial_work_is_rejected(self):
        for mode in ['precompute','construct','initial','host']:
            lines=[]
            for line in self.lines:
                if line.startswith('ROW '):
                    r=json.loads(line[4:])
                    if r['kind']=='divisor_initial' and r['route']==1:
                        if mode=='precompute':r['table']['precompute_ns']=0
                        if mode=='construct':r['table']['construction_divisors']=0
                        if mode=='initial':r['initial']['hits']=0
                        if mode=='host':r['resource']['cache_divisor_host_bytes']=0
                    line='ROW '+json.dumps(r)
                lines.append(line)
            with self.assertRaises(AssertionError):analyze('\n'.join(lines))

    def test_bad_nested_timing_or_forged_parent_is_rejected(self):
        for key,value in [('live_linear_ns',-1),('rebase_ns',10**20)]:
            with self.assertRaises(AssertionError):analyze(self.changed('divisor_update',lambda r:r['timings'].__setitem__(key,value)))

    def test_pair_and_resource_quota_are_audited(self):
        for text in [self.changed('divisor_pair',lambda r:r.__setitem__('witness_equal',False)),
                     self.changed('divisor_initial',lambda r:r['resource'].__setitem__('upper_bytes',1)),
                     self.changed('divisor_initial',lambda r:r['budget'].__setitem__('q','0')),
                     self.changed('divisor_update',lambda r:r.__setitem__('fr_elements',0))]:
            with self.assertRaises(AssertionError):analyze(text)

    def test_duplicate_or_unknown_rows_are_rejected(self):
        for extra in [next(x for x in self.lines if '"kind":"divisor_pair"' in x),'ROW {"kind":"unknown"}']:
            with self.assertRaises(AssertionError):analyze('\n'.join(self.lines+[extra]))

    def test_cached_update_requires_actual_lookup_hits(self):
        lines=[]
        for line in self.lines:
            if line.startswith('ROW '):
                row=json.loads(line[4:])
                if row['kind']=='divisor_update' and row['route']==1:row['cache']['divisor_cache_hits']=0
                line='ROW '+json.dumps(row)
            lines.append(line)
        with self.assertRaises(AssertionError):analyze('\n'.join(lines))


if __name__=='__main__':unittest.main()
