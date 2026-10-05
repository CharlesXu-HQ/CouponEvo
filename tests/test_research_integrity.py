"""Regressions from the real GPU search; no training or API calls required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from couponevo.evaluate import Budget
from couponevo.search import run_search


class ResearchIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seed = self.root / 'candidate.py'
        self.seed.write_text('def fit_predict(*args, **kwargs):\n    return None\n')
        self.task = {'dataset': 'frozen-data', 'manifest': 'frozen-manifest',
                     'objective': 'conversion', 'budget': {'kind': 'count', 'value': 0.2}}
        self.kwargs = dict(manifest_path=self.root / 'manifest.json', budget=Budget('count', 0.2),
                           seed=42, output=self.root / 'runs', initial_candidate=self.seed,
                           objective='conversion', max_steps=1, search_id='audit')
        self.proposal = {'operator': 'improve', 'parent_ids': ['seed'],
                         'hypothesis': 'A testable revision',
                         'candidate_py': self.seed.read_text() + '\n# revision\n'}
        self.reflection = {'verdict': 'inconclusive', 'evidence': 'No clear gain',
                           'lesson': 'Still exploratory', 'next_direction': 'Investigate'}

    def report(self, score):
        return {'run_id': f'run-{score}', 'holdout': 'validation',
                'policies': {'conversion': {'effects': {'conversion': {'mean': score}}}}}

    def run_with_reports(self, proposer, *, analyzer=None, reports=None):
        with patch('couponevo.search._task', return_value=self.task), \
                patch('couponevo.search._evaluate', side_effect=reports or
                      [self.report(0.01), self.report(0.02)]) as evaluate:
            result = run_search(**self.kwargs, proposer=proposer, analyzer=analyzer,
                                reflector=lambda obs: {**self.reflection, "verdict": "invalid"}
                                if obs.get("eligibility") == "blocked_implementation" else self.reflection)
        self.assertEqual(evaluate.call_count, 2)
        return result

    def test_invalid_terminal_requests_preserve_results_and_close_budget(self):
        def propose(context):
            if not context.get('experiment_budget_exhausted'):
                return self.proposal
            raise ValueError('feature_request.basis must be experimental_evidence or domain_requirement')

        try:
            result = self.run_with_reports(propose)
        except ValueError as error:
            self.fail(f'Terminal validation must preserve the completed search: {error}')
        self.assertEqual(result['steps'][0]['status'], 'evaluated')
        self.assertEqual(result['best_id'], 'step-001')
        self.assertEqual(result['budget_exhausted_decision']['reason'], 'invalid_terminal_proposal')
        self.assertEqual(len(result['proposal_errors']), 2)
        self.assertNotIn('data_request', result)
        saved = json.loads((self.root / 'runs/audit/journal.json').read_text())
        self.assertEqual(saved, result)

    def test_analyzer_receives_hypothesis_parent_source_and_code_diff(self):
        seen = []

        def analyze(before, after, candidate, output):
            seen.append(after)
            return {'high': {}}

        result = self.run_with_reports(
            lambda c: {'action': 'stop', 'reason': 'budget exhausted'}
            if c.get('experiment_budget_exhausted') else self.proposal, analyzer=analyze)
        self.assertIn('experiment_context', seen[0])
        context = seen[0]['experiment_context']
        self.assertEqual(context['hypothesis'], self.proposal['hypothesis'])
        self.assertEqual(context['parent_candidate_py'], self.seed.read_text())
        self.assertIn('+# revision', context['candidate_diff'])
        self.assertNotIn('experiment_context', result['steps'][0]['report'])

    def test_confirmed_implementation_error_cannot_become_champion(self):
        result = self.run_with_reports(
            lambda c: {'action': 'stop', 'reason': 'budget exhausted'}
            if c.get('experiment_budget_exhausted') else self.proposal,
            analyzer=lambda *_: {'high': {'implementation_check': {
                'status': 'contradicted', 'evidence': 'Returns a logit contrast, not the declared probability contrast'}}})
        self.assertEqual(result['best_id'], 'seed')
        self.assertEqual(result['steps'][0]['eligibility'], 'blocked_implementation')

    def test_runtime_prediction_mismatch_blocks_even_without_llm_flag(self):
        revised = self.report(0.02)
        revised['runtime_diagnostics'] = {'prediction_contract': {
            'status': 'inconsistent', 'evidence': 'uplift differs from mu1 - mu0'}}
        result = self.run_with_reports(
            lambda c: {'action': 'stop', 'reason': 'budget exhausted'}
            if c.get('experiment_budget_exhausted') else self.proposal,
            reports=[self.report(0.01), revised])
        self.assertEqual(result['best_id'], 'seed')
        self.assertEqual(result['steps'][0]['eligibility'], 'blocked_implementation')

    def test_declared_probability_contrast_requires_observable_arm_predictions(self):
        self.kwargs['harness_path'] = Path('custom-harness')
        with patch('couponevo.search.build_harness_context', return_value={'plugin': {}}), \
                patch('couponevo.search.validate_research', return_value={
                    'direction': 'outcome estimation', 'mechanism': 'two outcome models',
                    'prediction_semantics': 'probability_difference'}):
            result = self.run_with_reports(
                lambda c: {'action': 'stop', 'reason': 'budget exhausted'}
                if c.get('experiment_budget_exhausted') else self.proposal)
        self.assertEqual(result['best_id'], 'seed')
        self.assertEqual(result['steps'][0]['eligibility'], 'blocked_implementation')

    def test_direct_cate_does_not_require_per_arm_probability_columns(self):
        self.kwargs['harness_path'] = Path('custom-harness')
        with patch('couponevo.search.build_harness_context', return_value={'plugin': {}}), \
                patch('couponevo.search.validate_research', return_value={
                    'direction': 'effect estimation', 'mechanism': 'pseudo-outcome regression',
                    'prediction_semantics': 'direct_cate'}):
            result = self.run_with_reports(
                lambda c: {'action': 'stop', 'reason': 'budget exhausted'}
                if c.get('experiment_budget_exhausted') else self.proposal)
        self.assertEqual(result['best_id'], 'step-001')
        self.assertEqual(result['steps'][0]['eligibility'], 'eligible')

    def test_runtime_evidence_is_observed_but_inherited_timing_is_declared(self):
        from couponevo import research_evidence
        journal = {'harness': {'dataset_profile': {
            'statistics_partition': 'train', 'training_rows': 50,
            'features': {'V2': {'dtype': 'float64', 'unique_count': 50}},
            'feature_timing': 'declared_only', 'cost': 'assumed_fixed_send'}},
            'baseline': {'id': 'seed', 'status': 'evaluated', 'score': .01,
                         'report': {'runtime_diagnostics': {'preprocessing': {'raw_feature_count': 7}}}},
            'steps': []}
        facts = {item['id']: item for item in research_evidence.search_evidence(journal)}
        self.assertEqual(facts['seed.runtime.preprocessing']['status'], 'observed')
        self.assertEqual(facts['task.feature_timing']['status'], 'declared')
        self.assertFalse(any(fact.get('data_gap_candidate') for fact in facts.values()))

    def test_one_observed_change_does_not_prove_isolation(self):
        from couponevo import research_evidence
        old = {'runtime_diagnostics': {'preprocessing': {'get_dummies': [{'output_width': 7}]},
                                       'events': {'losses': ['MSELoss']}}}
        new = {'runtime_diagnostics': {'preprocessing': {'get_dummies': [{'output_width': 18}]},
                                       'events': {'losses': ['MSELoss']}}}
        self.assertIsNone(research_evidence.observed_change_audit(old, new))
        new['runtime_diagnostics']['events']['losses'] = ['BCEWithLogitsLoss']
        self.assertEqual(research_evidence.observed_change_audit(old, new)['changed_factors'],
                         ['preprocessing', 'loss'])


if __name__ == '__main__':
    unittest.main()
