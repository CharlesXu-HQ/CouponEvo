import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from couponevo.data import load_dataset, split_dataset
from couponevo.evaluate import Budget
from couponevo.experience import load_experience
from couponevo.harness import build_harness_context, harness_digest, load_harness, validate_research
from couponevo.search import finalize_search, run_search


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / 'data.csv'
        pd.DataFrame({'id': range(100), 'arm': [i % 2 for i in range(100)],
                      'x': [i / 100 for i in range(100)], 'y': [i % 3 == 0 for i in range(100)]}).to_csv(self.csv, index=False)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({
            'dataset': 'data.csv', 'unit_id': 'id', 'features': ['x'],
            'feature_timing': 'pre_treatment', 'outcomes': {'active': 'y'},
            'treatment': {'column': 'arm', 'control': 0, 'treated': 1, 'probability': 0.5,
                          'probability_source': 'protocol', 'probability_reference': 'test'},
        }))
        self.plugin = self.root / 'plugin.json'
        self.spec = load_harness(Path(__file__).resolve().parents[1] / 'harnesses/coupon-research.json')
        self.plugin.write_text(json.dumps(self.spec))
        self.seed = Path(__file__).resolve().parents[1] / 'src/couponevo/candidate.py'
        self.kwargs = dict(manifest_path=self.manifest, budget=Budget('count', 0.2), seed=7,
                           output=self.root / 'runs', initial_candidate=self.seed, objective='active',
                           search_id='plugin-test', harness_path=self.plugin)
        self.research = {
            'direction': 'A new direction outside the catalogue', 'mechanism': 'Train-fitted feature interaction',
            'why_now': 'Current evidence suggests an additive limitation',
            'data_rationale': 'x is available; its business meaning is unknown',
            'input_features': ['x'], 'comparison': 'Compare the same policy with and without the interaction',
            'falsification': 'No validation policy-value gain under the fixed budget',
            'alternatives': [{'direction': 'Training', 'mechanism': 'Stronger regularization',
                              'reason': 'No evidence of high variance yet'}],
        }
        self.proposal = {'action': 'experiment', 'operator': 'draft', 'parent_ids': [],
                         'hypothesis': 'Interactions help', 'expected_result': 'Policy value rises',
                         'candidate_py': self.seed.read_text(), 'research': self.research}
        self.report = {'holdout': 'validation', 'policies': {'active': {'effects': {'active': {'mean': 0.0}}}}}

    def test_profile_statistics_only_use_training_features(self):
        before = build_harness_context(self.plugin, self.manifest, 7, False)
        split = split_dataset(load_dataset(self.manifest), 7)
        frame = pd.read_csv(self.csv)
        heldout = [*split.validation.index, *split.test.index]
        frame.loc[heldout, 'x'] = float('nan')
        frame.loc[heldout, 'y'] = False
        frame.to_csv(self.csv, index=False)
        after = build_harness_context(self.plugin, self.manifest, 7, False)
        self.assertEqual(before, after)
        self.assertEqual(before['dataset_profile']['training_rows'], 60)
        self.assertEqual(set(before['dataset_profile']['features']), {'x'})
        self.assertNotIn('test', before['dataset_profile'])

    def test_custom_plugin_and_direction_are_not_a_closed_enum(self):
        self.spec['directions'] = [dict(self.spec['directions'][0], name='Custom mechanism')]
        self.plugin.write_text(json.dumps(self.spec))
        context = build_harness_context(self.plugin, self.manifest, 7, False)
        self.assertEqual(context['plugin']['directions'][0]['name'], 'Custom mechanism')
        self.assertEqual(validate_research(self.proposal, context), self.research)

    def test_missing_inputs_or_unfalsifiable_design_are_rejected(self):
        context = build_harness_context(self.plugin, self.manifest, 7, False)
        for changes in ({'input_features': ['missing_history']}, {'falsification': ''}, {'alternatives': []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_research({**self.proposal, 'research': {**self.research, **changes}}, context)

    def test_policy_only_design_can_use_no_raw_features(self):
        context = build_harness_context(self.plugin, self.manifest, 7, False)
        research = {**self.research, 'input_features': [],
                    'data_rationale': 'Only existing predicted effects and costs are used by the policy'}
        self.assertEqual(validate_research({**self.proposal, 'research': research}, context), research)

    def test_plugin_schema_is_validated_and_hash_is_content_based(self):
        digest = harness_digest(self.plugin)
        self.plugin.write_text(json.dumps(self.spec, indent=4))
        self.assertEqual(digest, harness_digest(self.plugin))
        self.plugin.write_text(json.dumps({**self.spec, 'schema_version': 2}))
        with self.assertRaisesRegex(ValueError, 'schema_version'):
            load_harness(self.plugin)

    def test_search_retries_design_then_preserves_it_through_reflection_and_experience(self):
        contexts, observations = [], []

        def propose(context):
            contexts.append(copy.deepcopy(context))
            if len(contexts) == 1:
                return {**self.proposal, 'research': {**self.research, 'input_features': ['absent']}}
            if context.get('experiment_budget_exhausted'):
                return {'action': 'stop', 'reason': 'Budget exhausted; other mechanisms remain untested'}
            self.assertIn('proposal_error', context)
            return self.proposal

        def reflect(observation):
            observations.append(observation)
            return {'verdict': 'inconclusive', 'evidence': 'No clear gain',
                    'lesson': 'This configuration lacks support', 'next_direction': 'Test regularization'}

        with patch('couponevo.search._evaluate', return_value=self.report) as evaluate:
            journal = run_search(**self.kwargs, max_steps=1, proposer=propose, reflector=reflect)
            self.assertEqual(evaluate.call_count, 2)
            final = {**self.report, 'holdout': 'test', 'secret_test_metric': 999}
            evaluate.return_value = final
            finalize_search(**self.kwargs, bootstrap_reps=10)
        self.assertEqual(journal['steps'][0]['research'], self.research)
        self.assertEqual(observations[0]['research'], self.research)
        self.assertEqual(contexts[-1]['research_history'][0]['direction'], self.research['direction'])
        lessons = load_experience(self.root / 'runs', journal['task'])
        self.assertEqual(lessons[0]['research'], self.research)
        self.assertNotIn('secret_test_metric', json.dumps(lessons))
        self.assertEqual(load_experience(self.root / 'runs', {**journal['task'], 'harness': 'changed'}), [])

    def test_cli_flag_attaches_plugin_to_the_real_search_loop(self):
        from couponevo.cli import main

        args = ["couponevo", "search", str(self.manifest), "--budget-kind", "count",
                "--budget", "0.2", "--objective", "active", "--search-id", "cli-plugin",
                "--seed", "7", "--max-steps", "1", "--output", str(self.root / "runs"),
                "--harness", str(self.plugin), "--agent-provider", "deepseek", "--unsafe-local-execution"]
        reflection = {"verdict": "inconclusive", "evidence": "No clear gain",
                      "lesson": "Keep testing", "next_direction": "Another mechanism"}
        with patch("sys.argv", args), patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.search._evaluate", return_value={**self.report, "run_id": "unit"}), \
                patch("couponevo.cli.propose_search_candidate", side_effect=[
                    self.proposal, {"action": "stop", "reason": "Experiment budget exhausted"}]) as propose, \
                patch("couponevo.cli.reflect_search_step", return_value=reflection), \
                patch("couponevo.cli.analyze_reports_deepseek", return_value={"high": {}}):
            main()
        context = propose.call_args_list[0].args[1]
        self.assertEqual(context["harness"]["plugin"]["name"], self.spec["name"])
        journal = json.loads((self.root / "runs/cli-plugin/journal.json").read_text())
        self.assertEqual(journal["steps"][0]["research"], self.research)
        self.assertNotIn("test-key", json.dumps(journal))

    def test_changed_plugin_blocks_resume_and_finalize(self):
        with patch('couponevo.search._evaluate', return_value=self.report):
            run_search(**self.kwargs, max_steps=1, proposer=lambda _: self.proposal)
        self.spec['guidance'] += ' A changed research protocol.'
        self.plugin.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_search(**self.kwargs, max_steps=2, proposer=lambda _: self.proposal, resume=True)
        with self.assertRaisesRegex(ValueError, 'changed'):
            finalize_search(**self.kwargs, bootstrap_reps=10)

    def test_missing_data_action_does_not_require_an_experiment_design(self):
        request = {'action': 'request_data', 'reason': 'History is unavailable',
                   'feature_request': {'name': 'history', 'definition': 'Events before assignment',
                                       'source': 'Event log', 'as_of': 'Before assignment',
                                       'evidence': 'Only x exists', 'validation_plan': 'Check cutoff'}}
        with patch('couponevo.search._evaluate', return_value=self.report) as evaluate:
            journal = run_search(**self.kwargs, max_steps=1, proposer=lambda _: request)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(journal['data_request']['feature_request'], request['feature_request'])


if __name__ == '__main__':
    unittest.main()
