import copy
import unittest

from batch_pilot import parse_batch


class BatchParsingTest(unittest.TestCase):
    def response(self, answer):
        return {'modelVersion': 'gemini-4-argon',
                'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': answer}]}}],
                'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 20,
                                  'thoughtsTokenCount': 30}}

    def test_accepts_complete_batch_and_counts_thinking(self):
        answer, predictions, usage = parse_batch(self.response('{"a":"A","b":"E"}'), ['a', 'b'])
        self.assertEqual(predictions, {'a': 'A', 'b': 'E'})
        self.assertEqual(usage['completion_including_reasoning'], 50)

    def test_rejects_missing_extra_duplicate_and_invalid_answers(self):
        for answer in ['{"a":"A"}', '{"a":"A","b":"B","c":"C"}',
                       '{"a":"A","a":"B","b":"C"}', '{"a":"A","b":"wrong"}']:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                parse_batch(self.response(answer), ['a', 'b'])

    def test_rejects_truncation_and_wrong_model(self):
        raw = self.response('{"a":"A","b":"B"}')
        for key in ['model', 'finish']:
            bad = copy.deepcopy(raw)
            if key == 'model':
                bad['modelVersion'] = 'different'
            else:
                bad['candidates'][0]['finishReason'] = 'MAX_TOKENS'
            with self.assertRaises(ValueError):
                parse_batch(bad, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
