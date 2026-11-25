import copy
import unittest

from prepare import remove_media


class FrozenPromptTest(unittest.TestCase):
    def test_removes_only_media_and_preserves_all_text(self):
        body = {
            'systemInstruction': {'parts': [{'text': 'original system\n'}]},
            'contents': [{'role': 'user', 'parts': [
                {'text': 'Video head_cam; this clip starts at 0 seconds.'},
                {'inlineData': {'mimeType': 'video/mp4', 'data': 'fixture'}},
                {'text': 'original user\n'}]}],
            'generationConfig': {'thinkingConfig': {'thinkingLevel': 'MEDIUM'}}}
        original = copy.deepcopy(body)
        expected = copy.deepcopy(body)
        del expected['contents'][0]['parts'][1]
        self.assertEqual(remove_media(body), expected)
        self.assertEqual(body, original)

    def test_refuses_missing_or_ambiguous_media(self):
        for parts in [[{'text': 'hello'}], [{'inlineData': {}}, {'fileData': {}}]]:
            with self.assertRaises(ValueError):
                remove_media({'contents': [{'parts': parts}]})


if __name__ == '__main__':
    unittest.main()
