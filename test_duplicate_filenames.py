import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path

# Import the module under test
import importlib.util
spec = importlib.util.spec_from_file_location("organizer", "Downloads-Organizer.py")
organizer_module = importlib.util.module_from_spec(spec)
sys.modules["organizer"] = organizer_module
spec.loader.exec_module(organizer_module)

from organizer import AICategorizer, SmartOrganizer, Config, Database

class TestDuplicateFilenames(unittest.TestCase):
    """
    حالات اختبار لمعالجة أسماء الملفات المكررة ومنطق التصنيف.
    """
    def setUp(self):
        """إعداد بيئة الاختبار باستخدام قاعدة بيانات وهمية وتكوين."""
        self.config = Config()
        self.config.enable_ml_clustering = False
        self.config.batch_size = 10
        self.db = MagicMock(spec=Database)
        self.db.get_category.return_value = None # No cache

        self.categorizer = AICategorizer(self.config, self.db)

        # Files with same name but different paths/content
        self.files = [
            {
                'name': 'document.txt',
                'path': '/path/to/dir1/document.txt',
                'hash': 'hash1',
                'ext': '.txt',
                'size_kb': 1,
                'content_text': 'Invoice content'
            },
            {
                'name': 'document.txt',
                'path': '/path/to/dir2/document.txt',
                'hash': 'hash2',
                'ext': '.txt',
                'size_kb': 1,
                'content_text': 'Homework content'
            }
        ]

    def test_categorize_batch_fallback_structure(self):
        """
        اختبار أن categorize_batch تعيد النتائج مرتبطة بالمسار.

        يتحقق من أنه حتى بدون الذكاء الاصطناعي، يقوم المصنف الاحتياطي بمعالجة الملفات بشكل صحيح
        ويعيد قاموسًا مرتبطًا بمسار الملف.
        """
        # Force fallback by ensuring no client
        self.categorizer.client = None

        results = asyncio.run(self.categorizer.categorize_batch(self.files))

        # Verify we have results for both files (keyed by path)
        self.assertEqual(len(results), 2, "Should return results for both files")
        self.assertIn('/path/to/dir1/document.txt', results)
        self.assertIn('/path/to/dir2/document.txt', results)

    def test_ai_prompt_structure(self):
        """
        اختبار أن _build_prompt يتضمن معرفات فريدة.

        يضمن أنه عند بناء الموجه (Prompt) للذكاء الاصطناعي، يتم تعيين معرف فريد لكل ملف
        لمنع الغموض في استجابة الذكاء الاصطناعي.
        """
        prompt = self.categorizer._build_prompt(self.files)

        # Extract JSON part from prompt
        start_idx = prompt.find("FILES:\n") + 7
        end_idx = prompt.find("\n\nRespond with")
        json_str = prompt[start_idx:end_idx]

        try:
            files_data = json.loads(json_str)
            self.assertEqual(len(files_data), 2)
            # Check if 'id' field exists
            self.assertIn('id', files_data[0])
            self.assertNotEqual(files_data[0]['id'], files_data[1]['id'])
        except json.JSONDecodeError:
            self.fail("Could not parse JSON from prompt")

    @patch('organizer.SmartOrganizer._move_file')
    def test_process_batch_integration(self, mock_move):
        """
        اختبار أن _process_batch يعين النتائج بشكل صحيح للملفات باستخدام المسار.

        يتحقق من أن المنظم يستخدم مسار الملف لتعيين نتائج التصنيف
        مرة أخرى إلى الملف الصحيح، مما يمنع المشاكل عندما تكون أسماء الملفات متطابقة.
        """
        organizer = SmartOrganizer(self.config)
        organizer.categorizer = self.categorizer

        # Mock categorize_batch to return what we expect after fix
        mock_results = {
            '/path/to/dir1/document.txt': {'folder': 'Finance'},
            '/path/to/dir2/document.txt': {'folder': 'School'}
        }

        async def mock_cat_batch(files):
            return mock_results

        organizer.categorizer.categorize_batch = mock_cat_batch
        organizer.db = self.db

        asyncio.run(organizer._process_batch(self.files))

        self.assertEqual(mock_move.call_count, 2)

        # Verify correct mapping
        args1, _ = mock_move.call_args_list[0]
        self.assertEqual(args1[0]['path'], '/path/to/dir1/document.txt')
        self.assertEqual(args1[1]['folder'], 'Finance')

        args2, _ = mock_move.call_args_list[1]
        self.assertEqual(args2[0]['path'], '/path/to/dir2/document.txt')
        self.assertEqual(args2[1]['folder'], 'School')

if __name__ == '__main__':
    unittest.main()
