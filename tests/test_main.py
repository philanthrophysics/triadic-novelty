import unittest
from triadic_novelty.main import main

class TestMain(unittest.TestCase):
    def test_main(self):
        # Capture the output of the main function
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output
        main()
        sys.stdout = sys.__stdout__
        
        # Check if the output is as expected
        self.assertEqual(captured_output.getvalue().strip(), "Welcome to MVPNovelty!")

if __name__ == '__main__':
    unittest.main()
