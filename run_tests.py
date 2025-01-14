import unittest

if __name__ == "__main__":
    tests = unittest.defaultTestLoader.discover("tests")
    runner = unittest.TextTestRunner()
    runner.run(tests)