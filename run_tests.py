import unittest

if __name__ == "__main__":
    tests = unittest.defaultTestLoader.discover("test")
    runner = unittest.TextTestRunner()
    runner.run(tests)