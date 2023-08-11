import unittest

import ml_service

class TestMLService(unittest.TestCase):

    def test_predict(self):

        input_data = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1]

        pred_name, pred_probability = ml_service.predict(input_data)
        self.assertEqual(pred_name, "This person needs to be hospitalized next year.")
        self.assertAlmostEqual(float(pred_probability), 96.92, 1)

if __name__ == "__main__":
    unittest.main()
