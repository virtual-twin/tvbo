#
# Module: test_lemsgenerator.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
import unittest
import tvbo.export.lemsgenerator as lg
from lems.api import Model


class TestLEMSGenerator(unittest.TestCase):
    def test_setup_lems_model(self):
        """Test if the LEMS model is set up with the correct base components."""
        model = lg.setup_lems_model()
        self.assertIsInstance(model, Model)
        # Assert that dimensions and units are added correctly.
        self.assertEqual(len(model.dimensions), 3)
        self.assertEqual(len(model.units), 3)

    def test_create_tvb_model(
        self,
    ):
        """Test the creation of a TVB model."""
        # Call create_tvb_model
        model = lg.create_tvb_model("JansenRit")
        self.assertIsNotNone(model)


# ... You can add more test cases as needed ...

if __name__ == "__main__":
    unittest.main()
