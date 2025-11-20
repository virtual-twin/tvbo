#
# Module: test_ontology.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
import unittest
from tvbo.knowledge.ontology import get_model, get_model_acronym


class TestOntology(unittest.TestCase):
    def setUp(self):
        # Setup code goes here. This will run before each test.
        pass

    def tearDown(self):
        # Teardown code goes here. This will run after each test.
        pass

    def test_get_model(self):
        # Here you can write tests for the get_model function.
        # For example:
        model = get_model("JansenRit")
        self.assertIsNotNone(model)

    def test_get_model_valid_input(self):
        # Test if the function returns the correct model for a valid input.
        model = get_model("JansenRit")
        self.assertEqual(model.name, "JansenRit")

    def test_get_model_invalid_input(self):
        # Test if the function raises an exception for an invalid input.
        with self.assertRaises(ValueError):
            get_model("InvalidInput")

    def test_get_model_acronym(self):
        # Test if the function returns the correct acronym for a given model.
        acronym = get_model_acronym("JansenRit")
        self.assertEqual(
            acronym, "JR"
        )  # Replace "JR" with the correct acronym for the "JansenRit" model.


if __name__ == "__main__":
    unittest.main()
