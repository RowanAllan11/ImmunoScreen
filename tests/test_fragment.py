import unittest
from src.fragment import fragment_protein

class TestFragmentation(unittest.TestCase):

    def test_fragment_protein(self):
        # Test case 1: Basic functionality
        protein_sequence = "ACDEFGHIKLMNPQRSTVWY"
        kmer_length = 3
        expected_fragments = [
            "ACD", "CDE", "DEF", "EFG", "FGH", "GHK", "HIK", "IKL", 
            "KLM", "LMN", "MNP", "NPR", "PQR", "QRS", "RST", "STV", 
            "TVW", "VWY"
        ]
        result = fragment_protein(protein_sequence, kmer_length)
        self.assertEqual(result, expected_fragments)

        # Test case 2: Edge case with kmer length greater than sequence length
        kmer_length = 25
        expected_fragments = []
        result = fragment_protein(protein_sequence, kmer_length)
        self.assertEqual(result, expected_fragments)

        # Test case 3: Edge case with kmer length of 1
        kmer_length = 1
        expected_fragments = list(protein_sequence)
        result = fragment_protein(protein_sequence, kmer_length)
        self.assertEqual(result, expected_fragments)

if __name__ == '__main__':
    unittest.main()