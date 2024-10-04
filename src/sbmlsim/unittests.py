import unittest
import pkg_resources
import pandas as pd
from sbmlsim import Batch  # assuming Batch is the class you want to test


class TestBatch(unittest.TestCase):
    def setUp(self):

        genbank_file = pkg_resources.resource_filename(
            "sbmlsim", "data/NC_045512.2.gbk.gz"
        )
        genbank_file2 = pkg_resources.resource_filename(
            "sbmlsim", "data/NC_000962.3.gbk.gz"
        )

        # Create an instance of Batch to use in tests
        self.batch1 = Batch(
            gene="S",
            drug="PZA",
            genbank_file=genbank_file,
            resistant_mutations=["S@F2L"],
        )

        self.batch2 = Batch(
            gene=["S", "ORF3a"],
            drug=["PZA", "RIF"],
            genbank_file=genbank_file,
            resistant_mutations=["S@V3L", "ORF3a@F4L"],
        )

        self.batch3 = Batch(
            gene="pncA",
            drug="PZA",
            catalogue_file="data/NC_000962.3_WHO-UCN-GTB-PCI-2021.7_v1.0_GARC1_RUS.csv",
            genbank_file=genbank_file2,
            ignore_catalogue_susceptibles=False,
        )

        self.batch4 = Batch(
            gene="pncA",
            drug="PZA",
            resistant_mutations=["pncA@M1V"]["S@F2L"],
            susceptible_mutations=["pncA@M1A", "pncA@R2A"],
            genbank_file=genbank_file2,
        )

    def test_generate_R(self):
        # Test generate method for one resistant sample of the Batch class
        sequence, mutation = self.batch1.generate(
            n_samples=1,
            proportion_resistant=1,
            n_res=1,
            n_sus=0,
            distribution="poisson",
        )
        sequence.drop(columns=["S"], inplace=True)
        expected_sequence = pd.DataFrame(
            {
                "sample_id": [0],
                "phenotype_label": ["R"],
                "number_resistant_mutations": [1],
                "number_susceptible_mutations": [0],
                # "pncA": ["*"],
            }
        )
        expected_sequence.set_index("sample_id", inplace=True)
        expected_mutation = pd.DataFrame(
            {
                "sample_id": [0],
                "mutation": ["F2L"],
                "gene": ["S"],
                "mutation_label": ["R"],
            }
        )
        expected_mutation.set_index(["sample_id", "mutation", "gene"], inplace=True)
        # print(mutation)
        # print(expected_mutation)
        self.assertTrue(sequence.equals(expected_sequence))
        self.assertTrue(mutation.equals(expected_mutation))

    def test_generate_S(self):
        # Test generate method for one susceptible sample of the Batch class
        sequence, mutation = self.batch1.generate(
            n_samples=1,
            proportion_resistant=0,
            n_res=0,
            n_sus=1,
            distribution="poisson",
        )
        sequence.drop(columns=["S", "number_susceptible_mutations"], inplace=True)
        mutation.reset_index(inplace=True)
        mutation.drop(columns=["mutation"], inplace=True)
        mutation.set_index(["sample_id", "gene"], inplace=True)

        # Check that all phenotype labels in the sequence and mutation DataFrame are "S"
        self.assertTrue((sequence["phenotype_label"] == "S").all())
        self.assertTrue((mutation["mutation_label"] == "S").all())

        # Check that the number of resistant mutations in the sequence DataFrame is 0
        self.assertTrue((sequence["number_resistant_mutations"] == 0).all())

    # Add more test methods as needed

    def test_infeasible_input(self):
        # Test generate method for nonsense input
        with self.assertRaises(ValueError):
            sequence, mutation = self.batch1.generate(
                n_samples=-1,
                proportion_resistant=1,
                n_res=1,
                n_sus=0,
                distribution="poisson",
            )

        with self.assertRaises(ValueError):
            sequence, mutation = self.batch1.generate(
                n_samples=1,
                proportion_resistant=-1,
                n_res=1,
                n_sus=0,
                distribution="poisson",
            )

        with self.assertRaises(ValueError):
            sequence, mutation = self.batch1.generate(
                n_samples=1,
                proportion_resistant=1,
                n_res=-1,
                n_sus=0,
                distribution="poisson",
            )

    def test_proportion_resistant(self):
        # Test resistant ratio is applied correctly
        sequence, mutation = self.batch1.generate(
            n_samples=10,
            proportion_resistant=1,
            n_res=1,
            n_sus=0,
            distribution="poisson",
        )

        self.assertTrue(
            sequence["phenotype_label"].value_counts(normalize=True)["R"] == 1
        )
        self.assertTrue(
            mutation["mutation_label"].value_counts(normalize=True)["R"] == 1
        )

        sequence, mutation = self.batch1.generate(
            n_samples=10,
            proportion_resistant=0,
            n_res=0,
            n_sus=1,
            distribution="poisson",
        )

        self.assertTrue(
            sequence["phenotype_label"].value_counts(normalize=True)["S"] == 1
        )
        self.assertTrue(
            mutation["mutation_label"].value_counts(normalize=True)["S"] == 1
        )

    # def test_n_res_input(self):
    #     sequence, mutation = self.batch1.generate(
    #         n_samples=10,
    #         proportion_resistant=1,
    #         n_res=0,
    #         n_sus=0,
    #         distribution="poisson",
    #     )

    def test_multiple_genes_generate(self):
        sequence, mutation = self.batch2.generate(
            n_samples=1,
            proportion_resistant=1,
            n_res=2,
            n_sus=0,
            distribution="poisson",
        )

        assert sequence["phenotype_label"].value_counts(normalize=True)["R"] == 1
        assert mutation["mutation_label"].value_counts(normalize=True)["R"] == 1
        assert len(mutation) > 0
        assert len(sequence) == 1
        assert len(sequence.columns) == 5
        assert len(mutation.columns) == 1
        # assert sequence["number_resistant_mutations"] == 2
        assert sequence.columns.tolist() == [
            "phenotype_label",
            "number_resistant_mutations",
            "number_susceptible_mutations",
            "S",
            "ORF3a",
        ]

    def test_generate_S_from_catalogue(self):
        sequence, mutations = self.batch3.generate(
            n_samples=1,
            proportion_resistant=0,
            n_res=0,
            n_sus=5,
        )

        mutations.reset_index(inplace=True)
        self.assertTrue(
            mutations["mutation"]
            .isin(self.batch3.susceptible_mutations["mutation"])
            .all()
        )

    def test_generate_S_from_list(self):
        sequence, mutations = self.batch4.generate(
            n_samples=1,
            proportion_resistant=1,
            n_res=1,
            n_sus=5,
        )

        mutations.reset_index(inplace=True)
        self.assertTrue(
            mutations[mutations["mutation_label"] == "S"]["mutation"]
            .isin(self.batch4.susceptible_mutations["mutation"])
            .all()
        )
        self.assertTrue(
            mutations[mutations["mutation_label"] == "R"]["mutation"]
            .isin(self.batch4.resistant_mutations["mutation"])
            .all()
        )


if __name__ == "__main__":
    unittest.main()
