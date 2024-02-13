import numpy
import copy
import random
import pandas as pd

import gumpy
import piezo


class Batch:
    """
    Class to instantiate a batch of samples from a given gene and drug

    Args:
        gene: str, name of gene
        drug: str, name of drug
        catalogue_file: str, path to catalogue file
        genbank_file: str, path to genbank file

    Example:
        sbmlsim.batch(gene="gyrA", drug="ciprofloxacin", catalogue_file="catalogue.csv", genbank_file="NC_000962.3.gbk")
    """

    def __init__(
        self,
        gene=None,
        drug=None,
        catalogue_file=None,
        genbank_file=None,
        resistant_mutations=None,
    ):
        # TODO: allow a list of genes to be specified for e.g. gyrA and gyrB
        self.gene = gene
        self.drug = drug
        self.genbank_file = genbank_file
        self.catalogue_file = catalogue_file

        # allow a user to either provide a catalogue file or a list of resistant mutations
        assert (catalogue_file is not None) != (
            resistant_mutations is not None
        ), "Either catalogue_file or resistant_mutations must be specified"

        if catalogue_file is not None:
            self.catalogue = piezo.ResistanceCatalogue(catalogue_file)
            resistant_mutations = self._get_mutations()  # make this an argument
        else:
            assert isinstance(resistant_mutations, list)
            resistant_mutations = resistant_mutations

        self.resistant_mutations = pd.DataFrame(
            resistant_mutations, columns=["mutation"]
        )

        # extract the position of each mutation
        def find_positions(row):
            return int(row.mutation[1:-1])

        self.resistant_mutations["position"] = self.resistant_mutations.apply(
            find_positions, axis=1
        )

        # since each position can have >1 resistance-conferring mutation, we need to
        # identify the unique positions that are resistant to choose from later
        self.resistant_positions = list(self.resistant_mutations.position.unique())

        # call private methods to complete instantiation
        self._define_lookups()

        # retain the Genome object as it does take a while to instantiate
        self.reference_genome = gumpy.Genome(genbank_file)

        # build the required Gene object
        self.reference_gene = self.reference_genome.build_gene(self.gene)

    def generate(
        self, n_samples, proportion_resistant, n_res=1, n_sus=1, distribution="poisson"
    ):
        """
        Generate a batch of samples.

        Args:
            n_samples: int, number of samples to generate
            proportion_resistant: float, proportion of samples that are resistant
            n_res: int, mean number of resistant mutations per resistant sample
            n_sus: int, mean number of susceptible mutations per susceptible sample
            distribution: str, distribution to use for number of resistant mutations per resistant sample (currently only poission is implemented)

        Returns:
            samples: pandas DataFrame with one row per sample
            mutations: pandas DataFrame with one row per mutation per drug per sample
        """

        # TODO: make this way tidier

        samples_rows = []
        mutations_rows = []

        # iterate through the required number of samples
        for n_sample in range(n_samples):
            # make a copy of the reference gene
            sample_gene = copy.deepcopy(self.reference_gene)

            # decide the phenotype of the sample
            label = self._get_sample_type(proportion_resistant)

            # determine the resistant mutations, if any
            if label == "R":
                (
                    number_resistant,
                    selected_resistant_mutations,
                    remaining_aa_positions,
                ) = self._get_resistant_mutations(n_res, distribution, sample_gene)

            else:
                number_resistant = 0
                selected_resistant_mutations = []
                remaining_aa_positions = sample_gene.amino_acid_number

            # determine the susceptible mutations, if any
            if n_sus > 0:
                (
                    number_susceptible,
                    selected_susceptible_mutations,
                ) = self._get_susceptible_mutations(
                    n_sus, remaining_aa_positions, sample_gene
                )
            else:
                selected_susceptible_mutations = []
                number_susceptible = 0

            # combine the resistant and susceptible mutations
            selected_mutations = (
                selected_resistant_mutations + selected_susceptible_mutations
            )

            # apply the mutations to the sample gene
            self._apply_mutations(selected_mutations, sample_gene)

            # translate the nucleotide sequence to amino acids
            sample_gene._translate_sequence()

            # create a string of the amino acid sequence
            sample_amino_acid_sequence = "".join(
                i for i in sample_gene.amino_acid_sequence
            )

            # construct the row for the sample table
            sample_metadata = [
                n_sample,
                sample_amino_acid_sequence,
                label,
                number_resistant,
                number_susceptible,
            ]

            samples_rows.append(sample_metadata)

            # mutations dataframe
            diff = self.reference_gene - sample_gene
            for i in diff.mutations:
                if i in selected_resistant_mutations:
                    mut_label = "R"
                else:
                    mut_label = "S"
                mutations_rows.append([n_sample, i, mut_label, self.gene])

        samples = pd.DataFrame(
            samples_rows,
            columns=[
                "sample_id",
                "allele",
                "phenotype_label",
                "number_resistant_mutations",
                "number_susceptible_mutations",
            ],
        )
        samples.set_index("sample_id", inplace=True)

        mutations = pd.DataFrame(
            mutations_rows, columns=["sample_id", "mutation", "mutation_label", "gene"]
        )
        mutations.set_index(["sample_id", "mutation"], inplace=True)

        return samples, mutations

    def __repr__(self) -> str:
        # print a summary of the Batch object

        line = ""
        line += f"Gene: {self.gene}\n"
        line += f"Drug: {self.drug}\n"
        line += f"GenBank file: {self.genbank_file}\n"
        if hasattr(self, "catalogue"):
            line += f"Catalogue: {self.catalogue_file}\n"
        if hasattr(self, "resistant_mutations"):
            line += f"Number of resistant mutations: {len(self.resistant_mutations)}\n"
        return line

    def _define_lookups(self):
        # build the various dicts to allow us to convert from amino acid to codon and vice versa

        aminoacids = "FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
        bases = ["t", "c", "a", "g"]
        all_codons = numpy.array(
            [a + b + c for a in bases for b in bases for c in bases]
        )
        self.codon_to_amino_acid = dict(zip(all_codons, aminoacids))
        self.amino_acid_to_codon = {}
        for amino_acid, codon in zip(aminoacids, all_codons):
            self.amino_acid_to_codon.setdefault(amino_acid, []).append(codon)

    def _get_mutations(self):
        # subset down to mutations that are missense (not premature stop) SNPs in the CDS of the gene of interest

        specific_resistance_mutations = self.catalogue.catalogue.rules[
            (self.catalogue.catalogue.rules.PREDICTION == "R")
            & (self.catalogue.catalogue.rules.MUTATION_TYPE == "SNP")
            & (self.catalogue.catalogue.rules.MUTATION_AFFECTS == "CDS")
            & (self.catalogue.catalogue.rules.MUTATION.str[-1] != "!")
            & (self.catalogue.catalogue.rules.POSITION != "*")
        ]

        # subset down to only those mutations that are in the gene of interest for the specified drug
        gene_resistant_mutations = specific_resistance_mutations[
            (specific_resistance_mutations.DRUG == self.drug)
            & (specific_resistance_mutations.GENE == self.gene)
        ]

        return list(gene_resistant_mutations.MUTATION)

    def _get_sample_type(self, proportion_resistant):
        # randomly assign a label to the sample

        if random.random() < proportion_resistant:
            label = "R"
        else:
            label = "S"

        return label

    def _get_resistant_mutations(self, n_res, distribution, sample_gene):
        # choose resistance-conferring mutations for a sample

        if distribution == "poisson":

            while True:
                number_resistant = numpy.random.poisson(n_res)
                if number_resistant > 0 and number_resistant <= len(
                    self.resistant_positions
                ):
                    break

        else:
            # TODO: Implement other distribution functions
            raise NotImplementedError(
                'Only "poisson" distribution is currently implemented'
            )

        assert (
            number_resistant >= 0
        ), "a resistant sample must have at least one resistance-conferring mutation"

        # randomly choose some positions to mutate
        selected_resistant_positions = random.sample(
            self.resistant_positions, k=number_resistant
        )

        # for each position, randomly choose a mutation to allow for a position having more than one mutation
        selected_resistant_mutations = []
        for i in selected_resistant_positions:
            df = self.resistant_mutations[self.resistant_mutations.position == i]
            value = df.sample(n=1).mutation.values[0]
            selected_resistant_mutations.append(value)

        assert len(selected_resistant_mutations) == len(
            set(selected_resistant_mutations)
        ), "cannot have duplicate mutations"

        # insist we cannot mutate to Stop codons
        for mutation in selected_resistant_mutations:
            assert mutation[-1] != "!", "cannot mutate to STOP codon"

        # Get amino acid positions that are not altered by selected resistant mutations
        remaining_aa_positions = sample_gene.amino_acid_number[
            ~numpy.isin(sample_gene.amino_acid_number, selected_resistant_positions)
        ]

        return number_resistant, selected_resistant_mutations, remaining_aa_positions

    def _get_susceptible_mutations(self, n_sus, remaining_aa_positions, sample_gene):
        # choose susceptible mutations for a sample

        # randomly choose a number of susceptible mutations
        number_susceptible = numpy.random.poisson(n_sus)

        selected_susceptible_mutations = []

        # now randomly choose from positions which have no resistant-conferring mutation
        for susceptible_position in random.sample(
            list(remaining_aa_positions), k=number_susceptible
        ):
            # find out the wildtype codon
            ref_codon = sample_gene.codons[
                sample_gene.amino_acid_number == susceptible_position
            ][0]

            # find out the wildtype amino acid
            ref_aa = self.codon_to_amino_acid.get(ref_codon)

            # exclude synonymous mutations and any mutation involving a STOP codon
            possible_alt_codons = [
                alt_codon
                for alt_codon, alt_aa in self.codon_to_amino_acid.items()
                if alt_aa != ref_aa and alt_aa != "!" and ref_aa != "!"
            ]

            # exclude resistant conferring mutations
            possible_alt_mutations = [
                f"{ref_aa}{susceptible_position}{self.codon_to_amino_acid.get(alt_codon)}"
                for alt_codon in possible_alt_codons
                if sum(1 for a, b in zip(ref_codon, alt_codon) if a != b) == 1
                and self.codon_to_amino_acid.get(alt_codon)
                not in self.resistant_mutations
            ]

            # select one of these possible
            if possible_alt_mutations:
                selected_susceptible_mutations.append(
                    random.sample(possible_alt_mutations, k=1)[0]
                )

        return number_susceptible, selected_susceptible_mutations

    def _apply_mutations(self, selected_mutations, sample_gene):
        # apply mutations to sample gene

        for mutation in selected_mutations:
            ref_aa = mutation[0]
            alt_aa = mutation[-1]
            aa_pos = int(mutation[1:-1])

            assert (
                ref_aa
                == sample_gene.amino_acid_sequence[
                    sample_gene.amino_acid_number == aa_pos
                ][0]
            ), (
                "reference amino acid supplied in mutation does not match gene! :"
                + mutation
            )

            assert alt_aa != "!", "cannot mutate to STOP codon"

            ref_codon = sample_gene.codons[sample_gene.amino_acid_number == aa_pos][0]

            # Get base changes
            alt_codon = None
            for codon in self.amino_acid_to_codon[alt_aa]:
                alt_codon = codon
                counter = sum(1 for a, b in zip(ref_codon, codon) if a != b)
                if counter == 1:
                    alt_codon = codon
                    break

            base_pos = 3 * aa_pos - 2
            for i, j in zip(ref_codon, alt_codon):
                if i != j:
                    ref_base = i
                    alt_base = j
                    break
                base_pos += 1

            assert (
                self.reference_gene.nucleotide_sequence[
                    self.reference_gene.nucleotide_number == base_pos
                ][0]
                == ref_base
            )

            sample_gene.nucleotide_sequence[
                sample_gene.nucleotide_number == base_pos
            ] = alt_base
