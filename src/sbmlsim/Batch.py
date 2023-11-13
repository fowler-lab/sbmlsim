import numpy
import copy
import random
import pandas as pd

import gumpy
import piezo

# * maybe move some functions to separate file
# import sbmlsim_functions as sf


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
        self.gene = gene
        self.drug = drug
        assert (catalogue_file is not None) != (
            resistant_mutations is not None
        ), "Either catalogue_file or resistant_mutations must be specified"

        if catalogue_file is not None:
            self.catalogue = piezo.ResistanceCatalogue(catalogue_file)
            self.resistant_mutations = self._get_mutations()  # make this an argument
        else:
            assert isinstance(resistant_mutations, list)
            self.resistant_mutations = resistant_mutations

        # call private methods to complete instantiation
        self._define_lookups()
        self.reference_gene = self._build_ref_gene(genbank_file)

    def __repr__(self) -> str:
        line = ""
        line += f"gene: {self.gene}\n"
        return line

    def _define_lookups(self):
        aminoacids = "FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
        bases = ["t", "c", "a", "g"]
        all_codons = numpy.array(
            [a + b + c for a in bases for b in bases for c in bases]
        )
        self.codon_to_amino_acid = dict(zip(all_codons, aminoacids))
        self.amino_acid_to_codon = {}
        for amino_acid, codon in zip(aminoacids, all_codons):
            self.amino_acid_to_codon.setdefault(amino_acid, []).append(codon)

    def _build_ref_gene(self, genbank_file):
        # instantiate a gumpy Genome object
        reference = gumpy.Genome(genbank_file)

        # return the gene of interest
        return reference.build_gene(self.gene)

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
        if random.random() < proportion_resistant:
            label = "R"
        else:
            label = "S"

        return label

    def _get_resistant_mutations(self, n_res, distribution, sample_gene):
        if distribution == "poisson":
            number_resistant = 0
            while number_resistant == 0:
                number_resistant = numpy.random.poisson(n_res)
        else:
            # TODO: Implement other distribution functions
            pass

        selected_resistant_mutations = random.choices(
            self.resistant_mutations, k=number_resistant
        )

        # first, identify the codons being mutated as we will want to avoid these for susceptible mutations
        positions_altered = []
        for mutation in selected_resistant_mutations:
            aa_pos = int(mutation[1:-1])
            positions_altered.append(aa_pos)

        # Get amino acid positions that are not altered by selected resistant mutations
        remaining_aa_positions = sample_gene.amino_acid_number[
            ~numpy.isin(sample_gene.amino_acid_number, positions_altered)
        ]

        return number_resistant, selected_resistant_mutations, remaining_aa_positions

    def _get_susceptible_mutations(self, n_sus, remaining_aa_positions, sample_gene):
        # Work out susceptible mutations
        number_susceptible = numpy.random.poisson(n_sus)

        selected_susceptible_mutations = []

        # now randomly choose some susceptible mutations (i.e. "uniformly")
        for susceptible_codon in random.choices(
            remaining_aa_positions, k=number_susceptible
        ):
            ref_codon = sample_gene.codons[
                sample_gene.amino_acid_number == susceptible_codon
            ][0]
            ref_aa = self.codon_to_amino_acid.get(ref_codon)

            possible_alt_codons = [
                alt_codon
                for alt_codon, alt_aa in self.codon_to_amino_acid.items()
                if alt_aa != ref_aa and alt_aa != "!"  # and ref_aa != "!"
            ]

            possible_alt_mutations = [
                f"{ref_aa}{susceptible_codon}{self.codon_to_amino_acid.get(alt_codon)}"
                for alt_codon in possible_alt_codons
                if sum(1 for a, b in zip(ref_codon, alt_codon) if a != b) == 1
                and self.codon_to_amino_acid.get(alt_codon)
                not in self.resistant_mutations
            ]

            if possible_alt_mutations:
                selected_susceptible_mutations.append(
                    random.choice(possible_alt_mutations)
                )

        return number_susceptible, selected_susceptible_mutations

    def _apply_mutations(self, selected_mutations, sample_gene):
        for mutation in selected_mutations:
            ref_aa = mutation[0]
            alt_aa = mutation[-1]
            aa_pos = int(mutation[1:-1])

            ref_codon = sample_gene.codons[sample_gene.amino_acid_number == aa_pos][0]

            # Get base changes
            alt_codon = None
            for codon in self.amino_acid_to_codon[alt_aa]:
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

    def generate_batch(
        self, n_samples, proportion_resistant, n_res=1, n_sus=1, distribution="poisson"
    ):
        """
        Generate a batch of samples given the supplied parameters.

        Args:
            n_samples: int, number of samples to generate
            proportion_resistant: float, proportion of samples that are resistant
            n_res: int, mean number of resistant mutations per resistant sample
            n_sus: int, mean number of susceptible mutations per susceptible sample
            distribution: str, distribution to use for number of resistant mutations per resistant sample (currently only poission is implemented)

        Returns:
            allele_df: pandas DataFrame with one row per sample
            mutations_df: pandas DataFrame with one row per mutation per drug per sample
        """

        # TODO: STOP codons in susceptible mutations ??
        # TODO: make this way tidier
        # BUG: if n_res is small, number_resistant can be 0, but then the sample is still labeled "R" even though it is not resistant

        allele_df = pd.DataFrame(
            columns=[
                "sample",
                "allele",
                "label",
                "num_res_mutations",
                "num_sus_mutations",
            ]
        ).set_index("sample")

        mutations_df = pd.DataFrame(
            columns=["sample", "mutation", "mutation_label", "gene"]
        ).set_index(["sample", "mutation"])

        for n_sample in range(n_samples):
            sample_gene = copy.deepcopy(self.reference_gene)

            # Work out resistant mutations
            label = self._get_sample_type(proportion_resistant)

            if label == "R":
                (
                    number_resistant,
                    selected_resistant_mutations,
                    remaining_aa_positions,
                ) = self._get_resistant_mutations(n_res, distribution, sample_gene)

            else:
                number_resistant = 0
                selected_resistant_mutations = []
                positions_altered = []
                remaining_aa_positions = sample_gene.amino_acid_number[
                    ~numpy.isin(sample_gene.amino_acid_number, positions_altered)
                ]

            (
                number_susceptible,
                selected_susceptible_mutations,
            ) = self._get_susceptible_mutations(
                n_sus, remaining_aa_positions, sample_gene
            )

            selected_mutations = (
                selected_resistant_mutations + selected_susceptible_mutations
            )

            # Get mutations for sample gene
            self._apply_mutations(selected_mutations, sample_gene)

            # Generate either mutations or mutated allele
            sample_gene._translate_sequence()
            # print("SAMPLE %i, LABEL %s, %i resistant mutations, %i susceptible mutations" % (n_sample, label, number_resistant, number_susceptible))
            # if output == "allele":
            #     sample_amino_acid_sequence = "".join(
            #         i for i in sample_gene.amino_acid_sequence
            #     )
            #     output_alleles.append(sample_amino_acid_sequence)

            sample_amino_acid_sequence = "".join(
                i for i in sample_gene.amino_acid_sequence
            )

            # allele dataframe
            allele_df.loc[n_sample] = [
                sample_amino_acid_sequence,
                label,
                number_resistant,
                number_susceptible,
            ]

            # mutations dataframe
            diff = self.reference_gene - sample_gene
            for i in diff.mutations:
                if i in selected_resistant_mutations:
                    mut_label = "R"
                else:
                    mut_label = "S"

                mutations_df.loc[(n_sample, i), ["mutation_label", "gene"]] = [
                    mut_label,
                    self.gene,
                ]

        return allele_df, mutations_df
