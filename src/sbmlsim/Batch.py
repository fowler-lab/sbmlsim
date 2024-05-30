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
        gene: str, name of gene or list of str, multiple gene names
        drug: str, name of drug
        catalogue_file: str, path to catalogue file
        genbank_file: str, path to genbank file
        resistant_mutations: list of str, list of mutations in the format 'gene@mutation'

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

        # add a check to see if gene is a string or a list of strings and store information as a boolean in self.multiple_genes
        if isinstance(gene, str):
            self.multiple_genes = False
        elif isinstance(gene, list):
            self.multiple_genes = True
        else:
            raise ValueError("gene must be a string or a list of strings")

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
            # check that the strings have the format "gene@mutation"
            assert all(
                "@" in x for x in resistant_mutations
            ), "resistant_mutations must be in the format 'gene@mutation'"

            resistant_mutations = resistant_mutations

        self.resistant_mutations = pd.DataFrame(
            [x.split("@") for x in resistant_mutations], columns=["gene", "mutation"]
        )

        # extract the position of each mutation
        def find_positions(row):
            return row.gene + "@" + row.mutation[1:-1]

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
        if self.multiple_genes:
            self.reference_gene = []
            for gene in self.gene:
                self.reference_gene.append(self.reference_genome.build_gene(gene))
        else:
            self.reference_gene = self.reference_genome.build_gene(self.gene)
            # turn into a list to make downstream data handling more uniform
            self.reference_gene = [self.reference_gene]

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

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")

        if not isinstance(n_res, int) or n_res < 0:
            raise ValueError("n_res must be a non-negative integer")

        if not isinstance(n_sus, int) or n_sus < 0:
            raise ValueError("n_sus must be a non-negative integer")

        if (
            not isinstance(proportion_resistant, int)
            or proportion_resistant < 0
            or proportion_resistant > 1
        ):
            raise ValueError("proportion_resistant must be an integer between 0 and 1")

        samples_rows = []
        mutations_rows = []

        # check that if proportion_resistant > 0, n_res > 0
        if proportion_resistant > 0:
            assert n_res > 0, "if proportion_resistant > 0, n_res must be > 0"

        # iterate through the required number of samples
        for n_sample in range(n_samples):
            # make a copy of the reference gene
            sample_gene = [copy.deepcopy(gene) for gene in self.reference_gene]

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
                remaining_aa_positions = [
                    f"{gene.name}@{pos}"
                    for gene in sample_gene
                    for pos in gene.amino_acid_number
                ]

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
            for gene in sample_gene:
                gene._translate_sequence()

            # create mutations rows
            for gene_index, gene in enumerate(sample_gene):
                diff = self.reference_gene[gene_index] - gene
                for mut in diff.mutations:
                    if f"{gene.name}@{mut}" in selected_resistant_mutations:
                        mut_label = "R"
                    else:
                        mut_label = "S"
                    mutations_rows.append([n_sample, mut, mut_label, gene.name])

            # create samples rows
            sample_amino_acid_sequence = pd.DataFrame(
                ["".join(gene.amino_acid_sequence) for gene in sample_gene],
                index=[gene.name for gene in sample_gene],
                columns=["amino_acid_sequence"],
            )

            sequence_dict = sample_amino_acid_sequence["amino_acid_sequence"].to_dict()

            sample_metadata = {
                "sample_id": n_sample,
                "phenotype_label": label,
                "number_resistant_mutations": number_resistant,
                "number_susceptible_mutations": number_susceptible,
            }

            # Merge the sequence dictionary into the sample metadata
            sample_metadata.update(sequence_dict)

            samples_rows.append(sample_metadata)

        # Create the samples DataFrame
        samples = pd.DataFrame(samples_rows)
        samples.set_index("sample_id", inplace=True)

        # Create the mutations DataFrame
        mutations = pd.DataFrame(
            mutations_rows, columns=["sample_id", "mutation", "mutation_label", "gene"]
        )
        mutations.set_index(["sample_id", "mutation", "gene"], inplace=True)

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
        if self.multiple_genes:
            gene_resistant_mutations = specific_resistance_mutations[
                (specific_resistance_mutations.DRUG == self.drug)
                & (specific_resistance_mutations.GENE.isin(self.gene))
            ]
        else:
            gene_resistant_mutations = specific_resistance_mutations[
                (specific_resistance_mutations.DRUG == self.drug)
                & (specific_resistance_mutations.GENE == self.gene)
            ]

        return list(
            gene_resistant_mutations.GENE + "@" + gene_resistant_mutations.MUTATION
        )

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
            row = df.sample(n=1)
            value = row.gene.values[0] + "@" + row.mutation.values[0]
            selected_resistant_mutations.append(value)

        assert len(selected_resistant_mutations) == len(
            set(selected_resistant_mutations)
        ), "cannot have duplicate mutations"

        # insist we cannot mutate to Stop codons
        for mutation in selected_resistant_mutations:
            assert mutation[-1] != "!", "cannot mutate to STOP codon"

        # Get amino acid positions that are not altered by selected resistant mutations

        remaining_aa_positions = [
            f"{gene.name}@{pos}"
            for gene in sample_gene
            for pos in gene.amino_acid_number
            if f"{gene.name}@{pos}" not in selected_resistant_positions
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
            gene_name, susceptible_position = susceptible_position.split("@")
            current_gene = next(
                (gene for gene in sample_gene if gene.name == gene_name), None
            )

            ref_codon = current_gene.codons[
                current_gene.amino_acid_number == int(susceptible_position)
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
                f"{current_gene.name}@{ref_aa}{susceptible_position}{self.codon_to_amino_acid.get(alt_codon)}"
                for alt_codon in possible_alt_codons
                if sum(1 for a, b in zip(ref_codon, alt_codon) if a != b) == 1
                and self.codon_to_amino_acid.get(alt_codon)
                not in self.resistant_mutations
            ]

            # select one of these possible mutations
            if possible_alt_mutations:
                selected_susceptible_mutations.append(
                    random.sample(possible_alt_mutations, k=1)[0]
                )

        # insist we cannot mutate to Stop codons
        for mutation in selected_susceptible_mutations:
            assert mutation[-1] != "!", "cannot mutate to STOP codon"

        return number_susceptible, selected_susceptible_mutations

    def _apply_mutations(self, selected_mutations, sample_gene):
        # apply mutations to sample gene

        for mutation in selected_mutations:
            gene_name, mutation = mutation.split("@")
            ref_aa = mutation[0]
            alt_aa = mutation[-1]
            aa_pos = int(mutation[1:-1])

            current_gene = None
            gene_index = None
            for i, gene in enumerate(sample_gene):
                if gene.name == gene_name:
                    current_gene = gene
                    gene_index = i
                    break

            assert (
                current_gene is not None
            ), f"gene {gene_name} was not supplied in initialisation of batch object"

            assert (
                ref_aa
                == current_gene.amino_acid_sequence[
                    current_gene.amino_acid_number == aa_pos
                ][0]
            ), (
                "reference amino acid supplied in mutation does not match gene! :"
                + mutation
            )

            assert alt_aa != "!", "cannot mutate to STOP codon"

            ref_codon = current_gene.codons[current_gene.amino_acid_number == aa_pos][0]

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
                self.reference_gene[gene_index].nucleotide_sequence[
                    self.reference_gene[gene_index].nucleotide_number == base_pos
                ][0]
                == ref_base
            )

            sample_gene[gene_index].nucleotide_sequence[
                sample_gene[gene_index].nucleotide_number == base_pos
            ] = alt_base
