import copy
import numpy


class Sample:
    """
    Class for a single sample

    Args:
        reference: reference gumpy.Gene object
    """

    def __init__(self, reference):
        self.reference = reference
        # make a copy of the reference gene
        self.gene = copy.deepcopy(reference)

        aminoacids = "FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
        bases = ["t", "c", "a", "g"]
        all_codons = numpy.array(
            [a + b + c for a in bases for b in bases for c in bases]
        )
        self.amino_acid_to_codon = {}
        for amino_acid, codon in zip(aminoacids, all_codons):
            self.amino_acid_to_codon.setdefault(amino_acid, []).append(codon)

    def apply_mutations(self, mutations):
        """
        Apply mutations to the sample

        Args:
            mutations: list of mutations to apply to the sample
        """

        for mutation in mutations:
            ref_aa = mutation[0]
            alt_aa = mutation[-1]
            aa_pos = int(mutation[1:-1])

            assert (
                ref_aa
                == self.gene.amino_acid_sequence[self.gene.amino_acid_number == aa_pos][
                    0
                ]
            ), "reference amino acid supplied in mutation does not match gene!"

            assert alt_aa != "!", "cannot mutate to STOP codon"

            ref_codon = self.gene.codons[self.gene.amino_acid_number == aa_pos][0]

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
                self.reference.nucleotide_sequence[
                    self.reference.nucleotide_number == base_pos
                ][0]
                == ref_base
            )

            self.gene.nucleotide_sequence[
                self.gene.nucleotide_number == base_pos
            ] = alt_base

        # translate the nucleotide sequence to amino acids
        self.gene._translate_sequence()

        # create a string of the amino acid sequence
        self.amino_acid_sequence = "".join(i for i in self.gene.amino_acid_sequence)

        diff = self.reference - self.gene

        self.mutations = diff.mutations
