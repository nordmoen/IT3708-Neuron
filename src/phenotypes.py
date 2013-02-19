#!/usr/bin/python

class Phenotype(object):
    def __init__(self, gene, fitness_func):
        self.gene = gene
        self.fit = fitness_func

    def get_gene(self):
        return self.gene

    def fitness(self, pop):
        return self.fit(self, pop)

    def __str__(self):
        return 'Phenotype({0!s}), fitness:{1:.1f}'.format(self.gene, self.fitness(None))

    def __repr__(self):
        return 'Phenotype({0!r}, {1!r})'.format(self.gene, self.fit)

    def __eq__(self, other):
        try:
            return self.gene == other.get_gene()
        except:
            return False

class ConvertGenome(object):
    def __init__(self, fitness):
        self.fitness = fitness

    def __call__(self, gene):
        return self.convert(gene)

    def convert(self, gene):
        return Phenotype(gene, self.fitness)
