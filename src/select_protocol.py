#!/usr/bin/python

from random import sample
from math import ceil

from population import Population

class SelectionProtocol(object):
    def __init__(self, select_alg, num_children=1.0, eliteism=0):
        assert select_alg, 'The selection algorithm must be something'
        assert 0.0 < num_children, 'Must select some amount of children'
        self.select_alg = select_alg
        self.num_children = num_children
        self.elite = eliteism

    def select(self, population):
        assert population, 'The population must contain something'
        size = len(population)
        next_pop = self.sub_select(population)
        assert len(next_pop) == size, 'The new population size is different than requested'
        return next_pop

    def sub_select(self, population):
        pass

class FullReplacement(SelectionProtocol):
    def sub_select(self, population):
        return self.create_population(len(population), population)

    def create_population(self, amount, population):
        next_gen = []

        while len(next_gen) < amount:
            mom = self.select_alg.sample(1, population.get())[0]
            dad = self.select_alg.sample(1, population.get())[0]
            c1, c2 = mom.get_gene().crossover(dad.get_gene())
            next_gen.append(c1)
            next_gen.append(c2)
        while len(next_gen) > len(population) - self.elite:
            next_gen.pop()
        map(lambda x: x.mutate(), next_gen)
        next_gen = map(lambda x: x.convert(), next_gen)

        if self.elite > 0:
            sort_pop = sorted(population.get(), key=lambda x:
                    x.fitness(population.get()), reverse=True)
            next_gen.extend(sort_pop[:self.elite])

        return Population(next_gen)

class OverProduction(FullReplacement):
    def sub_select(self, population):
        next_pop = super(OverProduction,
            self).create_population(len(population)*self.num_children,
                population)
        return Population(self.select_alg.sample(len(population), next_pop.get()))

class GenerationalMixing(FullReplacement):
    def sub_select(self, population):
        next_pop = super(GenerationalMixing,
            self).create_population(len(population)*self.num_children,
                population)
        next_pop.extend(population.get())
        return Population(self.select_alg.sample(len(population), next_pop.get()))
