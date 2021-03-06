#!/usr/bin/python

from bitarray import bitarray

class BitSequenceFitness(object):
    def __init__(self):
        self.__fits = {}

    def __call__(self, pheno, population):
        assert pheno, 'The given phoneme sequence is None'
        if pheno in self.__fits:
            return self.__fits[pheno]
        else:
            self.__fits[pheno] = self.sub_eval(pheno, population)
            return self.__fits[pheno]

    def reset(self):
        self.__fits.clear()

    def sub_eval(self, pheno, population):
        pass

class OneMaxFitness(BitSequenceFitness):
    def sub_eval(self, pheno, population):
        return pheno.get_gene().get_value().count()

class RandomBitSequenceFitness(BitSequenceFitness):
    def __init__(self, target):
        assert target, 'The target bit sequence is None'
        assert isinstance(target, bitarray), 'The target needs to be a bitarray'
        super(RandomBitSequenceFitness, self).__init__()
        self.__target = target

    def sub_eval(self, pheno, population):
        assert len(pheno.get_gene()) == len(self.__target), ('The target sequence has a ' +
                'different length than the given gene')
        count = 0
        bitArr = pheno.get_gene().get_value()
        for i in range(len(bitArr)):
            if bitArr[i] != self.__target[i]:
                count += 1
        return count
