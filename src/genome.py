#!/usr/bin/python

from bitarray import bitarray
from random import randrange, random

class Genome(object):
    def __init__(self, val, cover_rate, cross_rate, mute_rate, convert_func):
        assert val, 'The value of the genome can\'t be None'
        assert convert_func, 'There must exist a conversion function'
        assert 0.0 <= cross_rate <= 1.0, 'The crossover rate must be within the range [0.0, 1.0]'
        assert 0.0 <= mute_rate <= 1.0, 'The mutation rate must be within the range [0.0, 1.0]'
        assert 0.0 <= cover_rate < 1.0, 'The cover rate must be within the range [0.0, 1.0)'
        self.__val = bitarray(val)
        self.__cross_rate = cross_rate
        self.__cover_rate = cover_rate if cover_rate <= 0.5 else 1 - cover_rate
        self.__mute_rate = mute_rate
        self.__convert_func = convert_func
        self.__len = len(self.__val)
        self.__amount = None
        self.__m_amount = None

    def convert(self):
        if self.__convert_func:
            return self.__convert_func(self)

    def crossover(self, other):
        assert other, 'The other genome must not be None'
        if not self.__amount:
            self.__amount = int(self.__cover_rate * self.__len)
        point = randrange(self.__len - self.__amount)
        end = point + self.__amount
        my_val = self.get_value()
        other_val = other.get_value()
        if random() < self.__cross_rate:
            my_val[point:end] = other.get_value()[point:end]
        if random() < self.__cross_rate:
            other_val[point:end] = self.get_value()[point:end]
        return (Genome(my_val, self.__cover_rate, self.__cross_rate,
            self.__mute_rate, self.__convert_func), Genome(other_val,
                self.__cover_rate, self.__cross_rate, self.__mute_rate, self.__convert_func))

    def mutate(self):
        for i in range(len(self)):
            if self.__mute_rate > random():
                self.__val[i] = not self.__val[i]

    def get_value(self):
        return self.__val.copy()

    def __repr__(self):
        return "Genome({!r},{!r},{!r},{})".format(self.__val,
                self.__cross_rate, self.__mute_rate, self.__convert_func)

    def __str__(self):
        return "{0:s}".format(self.__val)

    def __len__(self):
        return self.__len
