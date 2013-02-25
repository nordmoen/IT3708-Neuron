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
        self.val = bitarray(val)
        self.cross_rate = cross_rate
        self.cover_rate = cover_rate if cover_rate <= 0.5 else 1 - cover_rate
        self.mute_rate = mute_rate
        self.convert_func = convert_func
        self.len = len(self.val)
        self.amount = None
        self.m_amount = None

    def convert(self):
        if self.convert_func:
            return self.convert_func(self)

    def crossover(self, other):
        assert other, 'The other genome must not be None'
        if not self.amount:
            self.amount = int(self.cover_rate * self.len)
        point = randrange(self.len - self.amount)
        end = point + self.amount
        my_val = self.get_value()
        other_val = other.get_value()
        if random() < self.cross_rate:
            my_val[point:end] = other.get_value()[point:end]
        if random() < self.cross_rate:
            other_val[point:end] = self.get_value()[point:end]
        return (Genome(my_val, self.cover_rate, self.cross_rate,
            self.mute_rate, self.convert_func), Genome(other_val,
                self.cover_rate, self.cross_rate, self.mute_rate, self.convert_func))

    def mutate(self):
        for i in range(len(self)):
            if self.mute_rate > random():
                self.val[i] = not self.val[i]

    def get_value(self):
        return self.val.copy()

    def __repr__(self):
        return "Genome({!r},{!r},{!r},{})".format(self.val,
                self.cross_rate, self.mute_rate, self.convert_func)

    def __str__(self):
        return "{0:s}".format(self.val)

    def __len__(self):
        return self.len
