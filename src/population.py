#!/usr/bin/python

from math import sqrt

class Population(object):
    def __init__(self, elements):
        assert elements, 'A population must contain some elements'
        self.__vals = elements
        self.__avg = None
        self.__best = None
        self.__stdev = None

    def get_stats(self):
        if not self.__avg:
            self.__avg = (reduce(lambda acc,y: acc + y.fitness(self.get()), self.__vals, 0) /
                    float(len(self.__vals)))
        if not self.__best:
            self.__best = max(self.__vals, key=lambda x: x.fitness(self.get()))
        if not self.__stdev:
            self.__stdev = sqrt(reduce(lambda acc,x: acc + (x.fitness(self.get()) - self.__avg)**2,
                self.__vals, 0) / float(len(self.__vals)))
        return (self.__best, self.__avg, self.__stdev)

    def get(self):
        return self.__vals[:]

    def __str__(self):
        return str(self.__vals)

    def __repr__(self):
        return "Population({!r})".format(self.__vals)

    def __len__(self):
        return len(self.__vals)

    def extend(self, li):
        assert li, 'List is None'
        self.__vals.extend(li)
