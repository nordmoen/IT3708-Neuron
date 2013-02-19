#!/usr/bin/python

class FitnessLogger(object):
    def __call__(self, i, best, avg, stdev, pop):
        return self.sub_call(i, best, avg, stdev, pop)

    def finish(self):
        return self.sub_finish()

    def sub_call(self, i, best, avg, stdev, population):
        pass

class CmdLogger(FitnessLogger):
    def sub_call(self, i, best, avg, stdev, pop):
        print '-'*30
        print 'Generation: {0:d}'.format(i)
        print 'Best: {0:s}'.format(best)
        print 'Average fitness: {0:.2f}, stdev:{1:.2f}'.format(avg, stdev)
        print '-'*30

    def sub_finish(self):
        pass

class PlotLogger(FitnessLogger):
    def __init__(self, name):
        assert name, 'Can\'t create a file without a name'
        self.__log = []
        self.filename = name

    def sub_call(self, i, best, avg, stdev, pop):
        self.__log.append('{0:d}\t{1:f}\t{2:f}\t{3:f}\n'.format(i, avg, stdev,
            best.fitness(pop.get())))

    def sub_finish(self):
        with open(self.filename, 'w') as f:
            f.writelines(self.__log)
