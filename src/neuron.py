#!/usr/bin/python

import math
import Gnuplot
from random import randrange, random

import phenotypes
import fitness
import genome
import logger

class ConvertNeuron(phenotypes.ConvertGenome):
    def __init__(self, fitness, timesteps=1001, bits_per_num=20):
        super(ConvertNeuron, self).__init__(fitness)
        self.__timesteps = timesteps
        self.bits = bits_per_num
        self.max_val = float(2**bits_per_num - 1)

    def grey_to_int(self, bits):
        '''Interpret the bits as if they were grey code'''
        b = [bits[0]]
        shift = b[-1]
        for i in bits[1:]:
            b.append(shift ^ i)
        s = 0
        for i in range(len(b), 0, -1):
            s += b[i-1] * 2**(len(b)-i)
        return s

    def convert(self, gene):
        gene_val = gene.get_value()
        a_perc = self.grey_to_int(gene_val[0:self.bits]) / self.max_val
        b_perc = self.grey_to_int(gene_val[self.bits:2*self.bits]) / self.max_val
        c_perc = self.grey_to_int(gene_val[2*self.bits:3*self.bits]) / self.max_val
        d_perc = self.grey_to_int(gene_val[3*self.bits:4*self.bits]) / self.max_val
        k_perc = self.grey_to_int(gene_val[4*self.bits:5*self.bits]) / self.max_val
        a = a_perc*(0.2 - 0.001) + 0.001
        b = b_perc*(0.3 - 0.01) + 0.01
        c = c_perc*(-30 + 80) - 80
        d = d_perc*(10 - 0.1) + 0.1
        k = k_perc*(1.0 - 0.01) + 0.01
        return NeuronPheno(gene, self.fitness, a, b, c, d, k, self.__timesteps)

class NeuronPheno(phenotypes.Phenotype):
    def __init__(self, gene, fit, a, b, c, d, k, timesteps):
        super(NeuronPheno, self).__init__(gene, fit)
        assert 0.001 <= a <= 0.2, 'a is out of range'
        assert 0.01 <= b <= 0.3, 'b is out of range'
        assert -80 <= c <= -30, 'c is out of range'
        assert 0.1 <= d <= 10, 'd is out of range'
        assert 0.01 <= k <= 1.0, 'k is out of range'
        assert 0 < timesteps, 'Timesteps must be larger than zero'
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d
        self.__k = k
        self.__time = timesteps
        self.__spike_train = None

    def get_config(self):
        return self.__a, self.__b, self.__c, self.__d, self.__k, self.__time

    def __str__(self):
        return 'Neuron {0!s}, fit: {1}'.format(self.get_config(), self.fitness(None))

    def get_spike_train(self):
        if self.__spike_train:
            return self.__spike_train[:]
        else:
            self.__spike_train = []
            u = 0
            v = -60
            dv = 0
            du = 0
            for i in range(self.__time):
                dv = 0.1 * (self.__k * (v**2) + 5*v + 140 - u + 10)
                du = (self.__a/10.0) * (self.__b*v - u)
                v += dv
                u += du
                self.__spike_train.append(v)
                if v > 35:
                    v = self.__c
                    u += self.__d
            return self.get_spike_train()

class NeuronFitness(fitness.BitSequenceFitness):
    def __init__(self, comp_filename):
        self.filename = comp_filename
        self.data = None
        self.__read_comparison()
        self.spike_data = self.calc_spikes(self.data)

    def __read_comparison(self):
        with open(self.filename, 'r') as f:
            data = f.readline()
            self.data = map(float, data.strip().split(' '))

    def calc_spikes(self, data):
        '''Retrieve the peaks in the spike train'''
        spikes = []
        found = False
        last_spike = -5
        data_len = len(data)
        for i in range(data_len):
            if data[i] >= 35:
                if i - last_spike > 2:
                    start = i - 1 if i - 1 >= 0 else 0
                    end = i + 2 if i + 2 < data_len else data_len - 1
                    found = False
                    for j in range(start, end + 1):
                        if i == j:
                            continue
                        if data[j] > data[i]:
                            found = True
                            break
                    if not found:
                        spikes.append((i, data[i]))
                        last_spike = i
        return spikes

    def spike_penalty(self, spike1, spike2, train1_len, train2_len):
        if len(spike1) < len(spike2):
            spike1, spike2 = spike2, spike1
            train1_len, train2_len = train2_len, train1_len
        return float((len(spike1) - len(spike2))*train1_len) / float(2*train1_len)

class WDM(NeuronFitness):
    '''Waveform Distance Metric fitness'''
    def __init__(self, filename, sample_interval=5):
        super(WDM, self).__init__(filename)
        self.__sample = sample_interval

    def sub_eval(self, pheno, population):
        spike = pheno.get_spike_train()
        assert len(spike) == len(self.data), 'Data and spike is different'
        s = 0
        for i in range(0, len(spike), self.__sample):
            s += abs(spike[i] - self.data[i])**2
        return 1.0 / (math.sqrt(s) / (len(spike) / float(self.__sample)))

class STDM(NeuronFitness):
    '''Spike Time Distance Metric fitness'''
    def sub_eval(self, pheno, population):
        spike_pheno = self.calc_spikes(pheno.get_spike_train())
        spike_data = self.spike_data
        s = 0
        n = min(len(spike_pheno), len(spike_data))
        for i in range(n):
            s += abs(spike_pheno[i][0] - spike_data[i][0])**2
        s += self.spike_penalty(spike_pheno, spike_data, len(pheno.get_spike_train()),
                len(self.data))
        return 1.0 / (math.sqrt(s) / (float(n) if n > 0 else 1.0))

class SIDM(NeuronFitness):
    '''Spike Interval Distance Metric fitness'''
    def sub_eval(self, pheno, population):
        spike_pheno = self.calc_spikes(pheno.get_spike_train())
        spike_data = self.spike_data
        s = 0
        for i in range(1, min(len(spike_pheno), len(spike_data))):
            s += abs((spike_pheno[i][0]- spike_pheno[i - 1][0]) -
                    (spike_data[i][0] - spike_data[i - 1][0]))**2
        s += self.spike_penalty(spike_pheno, spike_data, len(pheno.get_spike_train()),
                len(self.data))
        return 1.0 / (math.sqrt(s) / float(min(len(spike_pheno), len(spike_data))))

class NeuroGenome(genome.Genome):
    '''Use X bits per variable in the neuron, this means that in order to
    ensure that crossover passes on proper genes we need to re implement
    crossover. This genome will also just do 1 point crossover per variable.'''
    def __init__(self, val, cross_rate, mute_rate, convert_func, bits_per_num=20):
        super(NeuroGenome, self).__init__(val, 0.0, cross_rate, mute_rate,
                convert_func)
        assert len(self) == 5*bits_per_num, 'The length is not correct for a neuron genome'
        self.bits = bits_per_num

    def crossover(self, other):
        assert other, 'Other can\'t be nothing'
        my_val = self.get_value()
        other_val = other.get_value()
        my_cpy = self.get_value()
        other_cpy = other.get_value()
        if random() < self.cross_rate:
            for i in range(0, self.len, 10):
                my_val[i:i+5] = other_cpy[i:i+5]
        if random() < self.cross_rate:
            for i in range(0, self.len, 10):
                other_val[i+5:i+10] = my_cpy[i+5:i+10]
        return (NeuroGenome(my_val, self.cover_rate,
            self.mute_rate, self.convert_func, self.bits), NeuroGenome(other_val,
                 self.cross_rate, self.mute_rate, self.convert_func, self.bits))

    def __repr__(self):
        return "NeuroGenome({!r}, {!r}, {!r}, {})".format(self.val,
                self.cross_rate, self.mute_rate, self.convert_func)

class NeuronLogger(logger.FitnessLogger):
    def __init__(self, name, args, interactive=True):
        self.__args = args
        self.filename = name
        self.__gens = []
        self.__avgs = []
        self.__stdevs = []
        self.__bests = []
        self.__maxs = []
        self.__mins = []
        self.__best = None
        self.__inter = interactive
        if not interactive:
            assert self.filename, 'Filename must not be empty'

    def sub_call(self, i, best, avg, stdev, pop):
        self.__gens.append(i)
        self.__avgs.append(avg)
        self.__stdevs.append(stdev)
        self.__bests.append(best.fitness(pop.get()))
        self.__maxs.append(avg + stdev)
        self.__mins.append(avg - stdev)
        self.__best = best

    def read_comp(self, filename):
        res = None
        with open(filename, 'r') as f:
            res = f.readline()
        return map(float, res.strip().split(' '))

    def configure_plotter(self, g):
        g('set style line 80 lt rgb "#808080"')
        g('set style line 91 lt 0')
        g('set style line 81 lt rgb "#808080"')
        g('set border 3 back linestyle 80')
        g('set grid back linestyle 81')
        g('set xtics nomirror')
        g('set ytics nomirror')
        g('set key bottom right')
        return g

    def plot_fitness(self):
        g = Gnuplot.Gnuplot()
        g = self.configure_plotter(g)
        g.title('Fitness growth')
        g.xlabel('Generation')
        g.ylabel('Fitness')
        stdev_plot = Gnuplot.Data(self.__gens, self.__maxs, self.__mins,
                title='Standard deviation', with_='filledcurves')
        avg_plot = Gnuplot.Data(self.__gens, self.__avgs,
                title='Average', with_='lines')
        best_plot = Gnuplot.Data(self.__gens, self.__bests,
                title='Best', with_='lines')
        if not self.__inter:
            g('set terminal pdfcairo rounded enhanced')
            g('set output {}_fitness.pdf'.format(self.filename))
        g.plot(stdev_plot, avg_plot, best_plot)
        return g

    def plot_spike(self):
        g = Gnuplot.Gnuplot()
        g = self.configure_plotter(g)
        g.title('Spike Train')
        g.xlabel('Time(ms)')
        g.ylabel('Activation-Level(mV)')
        r = range(1001)
        other = self.read_comp(self.__args.data_file)
        my_spike = Gnuplot.Data(r, self.__best.get_spike_train(),
                title='Best approximation', with_='lines')
        target_spike = Gnuplot.Data(r, other,
                title='Target', with_='lines')
        if not self.__inter:
            g('set terminal pdfcairo rounded enhanced')
            g('set output {}_spike.pdf'.format(self.filename))
        g.plot(my_spike, target_spike)
        return g

    def write_config(self):
        if not self.__inter:
            with open('{}_conf.txt'.format(self.filename), 'w') as f:
                f.writeline('{0!s}\n{1!s}\n'.format(self.__args,
                    self.__best))
        else:
            print '{0!s}\n{1!s}\n'.format(self.__args,
                    self.__best)

    def sub_finish(self):
        a = self.plot_fitness()
        b = self.plot_spike()
        self.write_config()
        if self.__inter:
            raw_input('Press enter to continue...')
