#!/usr/bin/python

import math

import phenotypes
import fitness

class ConvertNeuron(phenotypes.ConvertGenome):
    def __init__(self, fitness, timesteps=1001):
        super(ConvertNeuron, self).__init__(fitness)
        self.__timesteps = timesteps

    def convert(self, gene):
        pass

class NeuronPheno(phenotypes.Phenotype):
    def __init__(self, gene, fit, a, b, c, d, k, timesteps):
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

    def get_config():
        return self.__a, self.__b, self.__c, self.__d, self.__k, self.__time

    def get_spike_train():
        if self.__spike_train:
            return self.__spike_train[:]
        else:
            u = 0
            v = -60
            dv = 0
            du = 0
            for i in range(self.__time):
                dv = 0.1 * (self.__k * (v**2) + 5*v + 140 - u + 10)
                du = (self.__a/10) * (self.__b*v - u)
                v += dv
                u += du
                self.__spike_train.append(v)
                if v > 35:
                    v = self.__c
                    u = self.__u + self.__d
            return self.get_spike_train()

class NeuronFitness(fitness.BitSequenceFitness):
    def __init__(self, comp_filename):
        self.filename = comp_filename
        self.data = None
        self.__read_comparison()

    def __read_comparison(self):
        with open(self.filename, 'r') as f:
            data = f.readline()
            self.data = map(float, data.split(' '))

    def calc_spikes(self, data):
        '''Retrieve the peaks in the spike train'''
        spikes = []
        max_spike = 0
        last_spike = -5
        for i in range(len(data)):
            if i - last_spike > 2:
                if i - 2 >= 0:
                    max_spike = max(max(data[i-2:i]), max(data[i+1:i+3]))
                elif i - 2 == -2:
                    max_spike = max(data[i+1:i+3])
                else:
                    max_spike = max(max(data[i-1:i]), max(data[i+1:i+3]))
                if data[i] >= max_spike > 0:
                    spikes.append((i, data[i]))
                    last_spike = i
        return spikes

    def spike_penalty(self, spike1, spike2, spike1_len, spike2_len):
        if len(spike1) < len(spike2):
            spike1, spike2 = spike2, spike1
            spike1_len, spike2_len = spike2_len, spike1_len
        return float((len(spike1) - len(spike2))*spike1_len) / float(2*spike1_len)

class WDM(NeuronFitness):
    '''Waveform Distance Metric fitness'''
    def sub_eval(self, pheno, population):
        spike = pheno.get_spike_train()
        assert len(spike) == len(self.data), 'Data and spike is different'
        s = 0
        for i in range(len(spike)):
            s += abs(spike[i] - self.data[i])**2
        return math.sqrt(s) / len(spike)

class STDM(NeuronFitness):
    '''Spike Time Distance Metric fitness'''
    def sub_eval(self, pheno, population):
        spike_pheno = self.calc_spikes(pheno.get_spike_train())
        spike_data = self.calc_spikes(self.data)
        s = 0
        for i in range(min(len(spike_pheno), len(spike_data))):
            s += abs(spike_pheno[i][0] - spike_data[i][0])**2
        s += self.spike_penalty(spike_pheno, spike_data, len(pheno.get_spike_train()),
                len(self.data))
        return math.sqrt(s) / min(len(spike_pheno), len(spike_data))

class SIDM(NeuronFitness):
    '''Spike Interval Distance Metric fitness'''
    def sub_eval(self, pheno, population):
        spike_pheno = self.calc_spikes(pheno.get_spike_train())
        spike_data = self.calc_spikes(self.data)
        s = 0
        for i in range(1, min(len(spike_pheno), len(spike_data))):
            s += abs((spike_pheno[i][0]- spike_pheno[i - 1][0]) -
                    (spike_data[i][0] - spike_data[i - 1][0]))**2
        s += self.spike_penalty(spike_pheno, spike_data, len(pheno.get_spike_train()),
                len(self.data))
        return math.sqrt(s) / min(len(spike_pheno), len(spike_data))
