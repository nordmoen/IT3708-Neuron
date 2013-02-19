#!/usr/bin/python

from argparse import ArgumentParser
from random import seed, randint
from bitarray import bitarray

from evolution import evolution_loop
from genome import Genome
from population import Population
from phenotypes import ConvertGenome
import fitness
import logger
import select_mech
import select_protocol
import blotto

def create_parser():
    parser = ArgumentParser(description='Evolutionary Algorithm implementation')
    parser.add_argument('loops', type=int, default=100, help='The max number of generations')
    parser.add_argument('bits', type=int, default=20, help='The number of bits in the vectors')
    parser.add_argument('size', type=int, default=20, help='The size of each population')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--cross_rate', type=float, default=0.5,
            help='The rate of crossover, a number between (0.0, 0.5]')
    parser.add_argument('--cover_rate', type=float, default=0.5,
            help='The amount to crossover, 0.5 is then a one point crossover')
    parser.add_argument('--mutation', type=float, default=0.015,
            help='The rate of mutation, a number between (0.0, 0.5]')
    parser.add_argument('--stop_cond', type=float,
            help='The condition telling the algorithm when to stop if that condition occurs')

    proto_parser = parser.add_argument_group('Protocol', 'The selection protocol to use')
    proto_parser.add_argument('protocol', help='Type of protocol to use',
            choices=['full_generational', 'overproduction', 'mixing'],
            default='full_generational')
    proto_parser.add_argument('--num_parents', type=float, help=('The percentage of' +
            ' parents select for the mating'), default=0.5)
    proto_parser.add_argument('--num_children', type=int, help=('The percentage of' +
        ' children created during mating'), default=0.5)

    mech_parser = parser.add_argument_group('Mechanism', 'The selection mechanism to use')
    mech_parser.add_argument('mech', help='The type of mechanism to use',
            choices=['proportionate', 'sigma', 'tournament', 'rank'],
            default='proportionate')
    mech_parser.add_argument('--k', type=int, help=('When using tournament ' +
            'selection this must be used. It signifies the size of the tournament'),
            default=10)
    mech_parser.add_argument('--e', type=float, help=('When using tournament ' +
            'selection this must be used. It signifies the probability range.' +
            ' Smaller values creates smaller selection pressure on the best'),
            default=0.2)
    mech_parser.add_argument('--max', type=float, help=('When using rank ' +
            'selection this must be used. It signifies the max value'),
            default=1.5)
    mech_parser.add_argument('--min', type=float, help=('When using rank ' +
            'selection this must be used. It signifies the min value'),
            default=0.5)
    mech_parser.add_argument('--elite', type=int, help='The number of the best individuals to chose',
            default=0)

    fit_parser = parser.add_argument_group('Fitness', 'The fitness function')
    fit_parser.add_argument('fitness', help='The fitness function to use',
            choices=['max_fitness', 'random_fitness', 'blotto_fit'], default='max_fitness')
    fit_parser.add_argument('--target', help='The random target one want to try and match')

    convert_parser = parser.add_argument_group('Convert', 'The conversion function')
    convert_parser.add_argument('convert', help='The conversion function',
            choices=['convert-genome', 'convert-blotto'])

    log_parser = parser.add_argument_group('Logger', 'The logger function')
    log_parser.add_argument('log_type', help='The type of logger to use',
            choices=['cmd', 'plot', 'plotto'], default='cmd')
    log_parser.add_argument('--filename', help=('When using a plot logger' +
            ' this must be used.'))
    blott_parser = parser.add_argument_group('Blotto', 'Special configuration' +
            ' options for blotto')
    blott_parser.add_argument('--r', type=float, help='The redeployment factor',
            default=0.2)
    blott_parser.add_argument('--l', type=float, help='The loss fraction',
            default=0.0)
    return parser

def get_protocol(args, select_alg):
    protocol = args.protocol
    parents = args.num_parents
    if protocol == 'full_generational':
        return select_protocol.FullReplacement(select_alg, parents, args.elite)
    elif protocol == 'overproduction':
        return select_protocol.OverProduction(select_alg, parents, args.elite)
    elif protocol == 'mixing':
        return select_protocol.GenerationalMixing(select_alg, parents,
                args.elite)

def get_selection(args):
    select = args.mech
    if select == 'proportionate':
        return select_mech.FitnessProportionate()
    elif select == 'sigma':
        return select_mech.SigmaScaling()
    elif select == 'tournament':
        return select_mech.TournamentSelection(args.k, args.e)
    elif select == 'rank':
        return select_mech.RankSelection(args.min, args.max)

def get_logger(args):
    log_type = args.log_type
    if log_type == 'cmd':
        return logger.CmdLogger()
    elif log_type == 'plot':
        return logger.PlotLogger(args.filename)
    elif log_type == 'plotto':
        return blotto.BlottoLogger(args.filename)

def get_fit(args):
    '''Everyone should'''
    fit_func = args.fitness
    if fit_func == 'max_fitness':
        return fitness.OneMaxFitness()
    elif fit_func == 'random_fitness':
        return fitness.RandomBitSequenceFitness(bitarray(args.target))
    elif fit_func == 'blotto_fit':
        return blotto.BlottoFitness(args.bits, args.r, args.l)

def get_convert(args, fit):
    if args.convert == 'convert-genome':
        return ConvertGenome(fit)
    elif args.convert == 'convert-blotto':
        return blotto.ConvertBlotto(fit, args.bits)

def create_initial_population(args, conv):
    pop = []
    frac = 1
    if type(conv) == blotto.ConvertBlotto:
        frac = 5
    for i in range(args.size):
        pop.append(Genome([randint(0, 1) for i in range(args.bits*frac)],
            args.cover_rate, args.cross_rate, args.mutation, conv))
    return Population(map(lambda x: x.convert(), pop))

def main():
    parser = create_parser()
    args = parser.parse_args()
    seed(args.seed)
    select = get_selection(args)
    proto = get_protocol(args, select)
    fit = get_fit(args)
    conv = get_convert(args, fit)
    log = get_logger(args)
    init = create_initial_population(args, conv)
    evolution_loop(init, proto, log, args.loops, args.stop_cond)

if __name__ == '__main__':
    main()
