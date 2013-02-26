#!/usr/bin/python

def evolution_loop(pop, protocol, logger, fit, max_loop=100, cond=None):
    population = pop
    for i in range(1, max_loop + 1):
        print '\rProgress: {0:.0%}'.format(i/float(max_loop)),
        fit.reset() #Done for space reasons
        best, avg, stdev = population.get_stats()
        logger(i, best, avg, stdev, population)
        if cond and best.fitness(population.get()) == cond:
            print 'Progress: {0:.0%}'.format(1)
            break
        population = protocol.select(population)
    logger.finish()
