import numpy as np
from random import randint
from collections import Counter
from sklearn.metrics import accuracy_score
from random_forest import RandomForestClassifier
from joblib import Parallel, delayed

def initialization_of_population(size, n_feat):
    population = np.random.randint(0, 2, (size, n_feat))  # Randomly initialize population
    return population

def generations(df, label, size, n_feat, crossover_rate, mutation_rate, n_gen, X_rf_train, X_rf_test, y_rf_train, y_rf_test):
    best_chromo = []
    best_score = []
    population_nextgen = initialization_of_population(size, n_feat)
    clf = RandomForestClassifier(n_trees=100)  
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen, clf, X_rf_train, X_rf_test, y_rf_train, y_rf_test)
        print('Best score in generation', i + 1, ':', scores[:1])
        pop_after_sel = selection(size, pop_after_fit, crossover_rate)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score

def fitness_score(population, clf, X_rf_train, X_rf_test, y_rf_train, y_rf_test):
    def evaluate(chromosome):
        selected_features = np.where(chromosome)[0]
        X_rf_train_selected = X_rf_train[:, selected_features]
        X_rf_test_selected = X_rf_test[:, selected_features]
        clf.fit(X_rf_train_selected, y_rf_train)
        predictions = clf.predict(X_rf_test_selected)
        return accuracy_score(y_rf_test, predictions)

    scores = Parallel(n_jobs=-1)(delayed(evaluate)(chromosome) for chromosome in population)
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(size, pop_after_fit, crossover_rate):
    population_nextgen = []
    parent = int(crossover_rate * size)
    for i in range(parent):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0, len(pop_after_sel), 2):
        new_par = []
        child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
        new_par = np.concatenate((child_1[:len(child_1)//2], child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen

def mutation(pop_after_cross, mutation_rate, n_feat):
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []
    for chromo in pop_after_cross:
        mutated_chromo = chromo.copy()
        for _ in range(mutation_range):
            pos = randint(0, n_feat - 1)
            mutated_chromo[pos] ^= 1
        pop_next_gen.append(mutated_chromo)
    return pop_next_gen
