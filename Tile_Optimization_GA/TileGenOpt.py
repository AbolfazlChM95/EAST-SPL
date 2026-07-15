'''

Tile Configuration optimization using genetic algorithm

Methods and functions to run the evolutionary algorithm for tiling

method(s):
    EvolutionaryTiling

Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL
'''


import numpy as np
import random
import copy


class EvolutionaryTiling:
    """
    Optimize tiling using a genetic algorithm.

    Each individual contains two ordered coordinate lists correspond to collocated points
    representing the horizontal and vertical tile boundaries within the field mask.
    """
    def __init__(
            self, 
            mask, 
            min_len, 
            max_len, 
            pop_size=50, 
            mutation_rate=0.2
        ):
        

        if min_len <= 0 or max_len <= 0:
            raise ValueError("min_len and max_len must be positive.")
    
        if min_len > max_len:
            raise ValueError("min_len cannot be greater than max_len.")

        ones_x, ones_y = np.where(mask == 1)
        self.x_bounds = (int(ones_x.min()), int(ones_x.max()))
        self.y_bounds = (int(ones_y.min()), int(ones_y.max()))
        self.max_len = int(max_len)
        self.min_len = int(min_len)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.mutation_shift_rate = 0.3
        self.mutation_shift_sigma = 3
        self.mutation_add_rate = 0.2 # 0.2
        self.mutation_remove_rate = 0.2 # 0.2
        self.evolve_keep_elite = 5 # 5
        self.new_immigrant_num = 5 # 5

    def generate_random_points(self, start, end):
        # Generates sorted points ensuring no gap > max_len
        points = [start]
        current = start
        while current < end:
            # Random step ensuring we don't exceed max_len but make progress
            step = random.randint(self.min_len, self.max_len)
            next_p = current + step
            if next_p >= end:
                break
            points.append(next_p)
            current = next_p
        points.append(end)
        return sorted(list(set(points)))
    
    def sanity_check(self, points):
        if not points: return points
        start_pt, end_pt = points[0], points[-1]
        clean_points = [start_pt]
        
        for i in range(1, len(points) - 1):
            if points[i] - clean_points[-1] >= self.min_len:
                clean_points.append(points[i])
        
        while len(clean_points) > 1 and (end_pt - clean_points[-1] < self.min_len):
            clean_points.pop()
            
        clean_points.append(end_pt)

        final_points = [clean_points[0]]
        for i in range(1, len(clean_points)):
            prev = final_points[-1]
            curr = clean_points[i]
            gap = curr - prev
            
            if gap > self.max_len:
                n_segments = int(np.ceil(gap / self.max_len))
                step_size = int(gap / n_segments)
                
                temp_curr = prev
                for _ in range(n_segments - 1):
                    temp_curr += step_size
                    final_points.append(temp_curr)
            
            final_points.append(curr)
            
        return sorted(list(set(final_points)))
       
    def init_population(self):
        pop = []
        for _ in range(self.pop_size):
            xs = self.generate_random_points(*self.x_bounds)
            ys = self.generate_random_points(*self.y_bounds)
            xs = self.sanity_check(xs)
            ys = self.sanity_check(ys)
            pop.append([xs, ys])
        return pop
    
    def crossover(self, p1, p2):
        # Value-based Interval Crossover
        # Select random interval within global bounds
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        
        for dim in range(2): # 0 for x, 1 for y
            axis_min, axis_max = (self.x_bounds if dim == 0 else self.y_bounds)

            # Define swap interval
            r1 = random.randint(axis_min, axis_max)
            r2 = random.randint(axis_min, axis_max)
            start_int, end_int = min(r1, r2), max(r1, r2)
            
            # Extract points within interval
            p1_in = [x for x in p1[dim] if start_int < x < end_int]
            p2_in = [x for x in p2[dim] if start_int < x < end_int]
            
            # Keep points outside interval
            p1_out = [x for x in p1[dim] if not (start_int < x < end_int)]
            p2_out = [x for x in p2[dim] if not (start_int < x < end_int)]
            
            # Swap and sort, but repair looking for min length
            raw_c1 = sorted(p1_out + p2_in)
            raw_c2 = sorted(p2_out + p1_in)
            c1[dim] = self.sanity_check(raw_c1)
            c2[dim] = self.sanity_check(raw_c2)
            
        return c1, c2

    def mutate(self, indiv):
        indiv_copy = copy.deepcopy(indiv)
        for dim in range(2): # Iterate x then y
            pts = indiv_copy[dim]
            if random.random() > self.mutation_rate: 
                continue

            # 1. Shift
            for i in range(1, len(pts)-1):
                if random.random() < self.mutation_shift_rate:
                    shift = int(round(random.gauss(0, self.mutation_shift_sigma)))

                    new_val = pts[i] + shift
                    dist_left = new_val - pts[i-1]
                    dist_right = pts[i+1] - new_val
                    if (dist_left >= self.min_len) and (dist_right >= self.min_len):
                            pts[i] = new_val

            # 2. Add new point
            if random.random() < self.mutation_add_rate:
                valid_indices = []
                for i in range(len(pts)-1):
                    if (pts[i+1] - pts[i]) >= (2 * self.min_len):
                        valid_indices.append(i)
                
                if valid_indices:
                    idx = random.choice(valid_indices)
                    
                    safe_min = pts[idx] + self.min_len
                    safe_max = pts[idx+1] - self.min_len
                    
                    if safe_max >= safe_min:
                        new_pt = int(random.randint(safe_min, safe_max))
                        pts.insert(idx+1, new_pt)

            # 3. Removal
            if len(pts) > 3 and random.random() < self.mutation_remove_rate:
                
                # Possible points to remove! considering Max_length size
                candidates = []
                for i in range(1, len(pts)-1): 
                    if (pts[i+1] - pts[i-1]) <= self.max_len:
                        candidates.append(i)
                
                if candidates:
                    rem_idx = random.choice(candidates)
                    pts.pop(rem_idx)
            
            indiv_copy[dim] = self.sanity_check(pts)
        return indiv_copy

    def evolve(self, objective_func, max_num_gens = 100, convergence_exit = 5):

        population = self.init_population()
        best_scores = []
        convergence_counter = 0
        generation = 0

        for generation  in range(max_num_gens):

            # Global sanity check
            for i in range(len(population)):
                population[i][0] = self.sanity_check(population[i][0])
                population[i][1] = self.sanity_check(population[i][1])

            # Evaluate
            scores = [objective_func(p) for p in population]
            
            # Sort by score (minimization)
            sorted_indices = np.argsort(scores)
            population = [population[i] for i in sorted_indices]
            best_score = scores[sorted_indices[0]]
            if best_scores and best_score == best_scores[-1]:
                convergence_counter += 1
            else:
                convergence_counter = 0
            best_scores.append(best_score)

            if convergence_counter >= convergence_exit:
                print("Convergence criteria met.")
                break

            print(f"Gen {generation}: Best Score {best_score:.4f} | Size X: {len(population[0][0])}| Size Y: {len(population[0][1])}")

            # Selection Elitism
            next_gen = population[:self.evolve_keep_elite]
            
            while len(next_gen) < (self.pop_size - self.new_immigrant_num):
                parent1 = random.choice(population[:self.pop_size//2])
                parent2 = random.choice(population[:self.pop_size//2])
                
                child1, child2 = self.crossover(parent1, parent2)
                next_gen.append(self.mutate(child1))
                if len(next_gen) < self.pop_size:
                    next_gen.append(self.mutate(child2))
            
            while len(next_gen) < self.pop_size:
                xs = self.generate_random_points(*self.x_bounds)
                ys = self.generate_random_points(*self.y_bounds)
                xs = self.sanity_check(xs)
                ys = self.sanity_check(ys)
                next_gen.append([xs, ys])
            
            population = next_gen
        else:
            print('Maximum number of generation has reached')

        scores = [objective_func(p) for p in population]
        best_idx = np.argmin(scores)

        return population[best_idx], scores[best_idx], best_scores