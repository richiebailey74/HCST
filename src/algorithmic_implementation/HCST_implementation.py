import math
import random; random.seed()
import numpy as np
import cupy as cp

class HCST:

    def __init__(self, k, alpha, pass_num):
        self.f_vals = None
        self.partition_indices = None
        # omega = {k, alpha, pass_num} - these are the hyperparams
        self.k = k
        self.alpha = alpha
        self.pass_num = pass_num


    def fit(self, X_pass, y):
        _ = self.fit_transform(X_pass ,y)
        return


    def fit_transform(self, X_pass, y):
        X = X_pass.copy()

        inds = list(range(X.shape[1]))

        partition_indices = []
        for i in range(int(math.ceil(X.shape[1 ] /self.k))):
            try:
                temp = random.sample(inds ,self.k)
            except:
                temp = random.sample(inds ,X.shape[1 ] %self.k)
            partition_indices.append(temp)
            inds = list(set(inds).difference(set(temp)))

        noBreak = True
        passes = 0
        while noBreak:
            f_vectors = []
            for part_ind in partition_indices:
                X_partition = X[: ,part_ind]
                _, f_temp = self.CST_fit_cuda(cp.array(X_partition) ,y)
                f_t = [x.get( ) +0 for x in f_temp]
                f_vectors.append(f_t)

            next_partition_indices = [[] for x in range(len(partition_indices))]
            npi_ind_track = 0

            curr_partition = random.randint(0 ,len(partition_indices ) -1)
            while True:
                toJump = False
                if len(partition_indices[curr_partition]) == 0:
                    toJump = True
                    del partition_indices[curr_partition]
                    del f_vectors[curr_partition]
                    if len(partition_indices) == 0:
                        break # will only ever occur when a delete happens

                # generate random number to either "walk" in the current partition or "jump" to the next
                if self.alpha < random.uniform(0 ,1) or toJump:
                    curr_partition = random.randint(0 ,len(partition_indices ) -1)

                ind_to_use = self.walkJumpHelper(f_vectors, curr_partition)
                if ind_to_use is None:
                    print("ERROR: this statement should never print")
                    print("ERROR: none ind to use", f_vectors[curr_partition])
                    print("ERROR: none ind to use", partition_indices[curr_partition])
                partition_to_add = partition_indices[curr_partition][ind_to_use]
                del partition_indices[curr_partition][ind_to_use]
                del f_vectors[curr_partition][ind_to_use]

                if len(next_partition_indices[npi_ind_track]) == self.k:
                    npi_ind_track += 1

                next_partition_indices[npi_ind_track].append(partition_to_add)

            partition_indices = next_partition_indices


            passes += 1
            if passes == self.pass_num: # hard value passed as param, should/could be terminated dynamically eventually
                noBreak = False # will occur after a base case condition is reached or a certain number of iterations


        # must change and make walk optimization method
        self.partition_indices = partition_indices
        all_data, all_f = self.CST_recursive_fit_cuda_walk_optimization(cp.array(X), y)
        self.f_vals = all_f

        return all_data[-1][0].get() # X_trans equivalent


    # k is the max partition size of a "recursed" feature space to optimize time performance
    def CST_recursive_fit_cuda_walk_optimization(self, X_pass, y):
        # should have a list of lists to keep and store the X_split_trans as well as the f_temp values
        all_f = []
        all_data = []

        level_pass = 0
        X = X_pass.copy()
        while True:
            # construct the next level of data (concatenate all the CST transformed feature spaces)
            if len(all_f) != 0:
                X = None
                first = True
                for val in level_data:
                    X = val if first else cp.concatenate((X ,val), axis=1)
                    first = False
            level_f = []
            level_data = []
            level = True
            shifter = 0

            if level_pass == 0: # to utilize the walk optimized feature indices
                for part_ind in self.partition_indices:
                    X_split = X[: ,part_ind]
                    X_shift_trans, f_temp = self.CST_fit_cuda(X_split ,y)
                    level_data.append(X_shift_trans)
                    level_f.append(f_temp)
                    shifter += 1

                all_f.append(level_f)
                all_data.append(level_data)

            else:
                while level is True:
                    if (shifter + 1 ) * self.k >= X.shape[1]:
                        X_split = X[: ,(shifter * self.k):X.shape
                            [1]] # accounts for last interval not dividing into the feature space's size
                        level = False
                    else:
                        X_split = X[: ,(shifter * self.k):((shifter +1 ) * self.k)] # extract interval of k

                    X_shift_trans, f_temp = self.CST_fit_cuda(X_split ,y)
                    level_data.append(X_shift_trans)
                    level_f.append(f_temp)
                    shifter += 1

                all_f.append(level_f)
                all_data.append(level_data)

            if len(all_f[-1]) == 1: break # last transformation calculated
            level_pass += 1

        # split the f vectors from the transformed data

        return all_data, all_f


    def walkJumpHelper(self, f_vectors, curr_partition):
        rand_range = sum([abs(x) for x in f_vectors[curr_partition]])
        thresh_val = random.uniform(0 ,rand_range)

        sum_tracker = 0
        ind_to_use = None

        # selecting a feature stochastically using f vector magnitudes, get its index
        for ind, element in enumerate(f_vectors[curr_partition]):
            if sum_tracker > thresh_val:
                ind_to_use = ind
                break
            sum_tracker += abs(element)
            if ind == len(f_vectors[curr_partition] ) -1:
                ind_to_use = len(f_vectors[curr_partition] ) -1

        if ind_to_use is None:
            print("walk jump ERROR: this statement should never print")

        return ind_to_use


    def CST_fit_cuda(self, X, y):
        sampleNum = X.shape[0]
        featureNum = X.shape[1]
        temp_A = cp.zeros((featureNum, featureNum))

        for i in range(sampleNu m -1):
            for j in range( i +1, sampleNum):
                sample_difference = X[i] - X[j]

                if y[i] == y[j]:
                    temp_A += (cp.outer(sample_difference, sample_difference.T))
                else:
                    temp_A -= (cp.outer(sample_difference, sample_difference.T))

        eigvals, eigvecs = cp.linalg.eigh(temp_A)
        min_ind = cp.argmin(eigvals)
        min_eigvec = eigvecs[: ,min_ind]
        X_trans = X @ min_eigvec
        return cp.real(X_trans).reshape(-1 ,1), cp.real(min_eigvec).reshape(-1 ,1)


    def transform(self, X_pass):
        if not self.f_vals:
            print("No data has been fit to. Fit to data first!")
            return
        first = True
        X_pass = cp.array(X_pass)
        for level in self.f_vals:
            X = X_pass if first else X_temp

            if first: # to utilize the passed partition_indices
                X_temp = None
                shifter = 0
                for ind in range(len(level)):
                    X_split = X[: ,self.partition_indices[ind]]
                    X_split_trans = (X_split @ level[ind]).reshape(-1 ,1)
                    X_temp = X_split_trans if shifter == 0 else cp.concatenate((X_temp ,X_split_trans), axis=1)
                    shifter += 1
            else:
                X_temp = None
                shifter = 0
                for f in level:
                    X_split = X[: ,shifter:(shifter + f.shape[0])]
                    X_split_trans = (X_split @ f).reshape(-1 ,1)
                    X_temp = X_split_trans if shifter == 0 else cp.concatenate((X_temp ,X_split_trans), axis=1)
                    shifter += 1

            first = False

        return cp.asnumpy(X_temp)


    def get_transformation_vectors(self):
        if not self.f_vals:
            print("No data has been fit to. Fit to data first!")
            return

        return self.f_vals

