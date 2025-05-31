import Compiler.types as types
import Compiler.library as lib

from Compiler import ml
from Compiler.types import sint, cfix, sfix, MultiArray, Array
from Compiler.library import print_ln

import operator
from functools import reduce
from Compiler import util

tf = ml


def comp_mod_zscore_threshold_matrix(matrix, threshold, debug):

    threshold_matrix = MultiArray(list(matrix.sizes), matrix.value_type)

    if debug:
        mad_matrix =  MultiArray(list(matrix.sizes), matrix.value_type)
    else:
        mad_matrix = None

    @lib.for_range(len(matrix))
    def _(i):
        raw_array = matrix.get_part(i, 1)[0]

        score_array = mad_modified_z_score(raw_array)

        @lib.for_range(score_array.length)
        def _(j):
            threshold_matrix[i][j] = score_array[j] > threshold

        if debug:
            mad_matrix.get_part(i, 1).assign(score_array)

    return threshold_matrix, mad_matrix


def compute_median(X_arr):


    X_copy = types.Array(X_arr.length, X_arr.value_type)
    X_copy.assign(X_arr)
    X_copy.sort()
    the_median = 0
    if X_arr.length % 2 == 0:
        the_median = (X_copy[(X_copy.length-1)//2] + X_copy[X_copy.length//2])/2
    else:
        the_median = X_copy[X_copy.length//2]
    return the_median




def mad_modified_z_score(input_arr):


    n = input_arr.length

    input_arr_median = compute_median(input_arr)


    abs_deviations = input_arr.same_shape()
    @lib.for_range(n)
    def _(i):
        abs_deviations[i] = abs(input_arr[i] - input_arr_median)

    MAD = compute_median(abs_deviations)
    consistency_constant = 1.486
    denominator_mad = (consistency_constant * MAD)


    consistency_constant_mean = 1.253314
    mean = sum(input_arr) / n

    tmp = input_arr.same_shape()
    @lib.for_range(n)
    def _(i):
        tmp[i] = abs(input_arr[i] - mean)
    mean_abs_deviation = sum(tmp) / n
    denominator_mad0 = consistency_constant_mean * mean_abs_deviation


    denominator = util.cond_swap(MAD==0, denominator_mad, denominator_mad0)[0]

    output_arr = input_arr.same_shape()
    @lib.for_range(n)
    def _(i):
        output_arr[i] = (input_arr[i] - input_arr_median) / denominator

    return output_arr



def print_predict_accuracy_model(model, test_samples, test_labels):
    guesses = model.predict(test_samples, batch_size=128)
    lib.print_ln('guess %s', guesses.reveal_nested()[:3])
    lib.print_ln('truth %s', test_labels.reveal_nested()[:3])

    @lib.map_sum_opt(28, 10000, [sint])
    def accuracy(i):
      correct = sint((ml.argmax(guesses[i].reveal()) == ml.argmax(test_labels[i].reveal())))
      return correct

    acc = accuracy().reveal()
    lib.print_ln("correct %s %s", acc, acc * cfix(0.0001))


def reveal_accuracy(preds, labels):

    lib.print_ln("Compute Accuracy:")

    lib.print_ln('  predictions %s', preds.reveal_nested()[:3])
    lib.print_ln('  labels %s', labels.reveal_nested()[:3])

    @lib.map_sum_opt(28, labels.sizes[0], [sint])
    def accuracy(i):
        correct = sint((ml.argmax(preds[i].reveal()) == ml.argmax(labels[i].reveal())))
        return correct

    acc = accuracy().reveal()
    lib.print_ln("  -> correct %s %s", acc, acc * cfix(0.0001))



def predict_on_model_copy(layers, original_model_to_copy_weights_from, batch_size, test_samples, test_labels):

    evaluation_model = tf.keras.models.Sequential(layers)
    optim = tf.keras.optimizers.SGD()
    evaluation_model.compile(optimizer=optim)
    evaluation_model.build(test_samples.sizes, batch_size=batch_size)

    for orig_var, eval_var in zip(original_model_to_copy_weights_from.trainable_variables, evaluation_model.trainable_variables):
        eval_var.address = orig_var.address

    evaluation_graph = evaluation_model.opt
    evaluation_graph.layers[0].X.address = test_samples.address
    evaluation_graph.layers[-1].Y.address = test_labels.address

    guesses = evaluation_graph.eval(test_samples, batch_size)

    return guesses




def compute_loss(X, Y, n_classes=10):
    n_data_owners = X.sizes[0]
    n_samples = X.sizes[1]
    loss_array = MultiArray([n_samples, n_data_owners], sfix)

    @lib.for_range(start=0, stop=n_samples, step=1)
    def _(sample_id):
        for model_id in range(n_data_owners):

            sum_holder = sfix(0)
            for out_class_id in range(n_classes):
                if n_classes == 2:
                    log_e = ml.log_e(X[model_id][sample_id][out_class_id])
                    tmp = -Y[sample_id] * log_e - (1 - Y[sample_id]) * log_e
                else:
                    tmp = Y[sample_id][out_class_id] * ml.log_e(X[model_id][sample_id][out_class_id])
                sum_holder += tmp

            loss_array[sample_id][model_id] = -sum_holder


    return loss_array



def euclidean_dist(A: MultiArray, B: MultiArray, n_threads):


    aTa = Array(len(A), A.value_type)
    @lib.for_range_multithread(n_threads, 1, len(A))
    def f(i):
        aTa[i] = sum(A[i] * A[i])

    print_ln("  aTa done")

    print(f"aTa={aTa.length}")

    bTb = Array(len(B) , B.value_type)
    @lib.for_range_multithread(n_threads, 1, len(B))
    def f(i):
        bTb[i] = sum(B[i] * B[i])

    print_ln("  bTb done")

    print(f"bTb={bTb.length}")

    L2 = A.dot(B.transpose())
    print(f"L2={L2.sizes}")

    print_ln("  AB done")

    @lib.for_range_opt_multithread(n_threads, len(A))
    def f(i):
        L2[i] = L2[i] * -2 + bTb + aTa[i]

    return L2


def euclidean_dist_dot_product(A: MultiArray, B: MultiArray, n_threads):


    aTa = Array(len(A), A.value_type)
    @lib.for_range_multithread(n_threads, 1, len(A))
    def f(i):
        aTa[i] = A.value_type.dot_product(A[i], A[i])

    print(f"aTa={aTa.length}")


    bTb = Array(len(B) , B.value_type)
    @lib.for_range_multithread(n_threads, 1, len(B))
    def f(i):
        bTb[i] = B.value_type.dot_product(B[i], B[i])

    print_ln("  bTb done")

    print(f"bTb={bTb.length}")

    L2 = A.dot(B.transpose())
    print(f"L2={L2.sizes}")

    print_ln("  AB done")

    @lib.for_range_opt_multithread(n_threads, len(A))
    def f(i):
        L2[i] = L2[i] * -2 + bTb + aTa[i]

    return L2


def flatten(M: MultiArray):
    part_size = reduce(operator.mul, M.sizes[1:])
    return MultiArray([M.sizes[0], part_size], M.value_type, address=M.address)


def from_numpy_to_multiarray(np_array, value_type):
    sizes = list(np_array.shape)
    multi_array = MultiArray(sizes, value_type)
    array = multi_array.to_array()
    array.set_range(0, np_array.flatten(order='C').tolist())
    return multi_array