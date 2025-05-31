
from Compiler.types import MultiArray, Array, sfix, cfix, sint, cint, regint, MemValue, Matrix
from Compiler.dijkstra import HeapQ
from Compiler.oram import OptimalORAM
from Compiler.permutation import cond_swap

from Compiler import library as lib
from Compiler.library import print_ln
from Compiler import util
from Compiler import types

from Compiler.script_utils import audit_function_utils as audit_utils
from Compiler.script_utils.timers import TIMER_AUDIT_SHAP_BUILD_COALITIONS, TIMER_AUDIT_SHAP_BUILD_Z_SAMPLES, \
    TIMER_AUDIT_SHAP_EVAL_SAMPLES, TIMER_AUDIT_SHAP_MARGINAL_CONTRIBUTION, TIMER_AUDIT_SHAP_LINREG

import ml
import numpy as np
import copy
from scipy.special import binom
import itertools


def audit(input_loader, config, debug: bool):

    audit_trigger_samples, _audit_trigger_mislabels = input_loader.audit_trigger()
    n_audit_trigger_samples = audit_trigger_samples.sizes[0]

    num_features = audit_trigger_samples.sizes[1]
    num_samples = 2 * 5

    if num_samples < num_features:
        print("WARNING: Number of samples is lower than number of features to explain, the explanations may not"
              "work correctly as the linear system is undetermined")

    train_samples, _train_labels = input_loader.train_dataset()
    if config.n_batches > 0:
        train_samples = train_samples.get_part(0, config.n_batches * config.batch_size)
        _train_labels = _train_labels.get_part(0, config.n_batches * config.batch_size)
        print("Using only first %s batches of training data" % config.n_batches)

    n_train_samples = len(train_samples)

    @lib.for_range_opt(1)
    def shap_loop(audit_trigger_idx):
        print_ln("  audit_trigger_idx=%s", audit_trigger_idx)

        audit_trigger_sample = audit_trigger_samples[audit_trigger_idx]
        audit_trigger_label = _audit_trigger_mislabels[audit_trigger_idx]
        lib.start_timer(TIMER_AUDIT_SHAP_BUILD_COALITIONS)
        z_coalitions, kernelWeights = build_subsets_order(num_samples, num_features)
        z_samples = Matrix(num_samples * n_train_samples, num_features, audit_trigger_sample.value_type)
        # regint style
        z_coalitions_runtime = Matrix(num_samples, num_features, regint)
        z_coalitions_list = z_coalitions.tolist()
        for i in range(len(z_coalitions_list)):
            z_coalitions_runtime[i].assign(z_coalitions_list[i])

        z_coalitions_kernelWeights_runtime = Matrix(num_samples, num_features, cfix)
        z_coalitions_kernelWeights_list = (np.expand_dims(kernelWeights, 1) * z_coalitions).tolist()
        for i in range(len(z_coalitions_list)):
            z_coalitions_kernelWeights_runtime[i].assign(z_coalitions_kernelWeights_list[i])
        lib.stop_timer(TIMER_AUDIT_SHAP_BUILD_COALITIONS)


        lib.start_timer(TIMER_AUDIT_SHAP_BUILD_Z_SAMPLES)
        @lib.for_range_opt(num_samples)
        def coalitions(z_i):
            @lib.for_range_opt(n_train_samples)
            def coalition_train_sample(train_idx):
                @lib.for_range_opt(num_features)
                def ran(f_i):
                    z_samples[(z_i * n_train_samples) + train_idx][f_i] = audit_trigger_sample[f_i] * z_coalitions_runtime[z_i][f_i] \
                                                                          + (1 - z_coalitions_runtime[z_i][f_i]) * train_samples[train_idx][f_i]
        lib.stop_timer(TIMER_AUDIT_SHAP_BUILD_Z_SAMPLES)

        print("Done generating z_samples", z_samples.sizes)

        lib.start_timer(TIMER_AUDIT_SHAP_EVAL_SAMPLES)

        model = input_loader.model()
        prediction_results = model.eval(z_samples, batch_size=config.batch_size)
        prediction_results_ex = Array(num_samples, sfix)
        prediction_results_ex.assign_all(sfix(0))

        lib.stop_timer(TIMER_AUDIT_SHAP_EVAL_SAMPLES)
        lib.start_timer(TIMER_AUDIT_SHAP_MARGINAL_CONTRIBUTION)

        @lib.for_range_opt(num_samples)
        def coalitions_marginal(z_i):
            @lib.for_range_opt(n_train_samples)
            def summer(i):
                prediction_results_ex[z_i] = prediction_results_ex[z_i] + prediction_results[(z_i * n_train_samples) + i]

            prediction_results_ex[z_i] = prediction_results_ex[z_i] * cfix(1. / n_train_samples)

        if debug:
            print_ln("Average prediction without feature: %s", prediction_results_ex.reveal())
        lib.stop_timer(TIMER_AUDIT_SHAP_MARGINAL_CONTRIBUTION)
        lib.start_timer(TIMER_AUDIT_SHAP_LINREG)


        invert_res = invert_compile_time(z_coalitions, kernelWeights)
        print("invert_res", invert_res.shape)

        runtime_xtxinv = Matrix(invert_res.shape[0], invert_res.shape[1], cfix)
        print("Compile-time", invert_res[0][0], invert_res[0][1])

        for i in range(num_samples):
            runtime_xtxinv[i].assign(invert_res[i].tolist())

        print("Compile-time", invert_res[0][0], invert_res[0][1])
        if debug:
            print_ln("Runtime %s %s", runtime_xtxinv[0][0], runtime_xtxinv[0][1])

        print("prediction_results_ex", prediction_results_ex)
        print("z_coalitions_runtime", z_coalitions_runtime)

        secret_xty = z_coalitions_kernelWeights_runtime.transpose().dot(prediction_results_ex)
        print("secret_xty", secret_xty)
        print("runtime_xtxinv", runtime_xtxinv)
        if debug:
            print_ln("runtime_xtxinv %s", runtime_xtxinv)
            print_ln("secret_xty %s", secret_xty.reveal_nested())
        secret_params = runtime_xtxinv.dot(secret_xty)
        if debug:
            print_ln("SECRET PARAMS %s", secret_params.reveal_nested())
            print_ln("SUM %s", sum(secret_params.reveal_nested()))
        lib.stop_timer(TIMER_AUDIT_SHAP_LINREG)


    return {}, {}



def convert_np_regint(input):
    return regint(int(input))

def audit_shap(input_loader, config, debug: bool):
    train_samples, _train_labels = input_loader.train_dataset()
    n_train_samples = len(train_samples)

    lib.start_timer(104)
    _train_labels_idx = Array(len(train_samples), sint)
    @lib.for_range(len(train_samples))
    def _(i):
        _train_labels_idx[i] = ml.argmax(_train_labels[i])

    lib.stop_timer(104)
    train_samples_ownership = Array(len(train_samples), sint)

    for party_id in range(input_loader.num_parties()):
        start, size = input_loader.train_dataset_region(party_id)
        train_samples_ownership.get_sub(start=start, stop=start+size).assign_all(party_id)

    train_samples_idx = Array.create_from(cint(x) for x in range(len(train_samples)))


    audit_trigger_samples, _audit_trigger_mislabels = input_loader.audit_trigger()



    model = input_loader.model()
    latent_space_layer, expected_latent_space_size = input_loader.model_latent_space_layer()
    latent_space_layer=None

    lib.start_timer(105)
    print_ln("Computing Latent Space for Training Set...")
    train_samples_latent_space = model.eval(train_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
    print_ln("Computing Latent Space for Audit Trigger...")
    audit_trigger_samples_latent_space = model.eval(audit_trigger_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)

    train_samples_latent_space = MultiArray([len(train_samples), expected_latent_space_size], sfix)
    train_samples_latent_space.assign_all(sfix(1))
    audit_trigger_samples_latent_space = MultiArray([len(audit_trigger_samples), expected_latent_space_size], sfix)
    audit_trigger_samples_latent_space.assign_all(sfix(1))

    lib.stop_timer(105)

    print(audit_trigger_samples_latent_space.sizes, train_samples_latent_space.sizes)
    print_ln("Computing L2 distance...")
    L2 = audit_utils.euclidean_dist_dot_product(A=train_samples_latent_space, B=audit_trigger_samples_latent_space, n_threads=config.n_threads)
    L2 = L2.transpose()
    assert L2.sizes == (len(audit_trigger_samples), len(train_samples)), f"L2 {L2.sizes}"


    n_audit_trigger_samples = len(audit_trigger_samples)

    if debug:
        knn_sample_id = MultiArray([n_audit_trigger_samples, config.K], sint)
    else:
        knn_sample_id = None


    knn_shapley_values = MultiArray([n_audit_trigger_samples, n_train_samples], sfix)
    knn_shapley_values.assign_all(-1)

    print_ln("Running knnshapley...")
    complex_division_at_compile_time = Array(n_train_samples, cfix)
    for i in range(1, n_train_samples):
        complex_division_at_compile_time[i] = min(config.K, i) / (float(i) * config.K)

    @lib.for_range_opt(n_audit_trigger_samples)
    def knn(audit_trigger_idx):
        print_ln("  audit_trigger_idx=%s", audit_trigger_idx)

        audit_trigger_label = _audit_trigger_mislabels[audit_trigger_idx]
        audit_trigger_label_idx = ml.argmax(audit_trigger_label)

        print(L2)
        print("L2")
        dist_arr = L2[audit_trigger_idx]

        print(_train_labels, train_samples_idx, dist_arr)

        data = concatenate([dist_arr, train_samples_idx, _train_labels_idx], axis=1)
        lib.start_timer(timer_id=101)
        if debug:
            print_ln("Before sort: %s", data.get_part(0, 10).reveal_nested())
        data.sort()
        lib.stop_timer(timer_id=101)

        assert data.sizes == (n_train_samples, 3), f"top k {data.sizes}"

        # TODO: Optimize comparison ?
        print(audit_trigger_idx, n_train_samples, knn_shapley_values.sizes, data.sizes)
        # TODO: uses idx instead of array
        knn_shapley_values[audit_trigger_idx][n_train_samples - 1] = \
            sfix(data[n_train_samples - 1][2] == audit_trigger_label_idx) / n_audit_trigger_samples

        lib.start_timer(timer_id=102)
        lib.start_timer(timer_id=103)

        precomputed_equality_array = Array(n_train_samples, sfix)
        @lib.for_range_opt(n_train_samples)
        def _(i):
            precomputed_equality_array[i] = sfix(data[i][2] == audit_trigger_label_idx)
        lib.stop_timer(timer_id=103)

        @lib.for_range_opt(n_train_samples - 1, 1, -1)
        def _(iplusone):
            i = iplusone - 1
            # TODO: Optimize comparison?
            complex_part_one = precomputed_equality_array[i] - precomputed_equality_array[iplusone]
            compile_time_part_two = complex_division_at_compile_time[iplusone]

            knn_shapley_values[audit_trigger_idx][i] = knn_shapley_values[audit_trigger_idx][iplusone] + (complex_part_one * compile_time_part_two)

        lib.stop_timer(timer_id=102)


    result = {"shapley_values": knn_shapley_values}
    debug_output = {}



    if debug:
        pass

    return result, debug_output

def build_subsets_order(num_samples, num_features):
    z_coalitions = np.zeros((num_samples, num_features), dtype=np.int64)

    num_subset_sizes = int(np.ceil((num_features - 1) / 2.0))
    num_paired_subset_sizes = int(np.floor((num_features - 1) / 2.0))

    weight_vector = np.array([(num_features - 1.0) / (i * (num_features - i)) for i in range(1, num_subset_sizes + 1)])

    weight_vector[:num_paired_subset_sizes] *= 2
    weight_vector /= np.sum(weight_vector)

    # print(f"weight_vector = {weight_vector}")
    print(f"num_subset_sizes = {num_subset_sizes}")
    print(f"num_paired_subset_sizes = {num_paired_subset_sizes}")
    print(f"num_features = {num_features}")



    num_full_subsets = 0
    num_samples_left = num_samples

    group_inds = np.arange(num_features, dtype='int64')
    mask = np.zeros(num_features, dtype=np.int64)
    remaining_weight_vector = copy.copy(weight_vector)

    n_samples_added = 0
    kernelWeights = np.zeros(num_samples)

    for subset_size in range(1, num_subset_sizes + 1):

        nsubsets = binom(num_features, subset_size)
        if subset_size <= num_paired_subset_sizes:
            nsubsets *= 2
        print(f"subset_size = {subset_size}")
        print(f"nsubsets = {nsubsets}")
        print("self.nsamples*weight_vector[subset_size-1] = {}".format(
            num_samples_left * remaining_weight_vector[subset_size - 1]))
        print("self.nsamples*weight_vector[subset_size-1]/nsubsets = {}".format(
            num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

        if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
            num_full_subsets += 1
            num_samples_left -= nsubsets

            if remaining_weight_vector[subset_size - 1] < 1.0:
                remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

            w = weight_vector[subset_size - 1] / binom(num_features, subset_size)
            if subset_size <= num_paired_subset_sizes:
                w /= 2.0
            for inds in itertools.combinations(group_inds, subset_size):
                mask[:] = 0
                mask[np.array(inds, dtype='int64')] = 1
                z_coalitions[n_samples_added, :] = mask
                kernelWeights[n_samples_added] = w
                n_samples_added += 1
                if subset_size <= num_paired_subset_sizes:
                    # TODO: Not sure what this does but i guess its an optimization?
                    mask[:] = np.abs(mask - 1)
                    z_coalitions[n_samples_added, :] = mask
                    kernelWeights[n_samples_added] = w
                    n_samples_added += 1
        else:
            break
    print(f"num_full_subsets = {num_full_subsets}")

    nfixed_samples = n_samples_added
    samples_left = num_samples - n_samples_added
    print(f"samples_left = {samples_left}")
    if num_full_subsets != num_subset_sizes:
        remaining_weight_vector = copy.copy(weight_vector)
        remaining_weight_vector[:num_paired_subset_sizes] /= 2
        remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
        remaining_weight_vector /= np.sum(remaining_weight_vector)
        print(f"remaining_weight_vector = {remaining_weight_vector}")
        print(f"num_paired_subset_sizes = {num_paired_subset_sizes}")
        np.random.seed(42)
        ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
        ind_set_pos = 0
        used_masks = {}
        while samples_left > 0 and ind_set_pos < len(ind_set):
            mask.fill(0)
            ind = ind_set[ind_set_pos] #
            ind_set_pos += 1
            subset_size = ind + num_full_subsets + 1
            mask[np.random.permutation(num_features)[:subset_size]] = 1

            mask_tuple = tuple(mask)
            new_sample = False
            if mask_tuple not in used_masks:
                new_sample = True
                used_masks[mask_tuple] = n_samples_added
                samples_left -= 1
                z_coalitions[n_samples_added, :] = mask
                kernelWeights[n_samples_added] = 1.0
                n_samples_added += 1
            else:
                kernelWeights[used_masks[mask_tuple]] += 1.0

            if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                mask[:] = np.abs(mask - 1)

                if new_sample:
                    samples_left -= 1
                    z_coalitions[n_samples_added, :] = mask
                    kernelWeights[n_samples_added] = 1.0
                    n_samples_added += 1
                else:
                    kernelWeights[used_masks[mask_tuple] + 1] += 1.0

        weight_left = np.sum(weight_vector[num_full_subsets:])
        print(f"weight_left = {weight_left}")
        print(f"kernelWeights {kernelWeights}")
        kernelWeights[nfixed_samples:] *= weight_left / kernelWeights[nfixed_samples:].sum()

        return z_coalitions, kernelWeights

def invert_compile_time(z_coalitions, kernelWeights):
    etmp = z_coalitions
    print(f"etmp[:4,:] {etmp[:4, :]}")
    print(f"kernelWeights {kernelWeights}")

    tmp = np.transpose(np.transpose(etmp) * np.transpose(kernelWeights))
    print(f"tmp {tmp}")
    etmp_dot = np.dot(np.transpose(tmp), etmp)
    print(f"etmp_dot {etmp_dot}")
    try:
        tmp2 = np.linalg.inv(etmp_dot)
    except np.linalg.LinAlgError:
        tmp2 = np.linalg.pinv(etmp_dot)
    return tmp2

