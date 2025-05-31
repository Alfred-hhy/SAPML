
from Compiler.types import MultiArray, Array, sfix, cfix, sint, cint, regint, MemValue
from Compiler.dijkstra import HeapQ
from Compiler.oram import OptimalORAM
from Compiler.permutation import cond_swap

from Compiler import library as lib
from Compiler.library import print_ln
from Compiler import util
from Compiler import types

from Compiler.script_utils import audit_function_utils as audit_utils

import ml


def audit(input_loader, config, debug: bool):

    train_samples, _train_labels = input_loader.train_dataset()

    if config.n_batches > 0:
        print("Approximating with cfg.n_batches")
        train_samples = train_samples.get_part(0, config.n_batches * config.batch_size)
        _train_labels = _train_labels.get_part(0, config.n_batches * config.batch_size)
        print("Running on", len(train_samples), "samples")

    n_train_samples = len(train_samples)


    lib.start_timer(104)
    _train_labels_idx = Array(len(train_samples), sint)
    @lib.for_range_opt_multithread(config.n_threads, len(train_samples))
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

    lib.start_timer(105)
    print_ln("Computing Latent Space for Training Set...")

    model.layers[-1].compute_loss = False

    if config.batch_size == 2 and config.n_batches == 1:
        print("Skipping second forward pass")
        train_samples_latent_space = MultiArray([len(train_samples), expected_latent_space_size], sfix)
    else:
        train_samples_latent_space = model.eval(train_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
        assert train_samples_latent_space.sizes == (len(train_samples), expected_latent_space_size), f"{train_samples_latent_space.sizes} != {(len(train_samples), expected_latent_space_size)}"

    print_ln("Computing Latent Space for Audit Trigger...")
    audit_trigger_samples_latent_space = model.eval(audit_trigger_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
    assert  audit_trigger_samples_latent_space.sizes == (len(audit_trigger_samples), expected_latent_space_size), f"{audit_trigger_samples_latent_space.sizes} != {(len(audit_trigger_samples), expected_latent_space_size)}"

    lib.stop_timer(105)

    print(audit_trigger_samples_latent_space.sizes, train_samples_latent_space.total_size())
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
        print("DATA SHAPE", data.sizes)
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
            # PRECOMP
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
    debug_output =  {"shapley_values": knn_shapley_values}

    if debug:
        pass

    return result, debug_output

def knn_naive(distance_array, idx_array, k):
    # O(nk * 3) swaps
    n = len(distance_array)
    k_nn = Array(k, sint)
    k_nn.assign(sint(10000))

    print(k, idx_array.value_type)




    @lib.for_range(n)
    def _(i):
        val = distance_array[i]
        val_idx = idx_array[i]
        val_was_placed = MemValue(sint(0))
        @lib.for_range(k)
        def _(j):
            val_is_lower = k_nn[j] > val
            print(val_is_lower, val_was_placed, j)
            val_should_be_placed = val_is_lower.bit_and(val_was_placed == 0)

            place_val = val_should_be_placed * val_idx + ((sint(1) - val_should_be_placed) * k_nn[j])
            print(place_val, val_should_be_placed, val, k_nn.reveal_nested())
            k_nn[j] = place_val

            val_was_placed.write(val_was_placed.read().bit_or(val_should_be_placed))

    print_ln("k_nn %s", k_nn.reveal_nested())
    return k_nn


def pre_sort_sort(array, input_loader):
    for party_id in range(input_loader.num_parties()):
        start, size = input_loader.train_dataset_region(party_id)
        print(array, start, size)
        distance_party = array.get_part(start=start, size=start+size).reveal_nested()
        print(len(distance_party), len(distance_party[0]))

        sortedx = lib.sort([x[0] for x in distance_party])
        print(sortedx)



def concatenate(arrays, axis):


    if axis != 1:
        raise ValueError("not implemented yet")

    n_rows = arrays[0].length
    n_cols = len(arrays)
    out = MultiArray([n_rows, n_cols], arrays[0].value_type)

    for a in arrays:
        assert a.length == arrays[0].length

    @lib.for_range(n_rows)
    def _(i):
        for j, a in enumerate(arrays):
            out[i][j] = a[i]

    return out