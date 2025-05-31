
from Compiler.types import MultiArray, Array, sfix, sint, cint, regint
from Compiler.dijkstra import HeapQ
from Compiler.oram import OptimalORAM

from Compiler import library as lib
from Compiler.library import print_ln
from Compiler import util
from Compiler import types

from Compiler.script_utils import audit_function_utils as audit_utils


def audit(input_loader, config, debug: bool):

    train_samples, _train_labels = input_loader.train_dataset()

    train_samples_ownership = Array(len(train_samples), sint)

    for party_id in range(input_loader.num_parties()):
        start, size = input_loader.train_dataset_region(party_id)
        train_samples_ownership.get_sub(start=start, stop=start+size).assign_all(party_id)

    train_samples_idx = Array.create_from(cint(x) for x in range(len(train_samples)))

    audit_trigger_samples, _audit_trigger_mislabels = input_loader.audit_trigger()



    model = input_loader.model()
    latent_space_layer, expected_latent_space_size = input_loader.model_latent_space_layer()

    print_ln("Computing Latent Space for Training Set...")
    train_samples_latent_space = model.eval(train_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
    assert train_samples_latent_space.sizes == (len(train_samples), expected_latent_space_size)

    print_ln("Computing Latent Space for Audit Trigger...")
    audit_trigger_samples_latent_space = model.eval(audit_trigger_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
    assert  audit_trigger_samples_latent_space.sizes == (len(audit_trigger_samples), expected_latent_space_size)


    print_ln("Computing L2 distance...")
    L2 = audit_utils.euclidean_dist_dot_product(A=train_samples_latent_space, B=audit_trigger_samples_latent_space, n_threads=config.n_threads)
    L2 = L2.transpose()
    assert L2.sizes == (len(audit_trigger_samples), len(train_samples)), f"L2 {L2.sizes}"


    n_audit_trigger_samples = len(audit_trigger_samples)

    if debug:
        knn_sample_id = MultiArray([n_audit_trigger_samples, config.K], sint)
    else:
        knn_sample_id = None

    knn_party_count = MultiArray([n_audit_trigger_samples, input_loader.num_parties()], sfix)
    knn_party_count.assign_all(0)

    print_ln("Running knn...")

    @lib.for_range_opt(n_audit_trigger_samples)
    def knn(audit_trigger_idx):
        print_ln("  audit_trigger_idx=%s", audit_trigger_idx)

        dist_arr = L2[audit_trigger_idx]

        data = concatenate([dist_arr, train_samples_idx, train_samples_ownership], axis=1)

        data.sort()
        data = data.get_part(start=0, size=config.K)
        assert data.sizes == (config.K, 3), f"top k {data.sizes}"

        @lib.for_range_opt(config.K)
        def _(k):

            @lib.for_range_opt(input_loader.num_parties())
            def _(party_id):

                sel_party_id = data[k][2]

                comp = party_id == sel_party_id
                update = util.cond_swap(comp, 0, 1)[0]

                knn_party_count[audit_trigger_idx][party_id] += update

            if debug:
                sel_sample_id = data[k][1]
                knn_sample_id[audit_trigger_idx][k] = sel_sample_id

    assert knn_party_count.sizes == (n_audit_trigger_samples, input_loader.num_parties()), f"knn_party_count={knn_party_count.sizes}"

    print_ln("Computing MAD scores...")
    threshold_matrix, mad_score_matrix = audit_utils.comp_mod_zscore_threshold_matrix(matrix=knn_party_count, threshold=config.mod_zscore_threshold, debug=debug)

    result = {"threshold_matrix": threshold_matrix}

    if debug:
        result["knn_party_count"] = knn_party_count
        result["knn_sample_id"] = knn_sample_id
        result["mad_score_matrix"] = mad_score_matrix
        result["l2_distances"] = L2.get_part(0, 3)

    return result


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