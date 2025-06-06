from typing import List, Any, Union

from Compiler.types import MultiArray, Array
from Compiler.types import sfix, cfix, MemValue, cint, regint
from Compiler import ml
from Compiler import library as lib

from Compiler.library import print_ln
from Compiler.script_utils import audit_function_utils as audit_utils



def audit(input_loader, config, debug: bool):

    model = input_loader.model()

    if debug:

        n_test_samples = min(input_loader.test_dataset_size(), 300)

        print_ln("Checking Test Set Accuracy of Initial Model...")
        test_samples, test_labels = input_loader.test_dataset()
        test_samples = test_samples.get_part(0, n_test_samples)
        test_labels = test_labels.get_part(0, n_test_samples)
        n_correct, avg_loss = model.reveal_correctness(data=test_samples, truth=test_labels, batch_size=input_loader.batch_size(), running=True)
        print_ln("  n_correct=%s  n_samples=%s  avg_loss=%s", n_correct, len(test_samples), avg_loss)

    train_samples, train_labels = input_loader.train_dataset()

    audit_trigger_samples, audit_trigger_mislabels = input_loader.audit_trigger()

    batch_size = input_loader.batch_size()
    n_audit_triggers = input_loader.audit_trigger_size()

    n_data_owners = input_loader.num_parties()

    audit_trigger_batch_size = min(batch_size, n_audit_triggers)

    backup_variables = backup_trainable_vars(model)

    train_optimizer = model
    train_optimizer.n_epochs = config.n_unlearn_epochs
    train_optimizer.report_losses = True
    train_optimizer.print_losses = True
    train_optimizer.gamma = MemValue(cfix(config.learning_rate))

    if isinstance(train_labels, Array):
        n_classes = 2
        unlearned_predictions = MultiArray([n_data_owners, n_audit_triggers, 2], sfix)
    else:
        n_classes = train_labels.sizes[1]
        unlearned_predictions = MultiArray([n_data_owners, n_audit_triggers, n_classes], sfix)

    null_label = sfix(1.0 / n_classes)

    starts = Array.create_from([regint(input_loader.train_dataset_region(id)[0]) for id in range(n_data_owners)])
    n_audit_triggers_per_party = Array.create_from([regint(input_loader.train_dataset_region(id)[1]) for id in range(n_data_owners)])

    @lib.for_range_opt(n_data_owners)
    def unlearn_party(data_owner_id):

        print_ln("Unlearning data from owner %s", data_owner_id)

        if n_classes == 2:
            train_labels_copy = Array(train_labels.length, sfix)
            train_labels_copy.assign(train_labels)
        else:
            train_labels_copy = MultiArray(train_labels.sizes, sfix)
            train_labels_copy.assign(train_labels)

        start = starts[data_owner_id]
        n_sample = n_audit_triggers_per_party[data_owner_id]

        train_labels_copy.get_part(start, n_sample).assign_all(null_label)

        @lib.if_(data_owner_id > 0)
        def _():
            for i, (model_var, initial_var) in  enumerate(zip(model.trainable_variables, backup_variables)):
                print_ln("  restoring trainable_variable %s", i)

                model_var.assign(initial_var)


        ml.Layer.back_batch_size = batch_size


        train_optimizer.layers[-1].Y.address = train_labels_copy.address
        train_optimizer.layers[0].X.address = train_samples.address

        train_optimizer.layers[-1].compute_loss = False
        train_optimizer.run(batch_size=batch_size)

        guesses = audit_utils.predict_on_model_copy(input_loader.model_layers(), model, audit_trigger_batch_size,
                                                            audit_trigger_samples, audit_trigger_mislabels)

        if debug:
            audit_utils.reveal_accuracy(preds=guesses, labels=audit_trigger_mislabels)

        unlearned_predictions[data_owner_id] = guesses

    print_ln("Computing losses...")
    loss_matrix = audit_utils.compute_loss(unlearned_predictions, audit_trigger_mislabels, n_classes=n_classes)

    assert loss_matrix.sizes == (n_audit_triggers, n_data_owners), f"loss_array={loss_matrix.sizes}"

    # Compute MAD Scores
    print_ln("Computing MAD scores...")
    threshold_matrix, mad_score_matrix = audit_utils.comp_mod_zscore_threshold_matrix(matrix=loss_matrix, threshold=config.mod_zscore_threshold, debug=debug)



    result = {"threshold_matrix": threshold_matrix}

    if debug:
        result["loss_matrix"] = loss_matrix,
        result["mad_score_matrix"] = mad_score_matrix


    return result


def backup_trainable_vars(model):
    backup: List[Union[MultiArray, Array]] = list()

    for i, var in enumerate(model.trainable_variables):

        copy_container = var.same_shape()
        copy_container.assign(var)
        backup.append(copy_container)
    return backup
