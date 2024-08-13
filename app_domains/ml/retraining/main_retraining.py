""" Main script that runs the ML model retraining pipeline"""
import copy
from openpyxl import Workbook

from ml.config_train import CFG_TR
from ml.performance import assess_performance
from ml.retraining.imports import *
from ml.retraining.config_retrain import *
from ml.retraining.ml_trainer import Trainer, get_most_common_label, get_list_prevailing_labels
from ml.retraining.out_excel import exc_handler_with_wb, DICO_FUNC_TO_NAMES, write_in_excel_df, add_ws_log, \
    add_ws_labeller, add_ws_performance, add_ws_config, add_ws_error, add_ws_history, add_ws_detail
from ml.retraining.utils import gtm, print_all_done, is_not_trivial_init2k_sample


def load_input_data():
    """Load samples data"""
    success = True
    err_mess = ""
    dico_data = {}

    # hisorical performances
    fp_historical_perfs = join(MAIN_DIR, HISTORICAL_PERFORMANCE)
    if os.path.isfile(fp_historical_perfs):
        with open(fp_historical_perfs, "r", encoding='utf-8') as f:
            historical_perfs = json.loads(f.read().encode('raw_unicode_escape').decode())

        dico_data["historical_perfs"] = historical_perfs
    else:
        mess_no_histo = "No historical perfs file found --> start new file"
        LG.add({"time": gtm(), "importance": "INFO", "file": "historical_performance", "category": "Loading error",
                "description": mess_no_histo})

    # initial 2k data
    list_init2k = []
    if TRAIN_AT_LV2:
        try:
            # load
            with open(join(MAIN_DIR, INIT_2K_SAMPLES), "r", encoding='utf-8') as f:
                list_init2k = json.loads(f.read().encode('raw_unicode_escape').decode())

            # convert
            for i in range(len(list_init2k)):
                list_init2k[i]["html_text"] = list_init2k[i].pop("clean_text")

        except Exception as e:
            success = False
            cur_err_mess = f"Error during reading of initial 2K domains file: {type(e)} - {str(e)}"
            LG.add({"time": gtm(), "importance": "CRITICAL", "file": "initial 2k", "category": "Reading error",
                    "description": cur_err_mess})
            err_mess += cur_err_mess
            return dico_data, success, err_mess

    # new samples
    json_current = os.listdir(join(MAIN_DIR, SAMPLES_FOLDER))
    list_current = []
    for elem_name in json_current:
        try:
            with open(join(MAIN_DIR, SAMPLES_FOLDER, elem_name), "r", encoding='utf-8') as f:
                # elem = json.loads(f.read().encode('raw_unicode_escape').decode())
                elem = json.loads(f.read())
        except Exception as e:
            success = False
            cur_err_mess = f"Error during reading of new samples: {type(e)} - {str(e)}"
            err_mess += cur_err_mess
            LG.add({"time": gtm(), "importance": "CRITICAL", "file": "new samples", "category": "Reading error",
                    "description": cur_err_mess})
            return dico_data, success, err_mess

        if isinstance(elem, list):
            print("QTY Samples from {}: {}".format(elem_name, len(elem)))
            list_current += elem
        elif isinstance(elem, dict):
            # categorisation flatten
            list_dico_resu = elem["results"]
            for i in range(len(list_dico_resu)):
                list_dico_resu[i].update(list_dico_resu[i]["categorisation"])
                list_dico_resu[i].pop("categorisation")
            list_current += list_dico_resu
        else:
            raise ValueError("Unexpected sample file formatting")
        a = 1

    # format labtool samples
    for i in range(len(list_current)):
        # user choices reformatting
        # ---labels
        list_current[i]["user_name"] = [e[1] for e in list_current[i]["user_choices"]]
        list_current[i]["user_choices"] = [e[0] for e in list_current[i]["user_choices"]]

        # ---comments
        list_current[i]["user_comments_name"] = [e[1] for e in list_current[i]["user_comments"]]
        list_current[i]["user_comments"] = [e[0] for e in list_current[i]["user_comments"]]

        # screenshot name parsing
        list_current[i]["ss_filename"] = str(list_current[i]["ss_filename"]).split("/")[-1]

    # format init2k samples
    print("init2k bef: {}".format(len(list_init2k)))
    list_init2k = [e for e in list_init2k if is_not_trivial_init2k_sample(e["html_text"])]
    print("init2k aft: {}".format(len(list_init2k)))

    # concat
    list_current += list_init2k
    dico_data["list_samples"] = list_current

    if SAMPLING is not None:
        dico_data["list_samples"] = dico_data["list_samples"][0:SAMPLING]

    return dico_data, success, err_mess


def is_not_registrar(spl):
    """Check if the sample is not a registrar"""
    if "user_choices" not in spl:
        return True
    else:
        classifs = list(set(spl["user_choices"]))
        return (len(classifs) > 1) or (classifs[0] != "parked_notice_registrar")


def is_labelled(spl):
    """check if the sample has a label"""
    return (("user_choices" in spl) and len(spl["user_choices"]) > 0) or (
            ("target" in spl) and (spl["target"] in [0., 1.]))


def has_non_empty_text(spl):
    """check if the sample has an empty HTML page"""
    if isinstance(spl["html_text"], str):
        return len(spl) > 0
    else:
        return False


def is_single_labelled(spl):
    """check if the sample has only one label"""
    return False if ("user_choices" not in spl) else (len(spl["user_choices"]) == 1)


def is_multi_labelled(spl):
    """check if the sample has multiple labels"""
    return False if ("user_choices" not in spl) else (len(spl["user_choices"]) > 1)

def is_unsure(url_obj):
    """check if the sample's label is unsure"""
    # unsure = "unsure" label as prevalent as other labels
    if "user_choices" in url_obj:
        labels = get_list_prevailing_labels(url_obj["user_choices"])
        return "unsure" in labels
    else:
        return False


def is_disagreement(spl):
    """check if the sample causes a disagreement among the different labellers"""
    # initial 2k are all valid
    if "user_choices" not in spl:
        return False

    n_labels = len(spl["user_choices"])
    n_unq_labels = len(list(set(spl["user_choices"])))

    if n_unq_labels == 1:
        return False
    elif n_unq_labels > N_MAX_LABELS:
        return True
    else:
        label_counts = [spl["user_choices"].count(elem) for elem in set(spl["user_choices"])]
        count_dominant_label = max(label_counts)

        return (float(count_dominant_label) / n_labels) < MIN_PREVAILING_THRESH


def is_full_agreement(x):
    """check if the sample causes a full consensus among labellers"""
    all_lbls = x["all_labels"]
    return len(list(set(all_lbls))) == 1


def has_one_agreement(x):
    """check if the sample at least 2 people agreeing"""
    label = x["label"]
    all_lbls = x["all_labels"]
    return all_lbls.count(label) > 1


def analyze_labeller(all_samples):
    """Function that assess each labellers'labelling qualities by comparing his labels to others"""
    labels_details = []
    columns = ["url", "label", "labeller", "comment", "commenter"]

    # extract
    for elem in all_samples:
        if "user_choices" not in elem:
            continue
        # if len(elem["user_choices"]) == 1: # keep multi labels only
        #     continue

        labels = elem["user_choices"]
        labellers = elem["user_name"]
        cmts = elem["user_choices"]
        commenters = elem["user_comments_name"]
        lg = len(labels)
        urls = [elem["url"]] * lg

        one_domain = list(zip(urls, labels, labellers, cmts, commenters))
        labels_details += copy.deepcopy(one_domain)

    df_flat = pd.DataFrame(labels_details, columns=columns)

    # url / labels
    df_urls = df_flat[["url", "label"]].groupby("url", as_index=False).agg({"label": lambda x: list(x)})
    df_urls = df_urls.rename(columns={"label": "all_labels"})
    df_flat = df_flat.merge(df_urls, on="url", how="left")

    # agreement
    df_flat["is_single_labelled"] = df_flat["all_labels"].apply(lambda x: len(x) == 1)
    df_flat["is_full_agreement"] = df_flat[["label", "all_labels"]].apply(is_full_agreement, axis=1)
    df_flat["has_one_agreement"] = df_flat[["label", "all_labels"]].apply(has_one_agreement, axis=1)
    df_flat["count"] = 1

    # single labels
    ind_single_label = df_flat["is_single_labelled"] == True
    df_single_count = df_flat[["labeller", "count"]].groupby("labeller", as_index=False).agg({"count": "sum"})
    df_single_count = df_single_count.rename(columns={"count": "QTY_labels"})
    df_flat = df_flat[~ind_single_label]

    # summarize
    dico_agg = {"is_full_agreement": "sum", "has_one_agreement": "sum", "count": "sum"}
    df_summary = df_flat.groupby("labeller", as_index=False).agg(dico_agg)
    df_summary["ratio_full_agreement"] = df_summary["is_full_agreement"] / df_summary["count"]
    df_summary["ratio_one_agreement"] = df_summary["has_one_agreement"] / df_summary["count"]
    df_summary = df_summary.sort_values(by=["ratio_full_agreement", "ratio_one_agreement"], ascending=True)

    df_summary = df_summary.rename(columns={"count": "QTY_labels_with_multi_lbls"})
    df_summary = df_summary.merge(df_single_count, how="left", on="labeller")

    return df_summary


def analyze_filter_data(dico_data):
    """Assess labelling quality and filter in reliable samples"""
    success = True
    err_mess = ""
    dico_data_train = {}

    all_samples = dico_data["list_samples"]

    # all_choices = set()
    # for elem in dico_data["list_samples"]:
    #     if "user_choices" in elem:
    #         all_choices = all_choices.union(set(elem["user_choices"]))

    lg_bef_lab = len(all_samples)
    all_samples = [e for e in all_samples if is_labelled(e)]
    lg_bef_empty_text = len(all_samples)
    all_samples = [e for e in all_samples if has_non_empty_text(e)]
    lg_no_lab = lg_bef_lab - lg_bef_empty_text
    lg_no_text = lg_bef_empty_text - len(all_samples)
    print(f"INFO: Non labelled samples :{lg_no_lab} Samples with no text :{lg_no_text} ")
    LG.add({"time": gtm(), "importance": "INFO", "file": "", "category": "Qty no label",
            "description": str(lg_no_lab)})
    LG.add({"time": gtm(), "importance": "INFO", "file": "", "category": "Qty no text",
            "description": str(lg_no_text)})

    # remove non ML
    all_samples = [e for e in all_samples if is_not_registrar(e)]

    # agreement matrix
    single_labelled_samples = [e for e in all_samples if is_single_labelled(e)]
    multi_labelled_samples = [e for e in all_samples if is_multi_labelled(e)]
    N_single_label = len(single_labelled_samples)
    N_multi_label = len(multi_labelled_samples)
    multi_agreement = [len(list(set(e["user_choices"]))) == 1 for e in multi_labelled_samples]
    n_disagreement = N_multi_label - sum(multi_agreement)
    if N_multi_label > 0:
        ratio_agreement = float(sum(multi_agreement)) / N_multi_label
    else:
        ratio_agreement = "NA"
    summary_agreement = pd.DataFrame(
        [["N_single_label", N_single_label], ["N_multi_label", N_multi_label], ["Ratio_agreement", ratio_agreement],
         ["N_disagreement", n_disagreement]],
        columns=["metric", "value"])
    dico_data_train["summary_agreement"] = summary_agreement

    # disagreement pyramid
    disagreement = [{"url": e["url"], "user_choices": e["user_choices"], "ss_filename": e["ss_filename"]} for e in
                    multi_labelled_samples]
    for i, dico in enumerate(disagreement):
        dico["n_labels"] = len(list(set(dico["user_choices"])))
        dico["label_counts"] = [(e, dico["user_choices"].count(e)) for e in set(dico["user_choices"])]
        dico["label_counts"] = sorted(dico["label_counts"], key=lambda x: -x[1])
        # dico["ss_filename"] = str(dico["ss_filename"]).split("/")[-1]

    disagreement = sorted(disagreement, key=lambda x: -x["n_labels"])
    dico_data_train["disagreement"] = disagreement

    # top N disagreement
    top_disagreement = copy.deepcopy([e for e in disagreement if (e["n_labels"] > 1)][0:N_TOP_DISAGREEMENT])
    for i, dico in enumerate(top_disagreement):
        dico["label_counts"] = " , ".join(["{}:{}".format(e[0], e[1]) for e in dico["label_counts"]])
        dico.pop("user_choices")
    dico_data_train["top_disagreement"] = pd.DataFrame(top_disagreement)

    # unsure filter
    lg_bef = len(all_samples)
    all_samples = [e for e in all_samples if not is_unsure(e)]
    n_unsure = lg_bef - len(all_samples)
    print(f"n unsure = {n_unsure}")
    LG.add({"time": gtm(), "importance": "INFO", "file": "", "category": "N unsure domains",
            "description": str(n_unsure)})

    # todo: labeller
    labeller_view = analyze_labeller(all_samples)
    dico_data_train["labeller_view"] = labeller_view

    # init 2K only
    # all_samples = [e for e in all_samples if ("target" in e)]
    # print(len(all_samples))

    # disagreement filter
    lg_bef = len(all_samples)
    all_samples = [e for e in all_samples if not is_disagreement(e)]
    n_filtered_disagreement = lg_bef - len(all_samples)
    print(f"n filtered disagreement = {n_filtered_disagreement}")
    LG.add({"time": gtm(), "importance": "INFO", "file": "", "category": "Qty filtered disagreement",
            "description": str(n_filtered_disagreement)})

    LG.add({"time": gtm(), "importance": "INFO", "file": "", "category": "Qty final training samples",
            "description": str(len(all_samples))})

    # add target label
    for i in range(len(all_samples)):
        if "user_choices" in all_samples[i]:
            all_samples[i]["target_label"] = get_most_common_label(all_samples[i]["user_choices"])
        else:
            all_samples[i]["target_label"] = "parked_notice_individual_content"

    dico_data_train["list_samples"] = all_samples

    return dico_data_train, success, err_mess


def summarize_performance(df_pred, df_pred_tr, mdl, X_test, y_test_cv, feats):
    """Assess performance and build performance report"""
    print("INFO:---DO_performance---")
    # PRED_COLS = ["id", "url", "actual", "pred", "filtered_text"]
    y_train = df_pred_tr["actual"]
    y_pred_tr = df_pred_tr["pred"]

    y_test = df_pred["actual"]
    y_pred_te = df_pred["pred"]

    assess_performance(y_train, y_test, y_pred_tr, y_pred_te, mdl, feats, X_test, y_test_cv)

    pass


def train_model(dico_data, out_folder):
    """Train and evaluate perfs of a ML model on dataset in dico_data"""
    success = True
    err_mess = ""
    output_data = {}

    dico_data["list_samples"] = dico_data["list_samples"]

    trainer = Trainer(dico_data["list_samples"])

    trainer.preprocess()

    if DO_HYPERPARAM:
        trainer.optimize_hyperparams(out_folder)

    # training
    df_pred_cv, df_pred_tr = trainer.full_cv_train()

    # performances
    trainer.assess_perfs(df_pred_cv, prefix="CV_", print_only=False)
    trainer.assess_perfs(df_pred_tr, prefix="TR_", print_only=False)

    # Languages and TLDs
    if TRAIN_AT_LV2:
        trainer.assess_tld_lgg_perfs(df_pred_cv, prefix="CV_", print_only=False)

    trainer.error_analysis(df_pred_cv)

    # full training
    if DO_FULL_TR and TRAIN_AT_LV2:
        trainer.init_model()
        trainer.train()
        trainer.save_config(out_folder)

    return trainer.output_data, success, err_mess


def format_report_and_save(output_data, input_data_train, input_data, out_folder, out_fname, ref_time):
    """Format results into output Excel and output database results"""

    # workbook
    wb = Workbook()
    wb.remove(wb.worksheets[0])

    # N-to-N configuration
    # sheet CONFIGURATION
    add_ws_config(wb)

    # Labeller analysis
    add_ws_labeller(wb, input_data_train)

    # performance
    add_ws_performance(wb, output_data)

    # historical performance
    if TRAIN_AT_LV2:
        historical_perfs = add_ws_history(wb, input_data, output_data, ref_time, out_folder)

    # error analysis
    add_ws_error(wb, output_data)

    # detail data
    add_ws_detail(wb, output_data)

    # log
    add_ws_log(wb)
    wb.move_sheet(DICO_FUNC_TO_NAMES["add_ws_log"], -(len(wb.sheetnames) - 1))

    # save excel
    wb.save(join(out_folder, out_fname))

    # save history
    if SAVE_HIST and TRAIN_AT_LV2:
        fp_historical_perfs = join(MAIN_DIR, HISTORICAL_PERFORMANCE)
        with open(fp_historical_perfs, "w") as f:
            json.dump(historical_perfs, f, indent=3)


def main_retrain():
    """
    Main function to retrain a model
    Uses samples in retraining/input/samples : update with latest labtool data before running
    Ouptut: save all results in out/retraining_{REFTIME} folder
    Replace xgb_v1 in 'input' folder to integrate the ML model into the main crawler
    """
    input_data = {}
    input_data_train = {}
    output_data = {}
    aborted = False

    # ref
    ref_time = gtm()
    out_folder = join("out", OUT_FOLDER_NAME.format(ref_time))
    out_fname = OUT_EXCEL_REPORT_NAME.format(ref_time)
    os.mkdir(out_folder)
    CFG_TR.set("dir_output", out_folder)

    # load
    LG.add({"time": gtm(), "importance": "INFO", "category": "load_input_data", "description": "start"})
    add_to_log("STAGE : load_input_data")
    input_data, success, err_msg = load_input_data()
    if not success:
        add_to_log("RUN ABORTED with data loading error:\n{}".format(err_msg))
        aborted = True
    LG.add({"time": gtm(), "importance": "INFO", "category": "load_input_data", "description": "end"})

    if not aborted:
        LG.add(
            {"time": gtm(), "importance": "INFO", "category": "samples analysis and filtering", "description": "start"})
        add_to_log("STAGE : samples analysis and filtering")
        input_data_train, success, err_msg = analyze_filter_data(input_data)
        if not success:
            add_to_log("RUN ABORTED with sample analysis error:\n{}".format(err_msg))
            aborted = True
        LG.add(
            {"time": gtm(), "importance": "INFO", "category": "samples analysis and filtering", "description": "end"})

    if not aborted:
        LG.add({"time": gtm(), "importance": "INFO", "category": "train", "description": "start"})
        add_to_log("STAGE : training")
        output_data, success, err_msg = train_model(input_data_train, out_folder)
        if not success:
            add_to_log("RUN ABORTED with training error:\n{}".format(err_msg))
            aborted = True
        LG.add({"time": gtm(), "importance": "INFO", "category": "train", "description": "end"})

    LG.add({"time": gtm(), "importance": "INFO", "category": "formatting", "description": "start"})
    add_to_log("STAGE : formatting")
    format_report_and_save(output_data, input_data_train, input_data, out_folder, out_fname, ref_time)

    if aborted:
        add_to_log("ERROR OCCURED --> cf output file")
    else:
        add_to_log("SUCCESSFUL RUN")
    print_all_done()


if __name__ == "__main__":
    main_retrain()
    # list_scenarios = get_scenarios(OPTI_XGB)
    # a = 1
