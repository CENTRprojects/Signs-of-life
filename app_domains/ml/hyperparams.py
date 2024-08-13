"""Script handling hyperparameters optimization of the ML models"""
import gc
import time
import sys
import os

pth = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(pth)
sys.path.append(pth)

from ml.config_train import CONFIG_TR
from ml.main_train import train

if __name__ == '__main__':

    tic = time.time()
    print(time.strftime("%Y_%m_%d-%H%M"))

    # Grid Search on following Explored parameters
    # # # scenario
    # n_esti = [20, 50, 100, 200]
    # lrs = [3e-1, 1e-1, 3e-2, 1e-2]
    # depths = [4, 5, 6, 9]

    min_child = [1, 2]
    col_sample = [0.8, 0.9, 1]
    col_sample2 = [0.8, 0.9, 1]
    lbda = [0.1, 0, 1, 10]

    svc_c = [0.01,0.03, 0.1, 0.3, 1]
    svc_kernel = [ "linear"]
    svc_degree= [3]

    # degree useless if rbf = linear
    scen = [[e, r, k] for e in svc_c for r in svc_kernel for k in svc_degree]
    dupli_lin=set()
    to_remove = set()
    for i in range(len(scen)):
        if scen[1] == "linear":
            clin = str(scen[0]) + str(scen[2])
            if clin in dupli_lin:
                to_remove.add(i)
            else:
                dupli_lin.add(clin)
    print(len(to_remove))
    scen = [e for i, e in enumerate(scen) if (i not in to_remove)]
    prefs = ["opti_svc_{}_{}_{}".format(sc[0], sc[1], sc[2]) for sc in scen]

    for i in list(range(len(scen))):
        CONFIG_TR["scenario"] = prefs[i]
        # CONFIG_TR["n_estimators"] = scen[i][0]
        # CONFIG_TR["lr"] = scen[i][1]
        # CONFIG_TR["max_depth"] = scen[i][2]

        CONFIG_TR["SVC_C"] = scen[i][0]
        CONFIG_TR["SVC_kernel"] = scen[i][1]
        CONFIG_TR["SVC_degree"] = scen[i][2]

        # CONFIG_TR["min_child_weight"] = scen[i][0]
        # CONFIG_TR["colsample_bytree"] = scen[i][1]
        # CONFIG_TR["colsample_bylevel"] = scen[i][2]
        # CONFIG_TR["reg_lambda"] = scen[i][3]

        train()

        gc.collect()

    print(time.strftime("%Y%m%d-%H%M"))
    print("Duration :{}".format(round(time.time() - tic)))
