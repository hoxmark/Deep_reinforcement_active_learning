from pprint import pprint
from visdom import Visdom
import pathlib
import json
import sys
import matplotlib.pyplot as plt

def download_env(env):
    vis = Visdom('http://logserver.duckdns.org', port=5010)
    data = vis.get_window_data(env=env)
    d = json.loads(data)
    
    n_deleted = []
    
    for key in d:
        try:
            #1 for MR 0 for UMICH
            x = list(d[key]["content"]["data"][1]["x"])
            y = list(d[key]["content"]["data"][1]["y"])            
            if 'n-deleted' in key:
                n_deleted = (x,y)
            print(key)
        except:
            pass


    return n_deleted

if __name__ == "__main__":
    # print(len(sys.argv))
    source = [  "SS_bjornhox_03-07-18_11:37_MR_cnn_sim_0.10_8a3f", 
                "SS_bjornhox_03-07-18_13:20_MR_cnn_sim_0.12_ef2b",
                "SS_bjornhox_03-07-18_11:50_MR_cnn_sim_0.15_b7f9"]
    
    legden = ["0.10", "0.12", "0.15"]

    path = './results/'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    test_acc_avg_full = []
    
     
    # for i in sys.argv[1:]:
        # legden.append(i.split("_")[7])
        # legden.append(i.split("_")[6])
        # legden.append(i.split("_")[8])

    for i in range(0, len(source)):        
        env = source[i]
        res = download_env(env)
        test_acc_avg_full.append(res)

    plt.figure(1)
    # plt.axis([0,250,50,80])
    plt.subplot(111)

    for line in test_acc_avg_full:        
        line[0].insert(0,0)
        line[1].insert(0,50)
        plt.plot(*line) 

    plt.legend(legden,
           loc='upper right')
    plt.savefig('results/CNN_MR_N_DEL.png' , dpi=600)
    plt.show()
