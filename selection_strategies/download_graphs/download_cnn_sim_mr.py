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
    
    test_acc_avg = []
    
    for key in d:
        try:
            #1 for MR 0 for UMICH
            x = list(d[key]["content"]["data"][1]["x"])
            y = list(d[key]["content"]["data"][1]["y"])            
            if 'test-acc-avg' in key:
                test_acc_avg = (x,y)

        except:
            pass


    return test_acc_avg

if __name__ == "__main__":
    # print(len(sys.argv))
    source = [  "SS_bjornhox_03-07-18_13:20_MR_cnn_sim_0.12_ef2b", 
                "SS_bjornhox_04-07-18_09:21_MR_self_cnn_sim_0.0_d0ca",  
                "SS_bjornhox_03-07-18_14:22_MR_UMICH_BASELINE_cb11"]

    path = './results/'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    test_acc_avg_full = []
    
    legden = [] 
    legden = ["0.12", "0.0", "random"]
    # for i in sys.argv[1:]:
        # legden.append(i.split("_")[7])
        # legden.append(i.split("_")[6])
        # legden.append(i.split("_")[8])

    for i in range(0, len(source)):        
        env = source[i]
        res = download_env(env)
        test_acc_avg_full.append(res)

    plt.figure(1)
    plt.axis([0,250,50,80])
    plt.subplot(111)

    for line in test_acc_avg_full:        
        line[0].insert(0,0)
        line[1].insert(0,50)
        plt.plot(*line) 

    plt.legend(legden,
           loc='upper right')
    plt.savefig('results/CNN_MR.png' , dpi=600)
    # plt.show()
