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
    
    source = [  "SS_bjornhox_04-07-18_14:32_MR_W2V_sim_0.0_e3cf", 
                "SS_bjornhox_06-07-18_12:26_MR_w2v_0.54_681f", 
                "SS_bjornhox_06-07-18_12:26_MR_w2v_0.56_8147", 
                "SS_bjornhox_06-07-18_13:01_MR_w2v_0.58_8753", 
                "SS_bjornhox_03-07-18_14:22_MR_UMICH_BASELINE_cb11"]
   
  
    path = './results/'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    test_acc_avg_full = []
    
    
    legend = ["0.0", "0.54", "0.56", "0.58", "random"]
   

    for i in range(0, len(source)):        
        env = source[i]
        res = download_env(env)
        test_acc_avg_full.append(res)

    plt.figure(1)
    plt.axis([0,250,50,80])
    plt.subplot(111)

    plt.xlabel("Amount of labeled data")
    plt.ylabel("% accuracy")

    for line in test_acc_avg_full:        
        line[0].insert(0,0)
        line[1].insert(0,50)
        # plt.plot(*line) 

    
    plt.plot(*test_acc_avg_full[0], color='#ff7f0e') #
    plt.plot(*test_acc_avg_full[1], dashes=[4, 2], color='#9467bd') #
    plt.plot(*test_acc_avg_full[2], color='#1f77b4') #
    plt.plot(*test_acc_avg_full[3], dashes=[6, 2], color='#17becf')#
    plt.plot(*test_acc_avg_full[4], color='#2ca02c')#



    plt.legend(legend,
           loc='lower right')
    plt.savefig('results/W2V/W2V_MR.png' , dpi=600)
    # plt.show()
