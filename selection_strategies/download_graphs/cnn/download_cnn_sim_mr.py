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
    # source = [  "SS_bjornhox_04-07-18_09:21_MR_self_cnn_sim_0.0_d0ca",  
    #             "SS_bjornhox_11-07-18_11:30_MR_cnn_sim_0.08_3b54",
    #             "SS_bjornhox_03-07-18_14:22_MR_UMICH_BASELINE_cb11"]
   
    source = [  "SS_bjornhox_04-07-18_09:21_MR_self_cnn_sim_0.0_d0ca",  
                "SS_bjornhox_11-07-18_11:29_MR_cnn_sim_0.05_c907", 
                "SS_bjornhox_11-07-18_11:30_MR_cnn_sim_0.08_3b54",
                "SS_bjornhox_10-07-18_10:44_MR_cnn_sim_0.12_896d",
                "SS_bjornhox_03-07-18_14:22_MR_UMICH_BASELINE_cb11"]
   
                # "SS_bjornhox_11-07-18_14:20_MR_cnn_sim_0.11_9b40",
    # source = [  "SS_bjornhox_04-07-18_09:21_MR_self_cnn_sim_0.0_d0ca",  
    #             "SS_bjornhox_03-07-18_11:37_MR_cnn_sim_0.10_8a3f", 
    #             "SS_bjornhox_03-07-18_13:20_MR_cnn_sim_0.12_ef2b",
    #             "SS_bjornhox_03-07-18_11:50_MR_cnn_sim_0.15_b7f9",
    #             "SS_bjornhox_03-07-18_14:22_MR_UMICH_BASELINE_cb11"]

    path = './results/'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    test_acc_avg_full = []
    
    legden = [] 
    # legden = ["0.0", "0.10", "0.12", "0.15", "random"]
    legden = ["0.0", "0.05", "0.08", "0.12", "random"]
    # legden = ["0.0", "0.08", "random"]
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
        # plt.plot(*line) 

    
    plt.plot(*test_acc_avg_full[0], color='#ff7f0e') #
    plt.plot(*test_acc_avg_full[1], dashes=[4, 2], color='#9467bd') #
    plt.plot(*test_acc_avg_full[2], color='#1f77b4') #
    plt.plot(*test_acc_avg_full[3], dashes=[6, 2], color='#17becf')#
    plt.plot(*test_acc_avg_full[4], color='#2ca02c')#


    # plt.plot(*test_acc_avg_full[0], color='#ff7f0e') #
    # # plt.plot(*test_acc_avg_full[1], dashes=[4, 2], color='#9467bd') #
    # plt.plot(*test_acc_avg_full[1], color='#1f77b4') #
    # # plt.plot(*test_acc_avg_full[3], dashes=[6, 2], color='#17becf')#
    # plt.plot(*test_acc_avg_full[2], color='#2ca02c')#

    plt.legend(legden,
           loc='lower right')
    plt.savefig('results/CNN_MR.png' , dpi=600)
    # plt.show()
