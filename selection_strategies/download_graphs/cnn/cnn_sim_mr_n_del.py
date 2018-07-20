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
    test_acc_avg = []
    
    for key in d:
        try:
            #1 for MR 0 for UMICH
            x = list(d[key]["content"]["data"][1]["x"])
            y = list(d[key]["content"]["data"][1]["y"])            
            if 'n-deleted' in key:
                n_deleted = (x,y)

            #1 for MR 0 for UMICH
            x = list(d[key]["content"]["data"][1]["x"])
            y = list(d[key]["content"]["data"][1]["y"])            
            if 'test-acc-avg' in key:
                test_acc_avg = (x,y)
        except:
            pass


    return n_deleted, test_acc_avg

if __name__ == "__main__":
    # print(len(sys.argv))
    source = [  "SS_bjornhox_11-07-18_11:29_MR_cnn_sim_0.05_c907", 
                "SS_bjornhox_11-07-18_11:30_MR_cnn_sim_0.08_3b54",
                "SS_bjornhox_10-07-18_10:44_MR_cnn_sim_0.12_896d"]
    
    
    legden = ["0.05", "0.08", "0.12"]

    path = './results/'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    n_deleted = []
    test_acc_avg = []
    
     
    # for i in sys.argv[1:]:
        # legden.append(i.split("_")[7])
        # legden.append(i.split("_")[6])
        # legden.append(i.split("_")[8])

    for i in range(0, len(source)):        
        env = source[i]
        res1, res2 = download_env(env)
        n_deleted.append(res1)
        test_acc_avg.append(res2)

    plt.figure(1)
    plt.axis([0,250,0,9000])
    plt.subplot(111)

    new_plot = []

    for i in range(0,len(n_deleted)):
        # print(test_acc_avg[i])
        # print(n_deleted[i])
        new = (test_acc_avg[i][0][0:8], n_deleted[i][1][0:8])

        new[0].insert(0,0) 
        new[1].insert(0,0)
        new_plot.append(new)
        # print(new)
        # print("---")
        # quit()
             
    

    plt.plot(*new_plot[0], dashes=[4, 2], color='#9467bd')
    plt.plot(*new_plot[1], color='#1f77b4')
    plt.plot(*new_plot[2], dashes=[6, 2], color='#17becf')

    plt.legend(legden,
           loc='center right')
    plt.savefig('results/CNN_MR_N_DEL.png' , dpi=600)
    plt.show()