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
    source = [  "SS_bjornhox_06-07-18_12:26_MR_w2v_0.54_681f", 
                "SS_bjornhox_06-07-18_12:26_MR_w2v_0.56_8147", 
                "SS_bjornhox_06-07-18_13:01_MR_w2v_0.58_8753"]
    
    
    legden = ["0.54", "0.56", "0.56"]

    path = './results/'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    n_deleted = []
    test_acc_avg = []
        
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
    
        new = (test_acc_avg[i][0][0:15], n_deleted[i][1][0:15])

        new[0].insert(0,0) 
        new[1].insert(0,0)
        new_plot.append(new)
        
    

    plt.plot(*new_plot[0], dashes=[4, 2], color='#9467bd')
    plt.plot(*new_plot[1], color='#1f77b4')
    plt.plot(*new_plot[2], dashes=[6, 2], color='#17becf')

    plt.legend(legden,
           loc='center right')
    plt.savefig('results/W2V/W2V_MR_N_DEL.png' , dpi=600)
    plt.show()