import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import os
import scipy.io
import random

# 添加路径
sys.path.insert(1, '../../utils/')
try:
    from utils import fetch_ucr_dataset_online
except ImportError:
    print("无法导入fetch_ucr_dataset_online，可能需要检查路径")

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f"Added directory to path: {parent_dir}")
print(f"Full sys.path: {sys.path}")

# 导入kGraph
try:
    from kgraph import kGraph
except ImportError:
    print("无法导入kGraph，请确保该模块已正确安装或路径已正确设置")
    sys.exit(1)

if __name__ == '__main__':
    # 定义基础路径
    base_path = r'D:\11\撞南墙的不定时记录\贝法\Xkmean_MATLAB\base_line\datasets'
    output_dir = "clustering_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 定义要处理的数据集列表
    datasets = ['ACSF1','Adiac','AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','ArrowHead','BME','Beef','BeetleFly','BirdChicken','CBF','Car','Chinatown','ChlorineConcentration','CinCECGTorso','Coffee','Computers','CricketX','CricketY','CricketZ','Crop','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','DodgerLoopDay','DodgerLoopGame','DodgerLoopWeekend','ECG200','ECG5000','ECGFiveDays','EOGHorizontalSignal','EOGVerticalSignal','Earthquakes','ElectricDevices','EthanolLevel','FaceAll','FaceFour','FacesUCR','FiftyWords','Fish','FordA','FordB','FreezerRegularTrain','FreezerSmallTrain','Fungi','GestureMidAirD1','GestureMidAirD2','GestureMidAirD3','GesturePebbleZ1','GesturePebbleZ2','GunPointAgeSpan','GunPointMaleVersusFemale','GunPointOldVersusYoung','GunPoint','Ham','HandOutlines','Haptics','Herring','HouseTwenty','InlineSkate','InsectEPGRegularTrain','InsectEPGSmallTrain','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lightning2','Lightning7','Mallat','Meat','MedicalImages','MelbournePedestrian','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MixedShapesRegularTrain','MixedShapesSmallTrain','MoteStrain','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','PigAirwayPressure','PigArtPressure','PigCVP','Plane','PowerCons','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','Rock','ScreenType','SemgHandGenderCh2','SemgHandMovementCh2','SemgHandSubjectCh2','ShapeletSim','ShapesAll','SmallKitchenAppliances','SmoothSubspace','SonyAIBORobotSurface1','SonyAIBORobotSurface2','StarLightCurves','Strawberry','SwedishLeaf','Symbols','SyntheticControl','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','TwoPatterns','UMD','UWaveGestureLibraryAll','UWaveGestureLibraryX','UWaveGestureLibraryY','UWaveGestureLibraryZ','Wafer','Wine','WordSynonyms','WormsTwoClass','Worms','Yoga'];
    # 定义用于存储结果的字典
    results = {}
    
    # 循环处理每个数据集
    for dataset in datasets:
        print(f"处理数据集: {dataset}")
        
        # 构建数据集的完整路径
        dataset_path = os.path.join(base_path, f"{dataset}.mat")
        
        try:
            # 加载.mat文件
            mat_data = scipy.io.loadmat(dataset_path)
            
            # 获取Data和Labels
            if 'Data' in mat_data:
                Data = mat_data['Data']
                X = Data  # 为了与kGraph代码保持一致
            else:
                print(f"警告: 数据集 {dataset} 中没有找到 'Data' 变量")
                continue
                
            # 查找标签变量 (可能是'Labels'或其他变量名)
            label_var = None
            for var in ['Labels', 'Lables', 'labels', 'Target', 'target', 'Class', 'class']:
                if var in mat_data:
                    label_var = var
                    break
                    
            if label_var:
                true_labels = mat_data[label_var]
                # 确保标签是一维数组
                if true_labels.ndim > 1:
                    true_labels = true_labels.flatten()
                y = true_labels  # 为了与kGraph代码保持一致
            else:
                print(f"警告: 数据集 {dataset} 中没有找到标签变量")
                continue
            
            # 查找聚类数量 (使用真实标签的唯一值数量)
            n_clusters = len(np.unique(true_labels))
            
            # 对每个数据集重复10次聚类
            for i in range(1, 2):
                print(f"  第 {i} 次运行")
                
                try:
                    # 设置随机种子以确保可重复性 (在不传递给kGraph的情况下)
                    np.random.seed(i)
                    random.seed(i)
                    
                    # 执行 kGraph 聚类 - 不使用random_state参数
                    clf = kGraph(n_clusters=n_clusters, n_lengths=10, n_jobs=4)
                    clf.fit(X)
                    
                    # 计算ARI得分
                    ari_score = adjusted_rand_score(clf.labels_, y)
                    
                    # 输出聚类标签、最优子序列长度、ARI得分
                    print(f"    聚类标签数量: {len(clf.labels_)}")
                    print(f"    聚类标签:{clf.labels_}")
                    print(f"    ARI 得分: {ari_score:.4f}")
                    
                    # 存储结果
                    result_key = f"{dataset}_{i}"
                    results[result_key] = {
                        'predicted_labels': clf.labels_,
                        'optimal_length': clf.optimal_length,
                        'ARI': ari_score
                    }
                    # 保存聚类结果到txt文件
                    output_file = os.path.join(output_dir, f"{dataset}.txt")
                    np.savetxt(output_file, clf.labels_, fmt='%d')
                    print(f"已保存 {dataset} 的聚类结果到 {output_file}")
                    
               
                except Exception as run_error:
                    print(f"    第 {i} 次运行时出错: {str(run_error)}")
        
        
        except Exception as dataset_error:
            print(f"处理数据集 {dataset} 时出错: {str(dataset_error)}")
    
    # 输出所有结果
    print("\n所有聚类指标结果:")
    for result_key, metrics in results.items():
        print(f"{result_key}:")
        print(f"  ARI: {metrics['ARI']:.4f}")
        print(f"  最优长度: {metrics['optimal_length']}")
    
