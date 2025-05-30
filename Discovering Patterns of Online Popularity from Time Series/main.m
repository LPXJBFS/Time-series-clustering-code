clear all
clc

testorgnames = {'ACSF1','Adiac','AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','ArrowHead','BME','Beef','BeetleFly','BirdChicken','CBF','Car','Chinatown','ChlorineConcentration','CinCECGTorso','Coffee','Computers','CricketX','CricketY','CricketZ','Crop','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','DodgerLoopDay','DodgerLoopGame','DodgerLoopWeekend','ECG200','ECG5000','ECGFiveDays','EOGHorizontalSignal','EOGVerticalSignal','Earthquakes','ElectricDevices','EthanolLevel','FaceAll','FaceFour','FacesUCR','FiftyWords','Fish','FordA','FordB','FreezerRegularTrain','FreezerSmallTrain','Fungi','GestureMidAirD1','GestureMidAirD2','GestureMidAirD3','GesturePebbleZ1','GesturePebbleZ2','GunPointAgeSpan','GunPointMaleVersusFemale','GunPointOldVersusYoung','GunPoint','Ham','HandOutlines','Haptics','Herring','HouseTwenty','InlineSkate','InsectEPGRegularTrain','InsectEPGSmallTrain','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lightning2','Lightning7','Mallat','Meat','MedicalImages','MelbournePedestrian','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MixedShapesRegularTrain','MixedShapesSmallTrain','MoteStrain','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','PigAirwayPressure','PigArtPressure','PigCVP','Plane','PowerCons','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','Rock','ScreenType','SemgHandGenderCh2','SemgHandMovementCh2','SemgHandSubjectCh2','ShapeletSim','ShapesAll','SmallKitchenAppliances','SmoothSubspace','SonyAIBORobotSurface1','SonyAIBORobotSurface2','StarLightCurves','Strawberry','SwedishLeaf','Symbols','SyntheticControl','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','TwoPatterns','UMD','UWaveGestureLibraryAll','UWaveGestureLibraryX','UWaveGestureLibraryY','UWaveGestureLibraryZ','Wafer','Wine','WordSynonyms','WormsTwoClass','Worms','Yoga'};

path = 'D:\11\撞南墙的不定时记录\贝法\Xkmean_MATLAB\base_line\datasets\';  % 文件夹路径

for i = 2 : length(testorgnames)

    disp('*******************************************************************');
    disp('*******************************************************************');
    disp('*******************************************************************');
    disp('*******************************************************************');
    disp(testorgnames{i});
    dataName=testorgnames{i};

    load([path,dataName],'Data','Labels');
    X=Data;
    K=max(Labels);

    [mem, cent, finalNorm, sqe] = multidim_KSC(X, K );
    ar = valid_RandIndex(mem,Labels).AR;
    ri = valid_RandIndex(mem,Labels).RI;
    nmi = val_nmi(mem,Labels);
    F = compute_f(mem,Labels);
    result_path = "D:\11\撞南墙的不定时记录\贝法\Xkmean_MATLAB\05timeseires_baseline\awesome-multivariate-time-series-clustering-algorithms-main\awesome-multivariate-time-series-clustering-algorithms-main\mts-clustering-master\"
    subfolder = "results\"; % 子文件夹
    file_name = "./" + dataName; % 文件名

    % 拼接完整路径
    full_path = fullfile(result_path, subfolder);
    % 检查并创建文件夹
    if ~exist(full_path, 'dir')
        mkdir(full_path);
    end

    % 保存文件
    save(fullfile(full_path, file_name),'-v7.3'); % 假设 clustering_result 是要保存的变量
end
