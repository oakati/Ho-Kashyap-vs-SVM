clc,clear, format shortE; format compact;
%% read datasets
formatSpec = '%f 1:%f 2:%f 3:%f 4:%f 5:%f 6:%f 7:%f 8:%f';
D.patterns.all=[];
T.patterns.all=[];
D.targets.all=[];
T.targets.all=[];
S=[];
S.learning = dir(fullfile(pwd,'Learning','*.txt'));
S.test = dir(fullfile(pwd,'Test','*.txt'));
c = length(S.learning);
for i=1:c
    fileName = fullfile(S.learning(i).folder,S.learning(i).name);
    fileID = fopen(fileName,'r');
    temp = fscanf(fileID,formatSpec,[9 Inf])';

    D.patterns.all = [D.patterns.all; temp(:,2:end)];
    D.targets.all = [D.targets.all; i*temp(:,1)];

    fileName = fullfile(S.test(i).folder,S.test(i).name);
    fileID = fopen(fileName,'r');
    temp = fscanf(fileID,formatSpec,[9 Inf])';

    T.patterns.all = [T.patterns.all; temp(:,2:end)];
    T.targets.all = [T.targets.all; i*temp(:,1)];
end
clearvars fileID fileName formatSpec i temp S
%% extract +ve samples
D.patterns.pos = D.patterns.all(D.targets.all>0,:);
D.targets.pos = D.targets.all(D.targets.all>0,:);
T.patterns.pos = T.patterns.all(T.targets.all>0,:);
T.targets.pos = T.targets.all(T.targets.all>0,:);
%% normalization
D.patterns.pos_n = normalize(D.patterns.pos,'range');
T.patterns.pos_n = normalize(T.patterns.pos,'range');

Uclasses = unique(T.targets.pos);
clearvars D.patterns.all D.targets.all T.patterns.all T.targets.all
%% Ho_Kashyap
params.ho_kashyap = '["Basic",1e3,1e1,1]';
votes.ho_kashyap = repmat(zeros(size(T.targets.pos)),1,4);
a = table();
b = table();
for i = 2:c
    for j = 1:i-1
        col = strcat("w_i=",string(i)," w_j=",string(j));
        m_filter = (D.targets.pos==i)|(D.targets.pos==j);
        train_targets = D.targets.pos(m_filter);
        train_targets(train_targets==j) = 0;
        train_targets(train_targets==i) = 1;
        [test_targets, a.(col), b.(col)] = Ho_Kashyap(D.patterns.pos_n(m_filter,:)',...
            train_targets,...
            T.patterns.pos_n',...
            params.ho_kashyap);
        for k = 1:length(test_targets)
            if test_targets(k) == 1
                votes.ho_kashyap(k,i) = votes.ho_kashyap(k,i) + 1;
            else
                votes.ho_kashyap(k,j) = votes.ho_kashyap(k,j) + 1;
            end
        end
    end
end
[~, predictions.ho_kashyap]=max(votes.ho_kashyap,[],2);
%% ho_kashyap recall, precision, accuracy
for i = 1:c
    metrics.ho_kashyap.tp_plus_fp(i,:) = sum(predictions.ho_kashyap == Uclasses(i));
    metrics.ho_kashyap.tp_plus_fn(i,:) = sum(T.targets.pos == Uclasses(i));
    metrics.ho_kashyap.recall(i,:) = sum((predictions.ho_kashyap == Uclasses(i)) & (T.targets.pos == Uclasses(i)))/metrics.ho_kashyap.tp_plus_fn(i);
    metrics.ho_kashyap.precision(i,:) = sum((predictions.ho_kashyap == Uclasses(i)) & (T.targets.pos == Uclasses(i)))/metrics.ho_kashyap.tp_plus_fp(i);
end
accuracy.ho_kashyap = sum(predictions.ho_kashyap == T.targets.pos)/length(T.targets.pos);
%% SVM one-against-one
params.svm_one_ag_one = '["Gauss" , 1e-1 , "Perceptron" , 1e-2]';
votes.svm_one_ag_one = repmat(zeros(size(T.targets.pos)),1,4);
a_star = cell(c,c-1);
for i = 2:c
    for j = 1:i-1
        m_filter = (D.targets.pos==i)|(D.targets.pos==j);
        train_targets = D.targets.pos(m_filter);
        train_targets(train_targets==j) = 0;
        train_targets(train_targets==i) = 1;
        [test_targets, a_star{i,j}] = SVM(D.patterns.pos_n(m_filter,:)',...
            train_targets',...
            T.patterns.pos_n',...
            params.svm_one_ag_one);
        for k = 1:length(test_targets)
            if test_targets(k) == 1
                votes.svm_one_ag_one(k,i) = votes.svm_one_ag_one(k,i) + 1;
            else
                votes.svm_one_ag_one(k,j) = votes.svm_one_ag_one(k,j) + 1;
            end
        end
    end
end
[~, predictions.svm_one_ag_one]=max(votes.svm_one_ag_one,[],2);
%% SVM one-against-one recall, precision, accuracy
for i = 1:c
    metrics.svm_one_ag_one.tp_plus_fp(i,:) = sum(predictions.svm_one_ag_one == Uclasses(i));
    metrics.svm_one_ag_one.tp_plus_fn(i,:) = sum(T.targets.pos == Uclasses(i));
    metrics.svm_one_ag_one.recall(i,:) = sum((predictions.svm_one_ag_one == Uclasses(i)) & (T.targets.pos == Uclasses(i)))/metrics.svm_one_ag_one.tp_plus_fn(i);
    metrics.svm_one_ag_one.precision(i,:) = sum((predictions.svm_one_ag_one == Uclasses(i)) & (T.targets.pos == Uclasses(i)))/metrics.svm_one_ag_one.tp_plus_fp(i);
end
accuracy.svm_one_ag_one = sum(predictions.svm_one_ag_one == T.targets.pos)/length(T.targets.pos);
%% table conversions
metrics.ho_kashyap = struct2table(metrics.ho_kashyap);
metrics.svm_one_ag_one = struct2table(metrics.svm_one_ag_one);
clearvars i j k m_filter col