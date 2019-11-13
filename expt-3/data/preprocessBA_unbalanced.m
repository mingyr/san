%% Extract all epochs from each subject, as long as the button press response is correct

clc; clear all; close all;

eeglab;
%% Extrac features and save
codes=[34 % Non-target image onset
    35 % Target image onset
    %     38 % Correct response; after a fixed period after stimulus; may be missing
    %     39 % Incorrect response
    ];
dataDir='E:\Data\Data Collection 1 - VEP Oddball\';
filesB=dir([dataDir 'B*.set']); fsB=512;
filesE=dir([dataDir 'A*.set']); fsA=256;
fFeature=64;
tLimits=[0 .7]; % time interval after stimulus; used in epoching
idsSubjects=1:18; % subjects with >=30 targets
idsGood=[];
for s=1:length(idsSubjects)
    try
        if s<10
            name=['_0' num2str(s) '_VEP.set'];
        else
            name=['_' num2str(s) '_VEP.set'];
        end
        
        %% Process all 64 channels in EEGsB first
        EEGsB=pop_loadset([dataDir 'B' name]);
        EEGsB = pop_select(EEGsB,'channel',1:64); % remove EEG channels
        EEGsB=pop_eegfiltnew(EEGsB,1,50); % Make dataset B the same as E: [.3 43]Hz bandpass
        EEGsB=pop_resample(EEGsB,fFeature); % Resample
        EEGsB=pop_reref(EEGsB,[]); % average re-reference; using common channels will reduce the number of channels
        EEGsB=pop_epoch(EEGsB,mat2cell(codes, ones(1,length(codes))),tLimits);
        EEGsB=pop_rmbase(EEGsB,[]);
        for c=1:EEGsB.nbchan
            channelsB(c)=upper({EEGsB.chanlocs(c).labels});
        end
        subjects(s).channelsB=channelsB;
        numEpochs=length(EEGsB.epoch);
        subjects(s).labelsBall=-ones(numEpochs,1); % all are non-targets; use class labels {-1, 1}
        labelsB=zeros(numEpochs,1);
        for i=1:numEpochs
            a=EEGsB.epoch(i).eventtype(1);
            if isnumeric(a)
                labelsB(i)=a;
            else
                labelsB(i)=cell2mat(a);
            end
        end
        idsTarget=find(labelsB==codes(2));
        subjects(s).labelsBall(idsTarget)=1; % assign target labels
        try
            events=[EEGsB.orig_urevent.type];
        catch error
            events=[EEGsB.urevent.type];
        end
        events(find(events==63):end)=[];
        ids=sort([find(events==codes(1)) find(events==codes(2))]);
        idsWrong=[];
        for i=1:length(ids)-1
            if ~ismember(38,events(ids(i)+1:ids(i+1)-1))
                idsWrong=[idsWrong i];
            end
        end
        if ~ismember(38,events(ids(end)+1:end))
            idsWrong=[idsWrong length(ids)];
        end
        subjects(s).labelsBall(idsWrong)=[];
        subjects(s).dataBall=EEGsB.data;
        subjects(s).dataBall(:,:,idsWrong)=[];
        
        %% Select common channels
        EEGsA=pop_loadset([dataDir 'A' name]);
        EEGsB=pop_loadset([dataDir 'B' name]);
        for c=1:EEGsA.nbchan
            channelsA(c)=upper({EEGsA.chanlocs(c).labels});
        end
        [channels,iB,iA]=intersect(channelsB,channelsA);
        nbchan=length(iB);
        EEGsB.nbchanAll=EEGsB.nbchan; EEGsB.dataAll=EEGsB.data; EEGsB.chanlocsAll=EEGsB.chanlocs;
        EEGsB.nbchan=nbchan; EEGsA.nbchan=nbchan;
        EEGsB.data=EEGsB.data(iB,:);  EEGsA.data=EEGsA.data(iA,:);
        EEGsB.chanlocs=EEGsB.chanlocs(iB); EEGsA.chanlocs=EEGsA.chanlocs(iA);
    catch
        s
        continue;
    end
    
    %% AMB, 1 EKG + 9 EEG
    EEGsA=pop_eegfiltnew(EEGsA,1,50);
    EEGsA=pop_resample(EEGsA,fFeature); % Resample
    EEGsA=pop_reref(EEGsA,[]); % average re-reference; using common channels will reduce the number of channels
    EEGsA=pop_epoch(EEGsA,mat2cell(codes, ones(1,length(codes))),tLimits);
    EEGsA=pop_rmbase(EEGsA,[]);
    subjects(s).channelsA=channelsA;
    numEpochs=length(EEGsA.epoch);
    subjects(s).labelsA=-ones(numEpochs,1); % all are non-targets; use class labels {-1, 1}
    labelsA=zeros(numEpochs,1);
    for i=1:numEpochs
        a=EEGsA.epoch(i).eventtype(1);
        if isnumeric(a)
            labelsA(i)=a;
        else
            labelsA(i)=cell2mat(a);
        end
    end
    idsTarget=labelsA==codes(2);
    subjects(s).labelsA(idsTarget)=1; % assign target labels
    %% ABM did not record button press response
    subjects(s).dataA=EEGsA.data;
    numSelectedA(s)=numEpochs;
    numTargetA(s)=sum(subjects(s).labelsA==1);
    
    %% Biosemi, 64 EEG + 4 EOG
    EEGsB=pop_eegfiltnew(EEGsB,1,50); % Make dataset B the same as E: [.3 43]Hz bandpass
    EEGsB=pop_resample(EEGsB,fFeature); % Resample
    EEGsB=pop_reref(EEGsB,[]); % average re-reference; using common channels will reduce the number of channels
    EEGsB=pop_epoch(EEGsB,mat2cell(codes, ones(1,length(codes))),tLimits);
    EEGsB=pop_rmbase(EEGsB,[]);
    subjects(s).channelsB=channelsB;
    numEpochs=length(EEGsB.epoch);
    subjects(s).labelsB=-ones(numEpochs,1); % all are non-targets; use class labels {-1, 1}
    labelsB=zeros(numEpochs,1);
    for i=1:numEpochs
        a=EEGsB.epoch(i).eventtype(1);
        if isnumeric(a)
            labelsB(i)=a;
        else
            labelsB(i)=cell2mat(a);
        end
    end
    idsTarget=find(labelsB==codes(2));
    subjects(s).labelsB(idsTarget)=1; % assign target labels
    try
        events=[EEGsB.orig_urevent.type];
    catch error
        events=[EEGsB.urevent.type];
    end
    events(find(events==63):end)=[];
    ids=sort([find(events==codes(1)) find(events==codes(2))]);
    idsWrong=[];
    for i=1:length(ids)-1
        if ~ismember(38,events(ids(i)+1:ids(i+1)-1))
            idsWrong=[idsWrong i];
        end
    end
    if ~ismember(38,events(ids(end)+1:end))
        idsWrong=[idsWrong length(ids)];
    end
    subjects(s).labelsB(idsWrong)=[];
    subjects(s).dataB=EEGsB.data;
    subjects(s).dataB(:,:,idsWrong)=[];
    numSelectedB(s)=numEpochs-length(idsWrong);
    numTargetB(s)=sum(subjects(s).labelsB==1);
    
    if min(numSelectedA(s),numSelectedB(s))>=20
        idsGood=[idsGood s];
    end
    
    clc; [numTargetB; numTargetA; numSelectedB; numSelectedA]
end
subjects=subjects(idsGood); numSelectedB=numSelectedB(idsGood); numSelectedA=numSelectedA(idsGood);
numTargetB=numTargetB(idsGood); numTargetA=numTargetA(idsGood);
save BA_Unbalanced.mat subjects numSelectedA numSelectedB numTargetB numTargetA channels;