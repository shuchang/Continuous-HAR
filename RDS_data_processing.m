%% Part 0
close all; clear; clc;
mydir = pwd;
% input_filepath = strcat(mydir,'\data');
input_filepath = uigetdir();
output_filepath = strcat(mydir,'\output');
%% Part 1: Read the data from the file you select
input_filefolder = fullfile(input_filepath);% Load all files in this folder
dir_output = dir(fullfile(input_filefolder,'*.dat'));% Store file information in an array
input_filenames = {dir_output.name};% "Filename" is an array that stores all the file names
for fnum = 1:length(input_filenames)
    close all;
    filename = char(input_filenames(fnum));% Extract the required file name
    myname = filename(1:end-4);% the name of the file
    cd(input_filefolder);
    fileID = fopen(filename, 'r');
    dataArray = textscan(fileID, '%f');
    fclose(fileID);% close the file
    cd(mydir);
    radarData = dataArray{1};
    clearvars fileID dataArray ans;
    fc = radarData(1); % Center frequency
    Tsweep = radarData(2); % Sweep time in ms
    Tsweep = Tsweep/1000; %then in sec
    NTS = radarData(3); % Number of time samples per sweep
    Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step, for CW, it is 0.
    Data = radarData(5:end); % raw data in I+j*Q format
    fs = NTS/Tsweep; % sampling frequency ADC
    record_length = size(Data,1)/fs; % length of recording in s
    nc = record_length/Tsweep; % number of chirps
    %% Part 2: Reshape data into chirps (Data_time) and do 1st FFT to get Range-Time
    %Note that also an IIR notch filter is applied to the data - this is done
    %to remove the 0Hz components in the frequency domain that would be
    %associated to static (0Hz Doppler shift) targets
    Data_time = reshape(Data, [NTS nc]);% 128×10000 matrix
    % NTS=128 每次扫描的采样次数； 测量持续时间为120s； nc=10000 扫描次数
    % Do the 1st FFT to convert Data_time to Data_range
    % win = ones(NTS,size(Data_time,2));
    win = hamming(NTS);
    win = repmat(win,[1 size(Data_time,2)]);
    tmp = fftshift(fft(Data_time.*win), 1); % 128x10000 -> 128x10000
    % 在快时间维度(沿列)做10000次DFT得到每个chirp的差拍频率
    Data_range = zeros(NTS/2, size(tmp,2));
    Data_range(1:NTS/2, :) = tmp(NTS/2+1:NTS, :); %64x10000
    % IIR Notch filter
    ns = nc; % number of samples after filtering
    % ns = oddnumber(size(Data_range,2)) - 1;
    Data_range_MTI = zeros(size(Data_range,1), ns);
    [b,a] = butter(4, 0.0075, 'high');
    [h, f1] = freqz(b, a, ns);
    for k = 1:size(Data_range, 1)
        Data_range_MTI(k,1:ns) = filter(b, a, Data_range(k, 1:ns));
    end
    % end of IIR Notch filter
    freq = (0:ns - 1)*fs/(2*ns);
    range_axis =(freq*3e8*Tsweep)/(2*Bw);
    Data_range_MTI = Data_range_MTI(2:size(Data_range_MTI, 1), :);
    Data_range = Data_range(2:size(Data_range, 1), :);
    %% Part 3: Visualization 1: Range Profiles after MTI
    figure();
    subplot(211);
    colormap(jet), imagesc(20*log10(abs(Data_range)));
    xlabel('No. of Sweeps'), ylabel('Range bins');
    title('Range Profiles without MTI'); clim = get(gca,'CLim'); axis xy;
    set(gca, 'CLim', clim(2)+[-40,0]); colorbar;
    subplot(212);
    colormap(jet), imagesc(20*log10(abs(Data_range_MTI)));
    xlabel('No. of Sweeps'), ylabel('Range bins');
    title('Range Profiles with MTI'); clim = get(gca,'CLim'); axis xy;
    set(gca, 'CLim', clim(2)+[-40,0]); colorbar;
    %% Part 4: Spectrogram processing for 2nd FFT to get Doppler
    window_size = 200;
    padding_factor = 0.95;
    forward_distance = window_size*padding_factor;
    num_of_win = floor((size(Data_range_MTI,2)-window_size)/forward_distance);
    for winidx=1:num_of_win % 51个window
        close all;
        Data_range_MTI_win=Data_range_MTI(:,(winidx-1)*forward_distance+1 ...
            :(winidx-1)*forward_distance+window_size); % 63x200
        %(n-1)*forward_distance+1:(n-1)*forward_distance+window_size
        % Do the 2nd FFT to convert range-time to range_dopper
        Range_doppler = fftshift(fft(Data_range_MTI_win,[],2),2);
        %% Part 5: Visualization 2: single Range-Doppler image
%         figure();
%         colormap(jet), imagesc(20*log10(abs(Range_doppler)));
%         xlabel('Doppler[Hz]'), ylabel('Range bins');
%         title('Range-Doppler Image'); clim = get(gca,'CLim');colorbar;
%         axis xy; set(gca, 'CLim', clim(2)+[-45,0]);
        %% Part 6: Target Detection using CFAR detector
        % A detection is declared when an image cell value exceeds a
        % threshod, which is set to a multiple of the image noise power.
        % Initializaion of CFAR parameter
        Pfa= 0.35; % the desired probability of false alarm
        TrainingBandSize=[2,2]; %The number of rows and columns of the
        % training band cells on each side of the CUT cell
        GuardBandSize=[1,1]; % The number of rows and columns of the guard
        % band cells on each side of the CUT cell
        
        % initialize the CA-CFAR detector
        [Row,Col]=size(Range_doppler);
        detector = phased.CFARDetector2D('TrainingBandSize',TrainingBandSize, ...
            'ThresholdFactor','Auto','GuardBandSize',GuardBandSize, ...
            'ProbabilityFalseAlarm',Pfa,'Method','CA','ThresholdOutputPort',true);
        Ngc = detector.GuardBandSize(2); % number of guardband columns
        Ngr = detector.GuardBandSize(1); % number of guardband rows
        Ntc = detector.TrainingBandSize(2); % number of trainingband columns
        Ntr = detector.TrainingBandSize(1); % number of trainingband rows
        cutidx = [];
        % 有D个测试窗口就应该有D个二位数组，cutidx是一个2-by-D 的矩阵，用于储存CUT的坐标
        % If there are D test Windows, there should be D 2-bit arrays for saving coordinate information.
        % The parameter "Cutidx" is a 2-by-d matrix that stores the CUT coordinates
        colstart = Ntc + Ngc + 1; % 1+3=4
        colend = Col - (Ntc + Ngc); % 200-3=197
        rowstart = Ntr + Ngr + 1; % 1+3=4
        rowend = Row - (Ntr + Ngr); % 63-3=60
        for m = colstart:colend % This loop determines the coordinate of the CFAR target
            for n = rowstart:rowend
                cutidx = [cutidx,[n;m]]; % 'cutidx' store the target matrix
            end
        end
        %% Part 7: Display of the CUT cells
%         ncutcells = size(cutidx,2);
%         cutimage = zeros(Row,Col);
%         for k = 1:ncutcells
%             cutimage(cutidx(1,k),cutidx(2,k)) = 1;
%         end
%         figure();
%         imagesc(cutimage);
%         axis equal;
        %% Part 8: Remove noise in RD image with CFAR detection results
        Range_doppler_dB = 20*log10(abs(Range_doppler));
        [Y,th] = step(detector,Range_doppler_dB,cutidx); %performs 2-D CFAR
        % detection on input image data, Y contains the detection results
        % for the CUT cells. th is the detection threshold
        Logical_matrix = reshape(Y,[rowend-rowstart+1,colend-colstart+1]);
        filter_matrix = zeros(Row,Col);
        filter_matrix(rowstart:rowend,colstart:colend) = Logical_matrix;
        Range_doppler_dB = filter_matrix.*Range_doppler_dB;
        %% Part 9: Visualization 3: Range Doppler after CFAR
%         figure();
%         subplot(211);
%         colormap(jet), imagesc(flipud(20*log10(abs(Range_doppler))));
%         xlabel('Doppler[Hz]'), ylabel('Range bins');
%         title('Range-Doppler before CFAR detection'); clim = get(gca,'CLim');
%         axis xy; set(gca, 'CLim', clim(2)+[-45,0]);colorbar;
%         subplot(212);
%         colormap(jet), imagesc(flipud(Range_doppler_dB));
%         xlabel('Doppler[Hz]'), ylabel('Range bins');
%         title('Range-Doppler after CFAR detection'); clim = get(gca,'CLim');
%         axis xy; set(gca, 'CLim', clim(2)+[-40,0]); colorbar;
        %% Part 10: RD Sequence GENERATION
        % row: Range; column: Doppler; layer: Time (window)
        if fnum == 1
            if winidx == 1
                RD_sequence = zeros(rowend-rowstart+1, colend-colstart+1, ...
                    num_of_win);
            end
        end
        RD_sequence(:,:,winidx) = flipud(Range_doppler_dB(rowstart:rowend,...
            colstart:colend));
    end
    %% Part 11: Parameter initialization for isosurface extraction
    isovalue = 110;% Threshold value setting
    [Range,Doppler,Slow_time] = size(RD_sequence);
    [x,y,z] = meshgrid(1:Doppler,1:Range,1:Slow_time);
    [faces,vertex] = isosurface(x,y,z,RD_sequence,isovalue);
    % Gernerat the vertex and the faces of the graph
    %% Part 12: Data Visualization Check 3 (For the construction of 3D model)
    figure();
    p = patch(isosurface(x,y,z,RD_sequence,isovalue));% Compute normal vector
    isonormals(x,y,z,RD_sequence,p);
    p.FaceColor = 'red';
    p.EdgeColor = 'none';
    daspect([1 1 1])% Control the proportion of coordinate axis unit length
    view(3);
    axis tight
    camlight
    lighting gouraud
    xlabel('Doppler');ylabel('Range');zlabel('Slow time');
    title('Range-Doppler Surface');
    %% Part 13: Final step
    ptCloud = pointCloud(vertex);
    figure();
    pcshow(ptCloud); % Visulization process
    xlabel('Doppler');ylabel('Range');zlabel('Slow time');
    title('Range-Doppler-time Point Clouds');
%     stepsize=0.01;
%     for gridStep=0.001:stepsize:1.501 % approach the beat threshold
%         ptCloudA = pcdownsample(ptCloud,'gridAverage',gridStep);
%         stop_point=length(ptCloudA.Location);
%         if stop_point<=600
%             break
%         elseif stop_point<=650
%             stepsize=0.005;
%         end
%     end
%     Result=zeros(512,3);
%     select=randperm(stop_point,512);
%     for K=1:512
%         Result(K,:)=ptCloudA.Location(select(1,K),:);
%     end
%     Result=sortrows(Result,1);
%     
%     figure(); % The processed image
%     pcshow(ptCloudA);% Visulization process
%     cd(output_filepath);
%     pcwrite(ptCloud,strcat(myname,'.ply'),'Encoding','ascii');
%     cd(mydir);% Switch back to the original folder (direction)

end
