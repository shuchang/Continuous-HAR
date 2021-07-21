%% Part 0: Initialization
close all; clear; clc;
mydir = pwd;
input_dir = uigetdir(fullfile(mydir,'datasets'));
output_dir = fullfile(mydir,'datasets','processed');
%%%%%%%%%%%%%%%%%%%%% discrete %%%%%%%%%%%%%%%%%%%%%
% dir_idx = input_dir(max(strfind(input_dir,'\'))+1);
% output_filename = strcat('discrete',dir_idx,'.mat');
%%%%%%%%%%%%%%%%%%%% continuous %%%%%%%%%%%%%%%%%%%%
output_filename = 'continuous.mat';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output_filepath = fullfile(output_dir,output_filename);
cd(input_dir)
dir_inputfile = dir('*.dat');
input_filenames = {dir_inputfile.name};
num_inputs = length(input_filenames);
for fnum = 1:num_inputs
    filename = char(input_filenames(fnum));
    myname = filename(1:end-4);
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID,'%f');
    fclose(fileID);
    radarData = dataArray{1};
    clearvars fileID dataArray ans;
    %% Part 1: Read the data from input files
    fc = radarData(1); % Center frequency
    Tsweep = radarData(2); % Sweep time in ms
    Tsweep = Tsweep/1000; %then in sec
    NTS = radarData(3); % Number of time samples per sweep
    Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step, for CW, it is 0.
    Data = radarData(5:end); % raw data in I+j*Q format
    fs = NTS/Tsweep; % sampling frequency ADC
    record_length = size(Data,1)/fs; % length of recording in s
    nc = record_length/Tsweep; % number of chirps
    %% Part 2: Reshape data into chirps and do range FFT (1st FFT)
    Data_time=reshape(Data, [NTS nc]);
    % win = ones(NTS,size(Data_time,2));
    win = repmat(hamming(NTS),1,size(Data_time,2));
    %Part taken from Ancortek code for FFT and IIR filtering
    tmp = fftshift(fft(Data_time.*win),1);
    Data_range = zeros(NTS/2, size(tmp,2));
    Data_range(1:NTS/2,:) = tmp(NTS/2+1:NTS,:);
    % IIR Notch filter
    ns = nc;
    % ns = oddnumber(size(Data_range,2))-1;
    Data_range_MTI = zeros(size(Data_range,1),ns);
    [b,a] = butter(4, 0.0075, 'high');
    [h, f1] = freqz(b,a,ns);
    for k=1:size(Data_range,1)
        Data_range_MTI(k,1:ns) = filter(b,a,Data_range(k,1:ns));
    end
    freq =(1:NTS/2-1);
    %frequency range(range bin)
    range_axis=freq*3e8/(2*Bw);
    time_axis=linspace(0,record_length,size(Data_range_MTI,2));
    %delete the DC component
    Data_range_MTI=Data_range_MTI(2:size(Data_range_MTI,1),:);
    %% Part 2.5: Visualization of the Range-Time Image
%     figure();
%     colormap(jet);
%     imagesc(time_axis,range_axis,20*log10(abs(Data_range_MTI)));
%     axis xy; ylim([1 23]); xlabel('Time [s]');ylabel('Range [m]');
%     title('Range Profiles after MTI filter');
%     clim = get(gca,'CLim'); set(gca, 'CLim', clim(2)+[-60,0]);% colorbar;
    %% Part 3: Spectrogram processing for 2nd FFT to get Doppler
    % This selects the range bins where we want to calculate the spectrogram
    bin_indl = 1;
    bin_indu = 63;
    %Parameters for spectrograms
    MD.PRF=1/Tsweep;
    MD.TimeWindowLength = 200;
    MD.OverlapFactor = 0.90;
    MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
    MD.Pad_Factor = 4;
    MD.FFTPoints = MD.Pad_Factor*MD.TimeWindowLength;
    MD.DopplerBin=MD.PRF/(MD.FFTPoints);
    MD.DopplerAxis=-MD.PRF/2:MD.DopplerBin:MD.PRF/2-MD.DopplerBin;
    MD.WholeDuration=size(Data_range_MTI,2)/MD.PRF;
    MD.NumSegments=floor((size(Data_range_MTI,2)-MD.TimeWindowLength)/...
        floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));

    %Method 2 - SUM OF RANGE BINS
    Data_spec_MTI2=0;
    for RBin=bin_indl:1:bin_indu
        Data_MTI_temp = fftshift(spectrogram(Data_range_MTI(RBin,:),...
            MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
        Data_spec_MTI2=Data_spec_MTI2+Data_MTI_temp;
    end
    MD.TimeAxis=linspace(0,MD.WholeDuration,size(Data_spec_MTI2,2));
    % Normalise and plot micro-Doppler
    Data_spec_MTI2= flipud(Data_spec_MTI2./max(Data_spec_MTI2(:)));
     %% Part 3.5: Visualization of the Doppler-Time image
%     figure();
%     colorbar; colormap('jet');
%     imagesc(MD.TimeAxis,MD.DopplerAxis,20*log10(abs(Data_spec_MTI2)),[-45 0]);
%     axis xy; ylim([-150 150]); xlabel('Time [s]'); ylabel('Doppler [Hz]');
    %% Part 4: Store the complex data and labels
    Range{fnum,:,:} = Data_range_MTI;
    Doppler{fnum,:,:} = Data_spec_MTI2;
%%%%%%%%%%%%%%%%%%%%% discrete %%%%%%%%%%%%%%%%%%%%%
%     Label(fnum) = str2double(myname(7));
end
% Label = Label';
%%%%%%%%%%%%%%%%%%%% continuous %%%%%%%%%%%%%%%%%%%%
Label = load('Label.mat');
Label = Label.UpdatedLabel(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 5: Save data
save(output_filepath,'Range');
save(output_filepath,'Doppler','-append');
save(output_filepath,'Label','-append');
cd(mydir)
