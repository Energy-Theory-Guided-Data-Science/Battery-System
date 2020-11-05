% File Name: viewData.m
% Author: Benedikt Rzepka, Institute of Data Processing and Electronics (IPE), KIT
% Email: benedikt.rzepka@student.kit.edu
% Exemplary MATLAB script for processing battery data. Published at ACM e-Energy 2019, Phoenix, AZ, US.
% This MATLAB script is divided into two sections: The first one gets the data from the .csv-files and stores it as MATLAB timeseries objects, the second one is for plotting.
% Use the first section for getting the data in your own MATLAB script for further analysis as you like. 

clear;
%%% LOAD DATA FROM FILES %%%
% presuming you have the data in the same directory as this .m-file, in a 'data' folder 

profile = 'profile_-10A_25A_19_11_18'; %select the profile (name of folder in 'data' directory)

%%load current data:
current_file = sprintf('./data/%s/inverter/Inverter_Current.csv', profile);
file = importdata(current_file, ';',4);
t_0 = file.data(1,1); %define first timestamp of current measurement as t_0
currentTS = timeseries(file.data(:,2), file.data(:,1) - t_0); %current data as timeseries object: time in se, current in A

%%load cell voltage and temperature data:
voltage_data_files = './data/%s/cells/Slave_%d_Cell_Voltages.csv';
temperature_data_files = './data/%s/cells/Slave_%d_Cell_Temperatures.csv';
cellVoltageTS = cell(4,11); %cell array to store cell voltage data
cellTemperatureTS = cell(4,11); %cell array to store cell temperature data

%loop through each slave and each cell
for slave = 0:3
	for cellNumber = 0:10
		file = importdata(sprintf(voltage_data_files, profile, slave), ';',4);
		timeData = file.data(:,1) - t_0; %use same t_0 as above
		voltageData = file.data(:,cellNumber+2);
		time = timeData(~isnan(voltageData)); %select all timestamps where selected column (=cell) has data entry, see next line
		voltage = voltageData(~isnan(voltageData)); %delete NaNs from voltage data. NaNs will appear in case a CAN message is not transmitted correctly. Won't occur regularly.
		cellVoltageTS{slave+1, cellNumber+1} = timeseries(voltage, time); %cell voltage data as timeseries object: time in s, voltage in V
        
		file = importdata(sprintf(temperature_data_files, profile, slave), ';',4);
		timeData = file.data(:,1) - t_0; %use same t_0 as above
		temperatureData = file.data(:,cellNumber+2);
		time = timeData(~isnan(temperatureData)); %select all timestamps where selected column (=cell) has data entry, see next line
		temperature = temperatureData(~isnan(temperatureData)); %delete NaNs from temperature data. NaNs will appear in case a CAN message is not transmitted correctly. Won't occur regularly.		
        cellTemperatureTS{slave+1, cellNumber+1} = timeseries(temperature, time); %cell temperature data as timeseries object: time in s, temperature in °C
	end
end

%The data of each cell can now be found as timeseries objects in cellVoltageTS{slave+1, cellNumber+1} resp. cellTemperatureTS{slave+1, cellNumber+1}, the current data in currentTS,
%and can be used for further analysis. In this script we present the plotting of the raw data without further computation.

%%% PLOT DATA %%%

%%plot settings
tmin = 0;
tmax = 1200;
font = 'Linux Libertine';
fontSize = 18;
fig_size = [0 0 700 500];

%%plot current:
current_fig = figure('Position',fig_size, 'Name', 'Current');
plot(currentTS, 'linewidth', 1.2)
grid on;
xlim([tmin tmax]);
ylim([-15 30])
xlabel('Time in s');
ylabel('Current in A');
title('Current from inverter','FontWeight','Normal');
set(gca, 'FontName', font)
set(gca, 'FontSize', fontSize)

%plot cell voltages
cellv_fig = figure('Position',fig_size, 'Name', 'Cell voltages');
for slave = 0:3
	for cellNumber = 0:10
		set(gca, 'colororderindex',  slave+1);
		if cellNumber == 0
			plot(cellVoltageTS{slave+1,cellNumber+1}, 'DisplayName', sprintf('cells of slave %d', slave))
		else
			plot(cellVoltageTS{slave+1,cellNumber+1}, 'HandleVisibility','off')
		end
		hold on;
	end
end

grid on;
xlim([tmin tmax]);
xlabel('Time in s');
ylabel('Cell voltage in V');
title('Voltage response of each battery cell','FontWeight','Normal');
legend('location', 'northwest');
set(gca, 'FontName', font)
set(gca, 'FontSize', fontSize)

%plot cell temperatures
cellT_fig = figure('Position',fig_size, 'Name', 'Cell temperatures');
for slave = 0:3
	for cellNumber = 0:10
		set(gca, 'colororderindex',  slave+1);
		if cellNumber == 0
			plot(cellTemperatureTS{slave+1,cellNumber+1}, 'DisplayName', sprintf('cells of slave %d', slave))
		else
			plot(cellTemperatureTS{slave+1,cellNumber+1}, 'HandleVisibility','off')
		end
		hold on;
	end
end

grid on;
xlim([tmin tmax]);
ylim([18 28])
xlabel('Time in s');
ylabel('Cell temperature in ^{\circ}C');
title('Cell temperature of each battery cell','FontWeight','Normal');
legend('location', 'northwest');
set(gca, 'FontName', font)
set(gca, 'FontSize', fontSize)