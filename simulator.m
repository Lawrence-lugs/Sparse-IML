%% example of using the proposed model to investigate the impact of different parameters

%% steps:
% 1. define and set parameters (detailed illustration of parameters please
%    refer to Figure 2(b) in the paper)
% 2. discretize a gesture and generate times series of hand positions.
%    Consider horizontal and vertical gestures individually.
% 3. calculate the solar cell output current when hand is at a certain
%    position, and then form the gesture pattern. Consider horizontal and 
%    vertical gestures individually.
% 4. vary the parameters and investigate their impacts

clear all 
close all

%% parameter define, defult values
radius_hand = 0.06; % set radius of hand to 6cm
radius_solar_cell = 0.02;  % set radius of solar cell to 2cm
light_intensity = 200;  % set light intensity to 1000lux
hand_position_low = 0.02; % set the minimum distance between solar cell and hand to 2cm
hand_position_high = 0.1; % set the maximum distance between solar cell and hand to 2cm
hand_displacement = hand_position_high - hand_position_low; % displacement of hand movement
hand_move_speed = 0.2; % set the speed of hand move to 0.2m/s
solar_cell_current_density = 7; % current density of solar cell, in mA/cm^2
hand_height = 0.05; % distance between hand and solar cell when performing horizontal gesture


%randomize parameters
radius_hand_dist = normrnd(0.09720,0.01137,[1,100]);
hand_position_low_dist = normrnd(0.02,0.001,[1,100]);
hand_position_high_dist = normrnd(0.1,0.01,[1,100]);
hand_displacement_dist = hand_position_high_dist-hand_position_low_dist;
hand_move_speed_dist = normrnd(0.2,0.01,[1,100]);
hand_height_dist = normrnd(0.05,0.01,[1,100]);

series_out = zeros(500,2000);
for i = 1:100
    gest_time_series = gest_creation_hori('Ges_LeftRight',radius_hand_dist(i),hand_move_speed_dist(i));
    current_time_series = current_calculation_hori(gest_time_series,radius_solar_cell,radius_hand_dist(i),light_intensity,solar_cell_current_density,hand_height_dist(i));
    series_out(i,:) = interp1(linspace(1,length(current_time_series),length(current_time_series)),current_time_series,linspace(1,length(current_time_series),2000));
    gest_time_series = gest_creation_hori('Ges_RightLeft',radius_hand_dist(i),hand_move_speed_dist(i));
    current_time_series = current_calculation_hori(gest_time_series,radius_solar_cell,radius_hand_dist(i),light_intensity,solar_cell_current_density,hand_height_dist(i));
    series_out(i+100,:) = interp1(linspace(1,length(current_time_series),length(current_time_series)),current_time_series,linspace(1,length(current_time_series),2000));
    gest_time_series = gest_creation_vert('Ges_UpDown',hand_position_low_dist(i),hand_position_high_dist(i),hand_move_speed_dist(i));
    current_time_series = current_calculation_vert(gest_time_series,radius_hand_dist(i),radius_solar_cell,light_intensity,solar_cell_current_density);
    series_out(i+200,:) = interp1(linspace(1,length(current_time_series),length(current_time_series)),current_time_series,linspace(1,length(current_time_series),2000));
    gest_time_series = gest_creation_vert('Ges_DownUp',hand_position_low_dist(i),hand_position_high_dist(i),hand_move_speed_dist(i));
    current_time_series = current_calculation_vert(gest_time_series,radius_hand_dist(i),radius_solar_cell,light_intensity,solar_cell_current_density);
    series_out(i+300,:) = interp1(linspace(1,length(current_time_series),length(current_time_series)),current_time_series,linspace(1,length(current_time_series),2000));
    gest_time_series = gest_creation_hori('Ges_Left',radius_hand_dist(i),hand_move_speed_dist(i));
    current_time_series = current_calculation_hori(gest_time_series,radius_solar_cell,radius_hand_dist(i),light_intensity,solar_cell_current_density,hand_height_dist(i));
    series_out(i+400,:) = interp1(linspace(1,length(current_time_series),length(current_time_series)),current_time_series,linspace(1,length(current_time_series),2000));
end

%% plot
figure(1);
hold on;
for i = 1:100
    p=plot(series_out(i,:)); 
    p.Color = 'red';
    p=plot(series_out(100+i,:)); 
    p.Color = 'yellow';
    p=plot(series_out(200+i,:)); 
    p.Color = 'green';
    p=plot(series_out(300+i,:)); 
    p.Color = 'blue';
    p=plot(series_out(400+i,:)); 
    p.Color = 'black';
end
hold off

save('trdata.mat','series_out')









