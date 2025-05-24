clc
clear all
filename = 'EE12.xlsx';
data = xlsread(filename);
data = data(6:end, :);

FB1 = data(:, 46:99);
FB2 = data(:, 99:153);
FB3 = data(:, 153:189);
FB4 = data(:, 189:216);

X0 = data(:, 2);

FB1_dot = [];
FB2_dot = [];
FB3_dot = [];
FB4_dot = [];

FB_CR1 = [];
FB_CR2 = [];
FB_CR3 = [];
FB_CR4 = [];

% Calculate CC and ED for FB1
for i = 1:length(FB1)
    CC = sum(X0.*FB1(i,:))./((sum(X0.^2).*sum(FB1(i,:).^2)).^(1/2));
    ED = sum((FB1(i,:)-X0).^2).^(1/2);
    FB1_dot = [FB1_dot; CC, ED]; % Use semicolon to concatenate rows
end

% Calculate CC_min, CC_max, ED_min, ED_max for FB1
CC_min = 10000;
CC_max = -10000;
ED_min = 10000;
ED_max = -10000;

for i = 1:length(FB1)
    CC = FB1_dot(i, 1);
    ED = FB1_dot(i, 2);
    
    if 1 - CC < CC_min
        CC_min = 1 - CC;
    end
    
    if 1 - CC > CC_max
        CC_max = 1 - CC;
    end
    
    if ED > ED_max
        ED_max = ED;
    end
    
    if ED < ED_min
        ED_min = ED;
    end
end

% Calculate CR for FB1
for i = 1:length(FB1)
    CC = FB1_dot(i, 1);
    ED = FB1_dot(i, 2);
    CR = 12 * ((1 - CC - CC_min)/(CC_max - CC_min) + (ED - ED_min)/(ED_max - ED_min)) * 100;
    FB_CR1(end+1)=CR;
end
L = length(FB1);


