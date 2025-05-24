clear all

filename = 'TF.xlsx';
data = xlsread(filename);
data = data(:, :);

FB = zeros(4, size(data, 1)); % Initialize a 4xN matrix to store results

i = 1;
num = (i - 1) * 30 + 46;
FB1 = data(:, num:num + 29); 
i = 2;
num = (i - 1) * 30 + 46;
FB2 = data(:, num:num + 29); 
i = 3;
num = (i - 1) * 30 + 46;
FB3 = data(:, num:num + 29); 
i = 4;
num = (i - 1) * 30 + 46;
FB4 = data(:, num:num + 29); 

X0 = data(:, 2);

FB1_dot = [];
FB2_dot = [];
FB3_dot = [];
FB4_dot = [];

for i = 1:30
CC = sum(X0.*FB1(:,i))./((sum(X0.^2).*sum(FB1(:,i).^2)).^(1/2));
ED = sum((FB1(:,i)-X0).^2).^(1/2);
FB1_dot = [FB1_dot; CC, ED]; % Use semicolon to concatenate rows
end

% Repeat for FB2, FB3, and FB4
for i = 1:30
CC = sum(X0.*FB2(:,i))./((sum(X0.^2).*sum(FB2(:,i).^2)).^(1/2));
ED = sum((FB2(:,i)-X0).^2).^(1/2);
FB2_dot = [FB2_dot; CC, ED];
end

for i = 1:30
CC = sum(X0.*FB3(:,i))./((sum(X0.^2).*sum(FB3(:,i).^2)).^(1/2));
ED = sum((FB3(:,i)-X0).^2).^(1/2);
FB3_dot = [FB3_dot; CC, ED];
end

for i = 1:30
CC = sum(X0.*FB4(:,i))./((sum(X0.^2).*sum(FB4(:,i).^2)).^(1/2));
ED = sum((FB4(:,i)-X0).^2).^(1/2);
FB4_dot = [FB4_dot; CC, ED];
end

% Use different colors for each scatter plot
scatter(FB1_dot(:,1), FB1_dot(:,2), 'r');
hold on
scatter(FB2_dot(:,1), FB2_dot(:,2), 'g');
scatter(FB3_dot(:,1), FB3_dot(:,2), 'b');
scatter(FB4_dot(:,1), FB4_dot(:,2), 'm');
xlabel('CC','FontSize',16);
ylabel('ED','FontSize',16);
legend('FB1', 'FB2', 'FB3', 'FB4');
title('Scatter plot of CC and ED for FB1-FB4');

% Modify the font size and weight of axis labels
set(gca,'FontSize',14,'FontWeight','bold');

%{
clc
clear all

filename = 'TF.xlsx';
data = xlsread(filename);
data = data(:, :);

FB = zeros(4, size(data, 1)); % Initialize a 4xN matrix to store results

i = 1;
num = (i - 1) * 30 + 46;
FB1 = data(:, num:num + 29); 
i = 2;
num = (i - 1) * 30 + 46;
FB2 = data(:, num:num + 29); 
i = 3;
num = (i - 1) * 30 + 46;
FB3 = data(:, num:num + 29); 
i = 4;
num = (i - 1) * 30 + 46;
FB4 = data(:, num:num + 29); 

X0 = data(:, 2);

FB1_dot = [];
FB2_dot = [];
FB3_dot = [];
FB4_dot = [];

for i = 1:30
CC = sum(X0.*FB1(:,i))./((sum(X0.^2).*sum(FB1(:,i).^2)).^(1/2));
ED = sum((FB1(:,i)-X0).^2).^(1/2);
FB1_dot = [FB1_dot; CC, ED]; % Use semicolon to concatenate rows
end

% Repeat for FB2, FB3, and FB4
for i = 1:30
CC = sum(X0.*FB2(:,i))./((sum(X0.^2).*sum(FB2(:,i).^2)).^(1/2));
ED = sum((FB2(:,i)-X0).^2).^(1/2);
FB2_dot = [FB2_dot; CC, ED];
end

for i = 1:30
CC = sum(X0.*FB3(:,i))./((sum(X0.^2).*sum(FB3(:,i).^2)).^(1/2));
ED = sum((FB3(:,i)-X0).^2).^(1/2);
FB3_dot = [FB3_dot; CC, ED];
end

for i = 1:30
CC = sum(X0.*FB4(:,i))./((sum(X0.^2).*sum(FB4(:,i).^2)).^(1/2));
ED = sum((FB4(:,i)-X0).^2).^(1/2);
FB4_dot = [FB4_dot; CC, ED];
end

% Use different colors for each scatter plot
scatter(FB1_dot(:,1), FB1_dot(:,2), 'r');
hold on
scatter(FB2_dot(:,1), FB2_dot(:,2), 'g');
scatter(FB3_dot(:,1), FB3_dot(:,2), 'b');
scatter(FB4_dot(:,1), FB4_dot(:,2), 'm');
xlabel('CC','FontSize',16);
ylabel('ED','FontSize',16);
legend('FB1', 'FB2', 'FB3', 'FB4');
title('Scatter plot of CC and ED for FB1-FB4');
%}