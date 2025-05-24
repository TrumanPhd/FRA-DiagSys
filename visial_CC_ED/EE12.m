clc
clear all
filename = 'EE12.xlsx';
data = xlsread(filename);
data = data(2:end,:);


FB1 = data(:, 46:99);
FB2 = data(:, 99:153);
FB3 = data(:, 153:189);
FB4 = data(:, 189:216);

X0 = data(:, 2);

FB1_dot = [];
FB2_dot = [];
FB3_dot = [];
FB4_dot = [];

for i = 1:99-46+1
CC = sum(X0.*FB1(:,i))./((sum(X0.^2).*sum(FB1(:,i).^2)).^(1/2));
ED = sum((FB1(:,i)-X0).^2).^(1/2);
FB1_dot = [FB1_dot; CC, ED]; % Use semicolon to concatenate rows
end

% Repeat for FB2, FB3, and FB4
for i = 1:153-99+1
CC = sum(X0.*FB2(:,i))./((sum(X0.^2).*sum(FB2(:,i).^2)).^(1/2));
ED = sum((FB2(:,i)-X0).^2).^(1/2);
FB2_dot = [FB2_dot; CC, ED];
end

for i = 1:189-153+1
CC = sum(X0.*FB3(:,i))./((sum(X0.^2).*sum(FB3(:,i).^2)).^(1/2));
ED = sum((FB3(:,i)-X0).^2).^(1/2);
FB3_dot = [FB3_dot; CC, ED];
end

for i = 1:216-189+1
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
xlabel('CC');
ylabel('ED');
legend('FB1', 'FB2', 'FB3', 'FB4');
title('Scatter plot of CC and ED for FB1-FB4');