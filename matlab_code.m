% Waste Classification System using Image Processing in MATLAB

% Load the images
img1 = imread('Plastic.jpeg'); %add ur image file name
img2 = imread('paperr.jpeg'); %add ur image file name
img3 = imread('galsss.jpeg'); %add ur image file name
img4 = imread('Metal.jpeg'); %add ur image file name

% Convert images to grayscale
gray_img1 = rgb2gray(img1);
gray_img2 = rgb2gray(img2);
gray_img3 = rgb2gray(img3);
gray_img4 = rgb2gray(img4);

% Apply median filtering to remove noise
filtered_img1 = medfilt2(gray_img1, [3 3]);
filtered_img2 = medfilt2(gray_img2, [3 3]);
filtered_img3 = medfilt2(gray_img3, [3 3]);
filtered_img4 = medfilt2(gray_img4, [3 3]);

% Normalize the images
mat2gray_img1 = mat2gray(filtered_img1);
mat2gray_img2 = mat2gray(filtered_img2);
mat2gray_img3 = mat2gray(filtered_img3);
mat2gray_img4 = mat2gray(filtered_img4);

% Extract features (color and texture)
features = zeros(4, 15);  % Adjusted for additional texture features
features(1, :) = extract_features(mat2gray_img1);
features(2, :) = extract_features(mat2gray_img2);
features(3, :) = extract_features(mat2gray_img3);
features(4, :) = extract_features(mat2gray_img4);

% Print features for debugging
disp('Features for each class:');
disp(features);

% Define the waste labels
waste_labels = [1; 2; 3; 4];  % 1: Plastic, 2: Paper, 3: Glass, 4: Metal

% Train a multi-class classifier
svm_model = fitcecoc(features, waste_labels, 'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus'));

% Classify a new image
new_img = imread('galsss.jpeg');  % Example: new image to classify         %give the image file of ur that is saved in the filepath to check which type of image it is in the output
new_gray_img = rgb2gray(new_img);
new_filtered_img = medfilt2(new_gray_img, [3 3]);
new_mat2gray_img = mat2gray(new_filtered_img);
new_features = zeros(1, 15);  % Adjusted for additional features
new_features(1:15) = extract_features(new_mat2gray_img);

% Data Augmentation
% Apply rotation
rotated_img = imrotate(new_mat2gray_img, 45);
rotated_features = extract_features(rotated_img);
new_features = [new_features; rotated_features];

% Apply flipping
flipped_img = fliplr(new_mat2gray_img);
flipped_features = extract_features(flipped_img);
new_features = [new_features; flipped_features];

% Apply scaling
scaled_img = imresize(new_mat2gray_img, 0.5);
scaled_features = extract_features(scaled_img);
new_features = [new_features; scaled_features];

% Apply cropping (ensure the crop region is valid)
cropped_img = imcrop(new_mat2gray_img, [10 10 50 50]);
cropped_features = extract_features(cropped_img);
new_features = [new_features; cropped_features];

% Apply color jittering (note: add noise carefully)
jittered_img = imnoise(new_mat2gray_img, 'gaussian', 0.1);
jittered_features = extract_features(jittered_img);
new_features = [new_features; jittered_features];

% Aggregate features for prediction
new_features_combined = mean(new_features);  % Use mean of features for prediction

% Predict the label
predicted_label = predict(svm_model, new_features_combined);

% Display the predicted label
disp(['Predicted label: ', num2str(predicted_label)]);

% Display the corresponding image
figure;
switch predicted_label
    case 1
        imshow(img1); title('Plastic');
    case 2
        imshow(img2); title('Paper');
    case 3
        imshow(img3); title('Glass');
    case 4
        imshow(img4); title('Metal');
end

% Evaluate the model with a confusion matrix
% Replace this with your actual true labels for validation images
true_labels = [1; 2; 3; 4];  % Update with actual labels for your validation images
predicted_labels = predict(svm_model, features);  % Predict labels for training data

% Calculate confusion matrix
cm = confusionmat(true_labels, predicted_labels);
disp('Confusion Matrix:');
disp(cm);

% Calculate classification metrics
accuracy = sum(diag(cm)) / sum(cm(:));  % Overall accuracy
precision = diag(cm) ./ sum(cm, 2);      % Precision for each class
recall = diag(cm) ./ sum(cm, 1)';        % Recall for each class
f1_score = 2 * (precision .* recall) ./ (precision + recall); % F1 Score for each class

% Display results
disp(['Overall Accuracy: ', num2str(accuracy)]);
disp('Precision for each class:');
disp(precision);
disp('Recall for each class:');
disp(recall);
disp('F1 Score for each class:');
disp(f1_score);

% Function to extract features from an image
function features = extract_features(img)
    features = zeros(1, 15);  % Adjusted for 15 features
    features(1) = mean(img(:));  % Mean
    features(2) = std(img(:));   % Standard Deviation
    [hist, ~] = imhist(img);
    hist_normalized = hist ./ sum(hist);
    features(3:9) = hist_normalized(1:7);  % First 7 bins of histogram
    
    % Texture features using GLCM
    glcm = graycomatrix(img, 'Offset', [0 1]);  % Create GLCM
    stats = graycoprops(glcm);
    features(10) = stats.Contrast;  % Contrast
    features(11) = stats.Correlation; % Correlation
    features(12) = stats.Energy;      % Energy
    features(13) = stats.Homogeneity;  % Homogeneity
    features(14) = mean2(img);         % Mean intensity
    features(15) = std2(img);          % Standard deviation of intensity
end
