%user specified values
hidden_neurons = 3;
epochs = 1000;
eta = 0.01;


% ------- load in the data -------

% XOR data
train_inp = [1 1; 1 0; 0 1; 0 0];
train_out = [1; 0; 0; 1];

% check same number of patterns in each
if size(train_inp,1) ~= size(train_out,1)
    disp('ERROR: data mismatch')
   return 
end    

%read how many patterns
patterns = size(train_inp,1);

%add a bias as an input
bias = ones(patterns,1);
train_inp = [train_inp bias];

%read how many inputs
inputs = size(train_inp,2);

%set initial random weights
weight_input_hidden = (randn(inputs,hidden_neurons) - 0.5)/10;
weight_hidden_output = (randn(1,hidden_neurons) - 0.5)/10;

%do a number of epochs
for iter = 1:epochs
    
    %get the learning rate from the slider
    blr = eta / 10;
    
    %loop through the patterns, selecting randomly
    for j = 1:patterns
        
        %select a random pattern
        patnum = round((rand * patterns) + 0.5);
        if patnum > patterns
            patnum = patterns;
        elseif patnum < 1
            patnum = 1;    
        end
       
        %set the current pattern
        this_pat = train_inp(patnum,:);
        act = train_out(patnum,1);
        
        %calculate the current error for this pattern
        hval = (tanh(this_pat*weight_input_hidden))';
        pred = hval'*weight_hidden_output';
        error = pred - act;

        % adjust weight hidden - output
        delta_HO = error.*blr .*hval;
        weight_hidden_output = weight_hidden_output - delta_HO';

        % adjust the weights input - hidden
        delta_IH= eta.*error.*weight_hidden_output'.*(1-(hval.^2))*this_pat;
        weight_input_hidden = weight_input_hidden - delta_IH';
        
    end
    % -- another epoch finished
    
    %plot overall network error at end of each epoch
    pred = weight_hidden_output*tanh(train_inp*weight_input_hidden)';
    error = pred' - train_out;
    err(iter) =  (sum(error.^2))^0.5;
    
    figure(1);
    plot(err)

    %stop if error is small
    if err(iter) < 0.001
        fprintf('converged at epoch: %d\n',iter);
        break 
    end
       
end


