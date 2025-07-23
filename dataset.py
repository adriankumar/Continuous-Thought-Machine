import numpy as np

#generates single arithmetic progression sample with random start and step
def generate_arithmetic_sample():
    start = np.random.randint(0, 51)  #range 0 to 50
    step = np.random.randint(-10, 11)  #range -10 to +10
    while step == 0:  #exclude step size 0
        step = np.random.randint(-10, 11)
    
    #generate sequence of length 5
    sequence = [start + i * step for i in range(5)]
    
    #input is first 4 numbers, label is 5th number
    input_seq = sequence[:4]
    label = sequence[4]
    
    return input_seq, label

#creates complete dataset with specified number of samples
def create_arithmetic_dataset(num_samples):
    inputs = []
    labels = []
    
    for _ in range(num_samples):
        input_seq, label = generate_arithmetic_sample()
        inputs.append(input_seq)
        labels.append(label)
    
    return np.array(inputs), np.array(labels)

#generates and saves training and test datasets
def generate_datasets():
    #generate training set
    train_inputs, train_labels = create_arithmetic_dataset(1000)
    
    #generate test set
    test_inputs, test_labels = create_arithmetic_dataset(200)
    
    print(f"training set shape: {train_inputs.shape}, {train_labels.shape}")
    print(f"test set shape: {test_inputs.shape}, {test_labels.shape}")
    
    #show some examples
    print("\nfirst 5 training examples:")
    for i in range(5):
        input_seq = train_inputs[i]
        label = train_labels[i]
        step = input_seq[1] - input_seq[0]  #calculate step for display
        print(f"input: {input_seq} → label: {label} (step: {step})")
    
    return train_inputs, train_labels, test_inputs, test_labels

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = generate_datasets()